"""
Adversarial Self-Play for Curve Classifier

This script trains the CurveClassiferCNN (Student) against the TeacherRNN.
The Teacher gets rewarded if the Student gets the operator classification wrong,
while being penalized for generating overly long/complex formulas.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import sys

# Ensure parent directory is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.teacher_generator import TeacherRNN, tokens_to_curve, get_ground_truth_labels, TOKENS
from scripts.curve_classifier_integration import CurveClassifierCNN, load_classifier
from scripts.generate_curve_data import extract_all_features, OPERATOR_CLASSES, apply_noise_augmentation
from collections import deque
import random

def main():
    parser = argparse.ArgumentParser(description="Train Curve Classifier via Adversarial Self-Play")
    parser.add_argument("--epochs", type=int, default=100, help="Number of adversarial epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per epoch")
    parser.add_argument("--teacher-lr", type=float, default=1e-3, help="Teacher learning rate")
    parser.add_argument("--student-lr", type=float, default=1e-3, help="Student learning rate")
    parser.add_argument("--student-steps", type=int, default=1, help="Number of steps Student trains per Teacher batch")
    parser.add_argument("--complexity-penalty", type=float, default=0.005, help="Penalty multiplier for formula length")
    parser.add_argument("--entropy-bonus", type=float, default=0.1, help="Bonus for high entropy to prevent mode collapse")
    parser.add_argument("--pretrained-student", type=str, default="", help="Path to pretrained Curve Classifier (.pt)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize Teacher
    teacher = TeacherRNN(vocab_size=len(TOKENS), max_len=20).to(device)
    teacher_opt = optim.Adam(teacher.parameters(), lr=args.teacher_lr)
    
    # 2. Initialize Student
    if args.pretrained_student and Path(args.pretrained_student).exists():
        print(f"Loading pretrained student from {args.pretrained_student}")
        # Note: Depending on how load_classifier is implemented, you may need to handle this manually 
        # to ensure it's a PyTorch model and extracts the state dict correctly.
        # For simplicity, we initialize a fresh CNN and load weights if possible.
        student = CurveClassifierCNN(n_classes=len(OPERATOR_CLASSES), n_features=366).to(device)
        try:
            checkpoint = torch.load(args.pretrained_student, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                student.load_state_dict(checkpoint['model_state_dict'])
            else:
                student.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Failed to load pretrained weights, starting fresh: {e}")
    else:
        print("Initializing fresh Student model.")
        student = CurveClassifierCNN(n_classes=len(OPERATOR_CLASSES), n_features=366).to(device)
        
    student_opt = optim.Adam(student.parameters(), lr=args.student_lr)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none') # Don't reduce so we can use individual losses as rewards
    
    print("\nStarting Adversarial Training Loop")
    print("-" * 50)
    
    moving_avg_loss = 0.693 # Natural log of 0.5 (random guessing)
    replay_buffer = deque(maxlen=2048)
    
    for epoch in range(args.epochs):
        teacher.train()
        student.train()
        
        # --- Environment Step ---
        # 1. Teacher generates a batch of sequences
        sequences, log_probs, entropies = teacher.sample(batch_size=args.batch_size, device=device)
        
        valid_features = []
        valid_labels = []
        valid_indices = []
        valid_sequence_lengths = []
        formulas = []
        
        # 2. Parse sequences into curves
        for i, seq in enumerate(sequences):
            # Try to build curve
            y, success = tokens_to_curve(seq)
            
            if success:
                try:
                    # Extract standard 366 features with noise like real training data
                    y_noisy = apply_noise_augmentation(y, noise_profile='multi')
                    features = extract_all_features(y_noisy)
                    labels = get_ground_truth_labels(seq, OPERATOR_CLASSES)
                    
                    valid_features.append(features)
                    valid_labels.append(labels)
                    valid_indices.append(i)
                    valid_sequence_lengths.append(len(seq))
                    formulas.append(" ".join(seq))
                except Exception:
                    pass # Feature extraction failed
                    
        # If the Teacher generated pure junk, punish it and continue
        if len(valid_features) == 0:
            print(f"Epoch {epoch+1:03d} | ALL sequences invalid! Punishing teacher.")
            # Give terrible reward to all
            rewards = torch.full((args.batch_size,), -1.0, device=device)
            policy_loss = -(log_probs.sum(dim=1) * rewards).mean()
            entropy_loss = -args.entropy_bonus * entropies.sum(dim=1).mean()
            loss_t = policy_loss + entropy_loss
            
            teacher_opt.zero_grad()
            loss_t.backward()
            teacher_opt.step()
            continue
            
        # Convert valid data to tensors and add to replay buffer
        if len(valid_features) > 0:
            for f, l in zip(valid_features, valid_labels):
                replay_buffer.append((f, l))
        
        # --- Student Step ---
        # 3. Student attempts to predict operators using replay buffer
        if len(replay_buffer) >= args.batch_size:
            for _ in range(args.student_steps):
                batch = random.sample(replay_buffer, args.batch_size)
                X_batch = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(device)
                y_batch = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32).to(device)
                
                logits = student(X_batch)
                
                # Calculate BCE loss per example
                student_losses_buffer = bce_loss_fn(logits, y_batch).mean(dim=1)
                loss_s = student_losses_buffer.mean()
                
                # Update Student weights
                student_opt.zero_grad()
                loss_s.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                student_opt.step()
                
            # Update curriculum learning moving average with the final loss after inner steps
            moving_avg_loss = 0.95 * moving_avg_loss + 0.05 * loss_s.item()
        else:
            loss_s = torch.tensor(moving_avg_loss)
            
        # Get Student's loss on the *current* Teacher batch for rewards
        if len(valid_features) > 0:
            X_curr = torch.tensor(np.array(valid_features), dtype=torch.float32).to(device)
            y_curr = torch.tensor(np.array(valid_labels), dtype=torch.float32).to(device)
            lengths_batch = torch.tensor(valid_sequence_lengths, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                logits_curr = student(X_curr)
                student_losses = bce_loss_fn(logits_curr, y_curr).mean(dim=1)
        else:
            student_losses = torch.zeros(0, device=device)
            lengths_batch = torch.zeros(0, device=device)
        
        # --- Teacher Step ---
        # 4. Calculate Teacher Reward
        # Reward = (Student Loss on this example) - (Complexity Penalty)
        # We detach student_losses so gradients don't flow back through Student to Teacher
        raw_rewards = student_losses.detach()
        
        # Dynamic curriculum: smoother formula (max ~4x at random guessing)
        curriculum_multiplier = 1.0 + 3.0 * max(0.0, (moving_avg_loss - 0.15) / 0.55)
        penalties = (args.complexity_penalty * curriculum_multiplier) * lengths_batch
        
        rewards_valid = raw_rewards - penalties
        
        # Build full reward tensor BEFORE baselining (invalid sequences get -1.0)
        rewards = torch.full((args.batch_size,), -1.0, device=device)
        for idx, r in zip(valid_indices, rewards_valid):
            rewards[idx] = r
            
        # Baseline reward normalization across ALL rewards
        rewards = rewards - rewards.mean()
            
        # PPO/REINFORCE Loss: -log_prob * reward
        # Sum log probs over the sequence length
        seq_log_probs = log_probs.sum(dim=1) 
        policy_loss = -(seq_log_probs * rewards).mean()
        
        # Entropy bonus to prevent collapse to a single formula (like 'x')
        entropy_loss = -args.entropy_bonus * entropies.sum(dim=1).mean()
        
        loss_t = policy_loss + entropy_loss
        
        # Update Teacher weights
        teacher_opt.zero_grad()
        loss_t.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=5.0)
        teacher_opt.step()
        
        # --- Logging ---
        valid_ratio = len(valid_features) / args.batch_size
        print(f"Epoch {epoch+1:03d} | Valid: {valid_ratio*100:.1f}% | S-Loss: {loss_s.item():.4f} (Avg: {moving_avg_loss:.4f}) | T-Loss: {loss_t.item():.4f} | Pen: {curriculum_multiplier:.1f}x")
        
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  Sample target formulas:")
            # Show a few valid formulas to see what the teacher is generating
            for f in formulas[:5]:
                print(f"    {f}")

    print("-" * 50)
    print("Training complete. Saving models...")
    os.makedirs("models", exist_ok=True)
    
    # Save the adversarial models
    student_checkpoint = {
        'model_state_dict': student.state_dict(),
        'operator_classes': list(OPERATOR_CLASSES.keys()),
        'model_type': 'cnn',
        'model_config': {
            'n_classes': len(OPERATOR_CLASSES),
            'n_features': 366,
            'curve_dim': 128
        }
    }
    torch.save(student_checkpoint, "models/adv_student_cnn.pt")
    torch.save(teacher.state_dict(), "models/adv_teacher_rnn.pt")
    print("Saved to models/adv_student_cnn.pt and models/adv_teacher_rnn.pt")

if __name__ == "__main__":
    main()
