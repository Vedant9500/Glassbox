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
    parser.add_argument("--replay-buffer-size", type=int, default=100000, help="Maximum size of the adversarial replay buffer")
    parser.add_argument("--static-data", type=str, default="data/curve_dataset_500k_v2.npz", help="Path to static data for anchor injection")
    parser.add_argument("--anchor-ratio", type=float, default=0.25, help="Fraction of the student batch drawn from static data")
    parser.add_argument("--variance-penalty", type=float, default=0.001, help="Penalty for highly oscillatory/singular Teacher output (Total Variation)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize Teacher
    teacher = TeacherRNN(vocab_size=len(TOKENS), max_len=20).to(device)
    teacher_opt = optim.Adam(teacher.parameters(), lr=args.teacher_lr)
    
    # Move static data loading BEFORE initializing the Student model
    print("\nStarting Adversarial Training Loop")
    print("-" * 50)
    
    # Load static anchor data
    anchor_features = None
    anchor_labels = None
    if args.static_data and Path(args.static_data).exists():
        print(f"Loading static anchor data from {args.static_data}...")
        static_data = np.load(args.static_data, allow_pickle=True)
        anchor_features = static_data["features"]
        anchor_labels = static_data["labels"]
        print(f"  Loaded {len(anchor_features)} static curves for anchoring.")
    else:
        print(f"Warning: Static data not found at {args.static_data}. Anchor injection will be disabled.")
        
    # Pad legacy anchor data to match the current 366 feature extractor dimension
    feature_dim = 366
    n_classes = len(OPERATOR_CLASSES)
    if anchor_features is not None:
         n_classes = anchor_labels.shape[1]
         if anchor_features.shape[1] < feature_dim:
             print(f"Padding anchor features from {anchor_features.shape[1]} to {feature_dim}...")
             pad_width = feature_dim - anchor_features.shape[1]
             anchor_features = np.pad(anchor_features, ((0, 0), (0, pad_width)), mode='constant')
         # We will map anchor_labels to 9 classes safely later if necessary
         
    feature_scaler = None
         
    # 2. Initialize Student
    is_mlp = False
    if args.pretrained_student and Path(args.pretrained_student).exists():
        print(f"Loading pretrained student from {args.pretrained_student}")
        try:
            student = load_classifier(args.pretrained_student, device=str(device))
            # Extract metadata
            from scripts.curve_classifier_integration import _make_cache_key, _resolve_device, _cached_metadata_by_device
            cache_key = _make_cache_key(args.pretrained_student, _resolve_device(str(device)))
            metadata = _cached_metadata_by_device.get(cache_key, {})
            
            if metadata.get('feature_scaler') is not None:
                feature_scaler = metadata['feature_scaler']
                print("  Loaded feature scaler from pretrained student checkpoint.")
                
            if metadata.get('operator_classes') is not None:
                n_classes = len(metadata['operator_classes'])

            model_type = metadata.get('type')
            is_mlp = getattr(student, 'net', None) is not None
        except Exception as e:
            print(f"Failed to load pretrained weights, starting fresh: {e}")
            student = CurveClassifierCNN(n_classes=n_classes, n_features=feature_dim).to(device)
    else:
        print("Initializing fresh Student model.")
        student = CurveClassifierCNN(n_classes=n_classes, n_features=feature_dim).to(device)

    # If no scaler was loaded but we have anchor data, generate one from the anchor data
    if feature_scaler is None and anchor_features is not None:
        print("Computing feature scaler from anchor data...")
        valid_anchor = anchor_features[:, :feature_dim]
        # Ignore NaNs/Infs
        valid_anchor = np.nan_to_num(valid_anchor, posinf=0.0, neginf=0.0)
        feature_scaler = {
            'mean': valid_anchor.mean(axis=0),
            'std': valid_anchor.std(axis=0)
        }
    elif feature_scaler is None:
        print("Notice: Training without feature scaling (no pretrained scaler and no anchor data provided).")
        
    student_opt = optim.Adam(student.parameters(), lr=args.student_lr)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none') # Don't reduce so we can use individual losses as rewards
    
    moving_avg_loss = 0.693 # Natural log of 0.5 (random guessing)
    replay_buffer = deque(maxlen=args.replay_buffer_size)
    
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
        valid_tvs = []
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
                    # Pad features up to the global feature_dim (e.g., 366) to avoid shape mismatch
                    if len(features) < feature_dim:
                         features = np.pad(features, (0, feature_dim - len(features)), mode='constant')

                    labels = get_ground_truth_labels(seq, OPERATOR_CLASSES)
                    # Make sure labels match the required length
                    if len(labels) > n_classes:
                         labels = labels[:n_classes]
                    elif len(labels) < n_classes:
                         labels = np.pad(labels, (0, n_classes - len(labels)), mode='constant')
                    
                    valid_features.append(features)
                    valid_labels.append(labels)
                    valid_indices.append(i)
                    valid_sequence_lengths.append(len(seq))
                    # Calculate Total Variation (TV) as a proxy for unrealistic oscillations or singularities
                    tv = float(np.mean(np.abs(np.diff(y))))
                    valid_tvs.append(tv)
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
        # 3. Student attempts to predict operators using replay buffer (and anchor data)
        # We need a full batch, either purely from replay or mixed.
        min_replay_needed = args.batch_size
        anchor_batch_size = 0
        adv_batch_size = args.batch_size
        
        if anchor_features is not None and args.anchor_ratio > 0:
            anchor_batch_size = int(args.batch_size * args.anchor_ratio)
            adv_batch_size = args.batch_size - anchor_batch_size
            min_replay_needed = adv_batch_size

        if len(replay_buffer) >= min_replay_needed:
            for _ in range(args.student_steps):
                adv_batch = random.sample(replay_buffer, adv_batch_size)
                
                # Extract features and labels
                X_adv = np.array([b[0] for b in adv_batch])
                y_adv = np.array([b[1] for b in adv_batch])
                
                # Mix in anchor data if available
                if anchor_batch_size > 0:
                    indices = np.random.choice(len(anchor_features), size=anchor_batch_size, replace=False)
                    X_anchor = anchor_features[indices]
                    y_anchor = anchor_labels[indices]
                    
                    # Ensure feature dimensions match (pad the shorter one)
                    min_feat_dim = max(X_adv.shape[1], X_anchor.shape[1])
                    if X_adv.shape[1] < min_feat_dim:
                        X_adv = np.pad(X_adv, ((0, 0), (0, min_feat_dim - X_adv.shape[1])), mode='constant')
                    if X_anchor.shape[1] < min_feat_dim:
                        X_anchor = np.pad(X_anchor, ((0, 0), (0, min_feat_dim - X_anchor.shape[1])), mode='constant')
                    
                    # Also pad labels if necessary
                    if y_anchor.shape[1] < y_adv.shape[1]:
                        y_anchor = np.pad(y_anchor, ((0, 0), (0, y_adv.shape[1] - y_anchor.shape[1])), mode='constant')
                    elif y_adv.shape[1] < y_anchor.shape[1]:
                        y_adv = np.pad(y_adv, ((0, 0), (0, y_anchor.shape[1] - y_adv.shape[1])), mode='constant')
                    
                    X_batch_np = np.concatenate([X_adv, X_anchor], axis=0)
                    y_batch_np = np.concatenate([y_adv, y_anchor], axis=0)
                else:
                    X_batch_np = X_adv
                    y_batch_np = y_adv
                
                # Apply scaling
                if feature_scaler is not None:
                    # Pad the scaler if it's smaller, truncate if it's larger
                    scaler_mean = feature_scaler['mean'][:X_batch_np.shape[1]]
                    scaler_std = feature_scaler['std'][:X_batch_np.shape[1]]
                    if len(scaler_mean) < X_batch_np.shape[1]:
                        scaler_mean = np.pad(scaler_mean, (0, X_batch_np.shape[1] - len(scaler_mean)), mode='constant')
                        scaler_std = np.pad(scaler_std, (0, X_batch_np.shape[1] - len(scaler_std)), constant_values=1.0)
                    X_batch_np = (X_batch_np - scaler_mean) / (scaler_std + 1e-8)
                
                # Clean NaNs and clip extreme out-of-distribution values to prevent gradient overflow
                X_batch_np = np.nan_to_num(X_batch_np, nan=0.0, posinf=5.0, neginf=-5.0)
                X_batch_np = np.clip(X_batch_np, -5.0, 5.0)

                X_batch = torch.tensor(X_batch_np, dtype=torch.float32).to(device)
                y_batch = torch.tensor(y_batch_np, dtype=torch.float32).to(device)
                
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
            X_curr_np = np.array(valid_features)
            # Pad if features are somehow shorter than the model expectation
            if X_curr_np.shape[1] < feature_dim:
                 X_curr_np = np.pad(X_curr_np, ((0, 0), (0, feature_dim - X_curr_np.shape[1])), mode='constant')
            # Truncate if they are longer
            if X_curr_np.shape[1] > feature_dim:
                 X_curr_np = X_curr_np[:, :feature_dim]
            
            if feature_scaler is not None:
                scaler_mean = feature_scaler['mean'][:X_curr_np.shape[1]]
                scaler_std = feature_scaler['std'][:X_curr_np.shape[1]]
                if len(scaler_mean) < X_curr_np.shape[1]:
                     scaler_mean = np.pad(scaler_mean, (0, X_curr_np.shape[1] - len(scaler_mean)), mode='constant')
                     scaler_std = np.pad(scaler_std, (0, X_curr_np.shape[1] - len(scaler_std)), constant_values=1.0)
                X_curr_np = (X_curr_np - scaler_mean) / (scaler_std + 1e-8)
            
            # Clean NaNs and clip extreme values for the current batch
            X_curr_np = np.nan_to_num(X_curr_np, nan=0.0, posinf=5.0, neginf=-5.0)
            X_curr_np = np.clip(X_curr_np, -5.0, 5.0)
            
            X_curr = torch.tensor(X_curr_np, dtype=torch.float32).to(device)
            y_curr = torch.tensor(np.array(valid_labels), dtype=torch.float32).to(device)
            lengths_batch = torch.tensor(valid_sequence_lengths, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                logits_curr = student(X_curr)
                student_losses = bce_loss_fn(logits_curr, y_curr).mean(dim=1)
        else:
            student_losses = torch.zeros(0, device=device)
            lengths_batch = torch.zeros(0, device=device)
            valid_tvs = []
        
        # --- Teacher Step ---
        # 4. Calculate Teacher Reward
        # Reward = (Student Loss on this example) - (Complexity Penalty)
        # We detach student_losses so gradients don't flow back through Student to Teacher
        raw_rewards = student_losses.detach()
        
        # Dynamic curriculum: smoother formula (max ~4x at random guessing)
        curriculum_multiplier = 1.0 + 3.0 * max(0.0, (moving_avg_loss - 0.15) / 0.55)
        
        # Penalize for length and high variance/oscillations
        lengths_penalties = (args.complexity_penalty * curriculum_multiplier) * lengths_batch
        
        tv_tensor = torch.tensor(valid_tvs, dtype=torch.float32).to(device) if len(valid_tvs) > 0 else torch.zeros(0, device=device)
        variance_penalties = args.variance_penalty * tv_tensor
        
        penalties = lengths_penalties + variance_penalties
        
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
        'operator_classes': list(OPERATOR_CLASSES.keys())[:n_classes],
        'feature_scaler': feature_scaler,
        'model_type': 'mlp' if is_mlp else 'cnn',
        'model_config': {
            'n_classes': n_classes,
            'n_features': feature_dim,
            'curve_dim': getattr(student, 'curve_dim', 128)
        }
    }
    torch.save(student_checkpoint, "models/adv_student_cnn.pt")
    torch.save(teacher.state_dict(), "models/adv_teacher_rnn.pt")
    print("Saved to models/adv_student_cnn.pt and models/adv_teacher_rnn.pt")

if __name__ == "__main__":
    main()
