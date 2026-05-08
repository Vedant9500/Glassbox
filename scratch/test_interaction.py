import numpy as np
import torch
from glassbox.curve_classifier.curve_classifier_integration import predict_operators

def test_interaction():
    # Create multi-input data with interaction: y = x1 * sin(x2)
    np.random.seed(42)
    N = 1000
    x1 = np.random.uniform(-5, 5, N)
    x2 = np.random.uniform(-5, 5, N)
    x = np.column_stack([x1, x2])
    y = x1 * np.sin(x2)

    predictions = predict_operators(x, y, "models/curve_classifier_wide.pt")
    print("Predictions for y = x1 * sin(x2):")
    for k, v in sorted(predictions.items(), key=lambda item: -item[1]):
        print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    test_interaction()
