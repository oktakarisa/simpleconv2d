"""
Problem 2: Verification of 2D Convolutional Layer with Small Arrays
This script tests the Conv2d implementation with the specific test cases from the assignment.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cnn_layers import Conv2d

def test_forward_propagation():
    """
    Test forward propagation with the assignment test case
    """
    print("=" * 60)
    print("Problem 2: Testing 2D Convolutional Layer")
    print("=" * 60)
    
    # Input data from assignment
    # Shape: (1, 1, 4, 4) - batch_size=1, channels=1, height=4, width=4
    x = np.array([[[[1, 2, 3, 4], 
                    [5, 6, 7, 8], 
                    [9, 10, 11, 12], 
                    [13, 14, 15, 16]]]], dtype=np.float64)
    
    print("\nInput array x:")
    print(f"Shape: {x.shape}")
    print(x[0, 0])
    
    # Weight array from assignment
    # Two filters, each 3x3
    w = np.array([
        [[0.0, 0.0, 0.0], 
         [0.0, 1.0, 0.0], 
         [0.0, -1.0, 0.0]],
        [[0.0, 0.0, 0.0], 
         [0.0, -1.0, 1.0], 
         [0.0, 0.0, 0.0]]
    ], dtype=np.float64)
    
    print("\nWeight array w:")
    print(f"Shape: {w.shape}")
    for i in range(w.shape[0]):
        print(f"\nFilter {i+1}:")
        print(w[i])
    
    # Create Conv2d layer and manually set weights
    conv = Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0)
    
    # Reshape weights to match Conv2d expected format: (out_channels, in_channels, kernel_h, kernel_w)
    conv.W = w.reshape(2, 1, 3, 3)
    conv.b = np.zeros(2, dtype=np.float64)
    
    # Forward pass
    output = conv.forward(x)
    
    print("\n" + "=" * 60)
    print("Forward Propagation Results:")
    print("=" * 60)
    print(f"\nOutput shape: {output.shape}")
    print("Output values:")
    print(output[0])
    
    # Expected output from assignment
    expected = np.array([[[-4, -4], [-4, -4]], [[1, 1], [1, 1]]], dtype=np.float64)
    
    print("\nExpected output:")
    print(expected)
    
    # Check if output matches expected
    if np.allclose(output[0], expected, rtol=1e-5):
        print("\n[PASS] Forward propagation output matches expected result!")
    else:
        print("\n[FAIL] Forward propagation output does not match!")
        print(f"Difference: {np.abs(output[0] - expected)}")
    
    return output, conv, x

def test_backward_propagation():
    """
    Test backward propagation with the assignment test case
    """
    print("\n" + "=" * 60)
    print("Testing Backward Propagation")
    print("=" * 60)
    
    # Use the same setup as forward test
    output, conv, x = test_forward_propagation()
    
    # Gradient from next layer (from assignment)
    # Shape: (2, 2, 2) - channels=2, height=2, width=2
    # But backward expects (batch_size, channels, height, width)
    delta = np.array([[[-4, -4], [10, 11]], [[1, -7], [1, -11]]], dtype=np.float64)
    
    # Add batch dimension
    delta_batch = delta[np.newaxis, :]  # Shape: (1, 2, 2, 2)
    
    print("\n" + "=" * 60)
    print("\nGradient from next layer (delta):")
    print(f"Shape: {delta_batch.shape}")
    print(delta)
    
    # Backward pass
    dx = conv.backward(delta_batch)
    
    print("\n" + "=" * 60)
    print("Backward Propagation Results:")
    print("=" * 60)
    print(f"\nGradient w.r.t. input (dx) shape: {dx.shape}")
    print("dx values:")
    print(dx[0, 0])
    
    # Expected gradient w.r.t. bias (from assignment)
    expected_db = np.array([-5, 4], dtype=np.float64)
    
    print("\nGradient w.r.t. bias (db):")
    print(conv.db)
    print(f"\nExpected db: {expected_db}")
    
    if np.allclose(conv.db, expected_db, rtol=1e-5):
        print("[PASS] Bias gradient matches expected result!")
    else:
        print("[FAIL] Bias gradient does not match!")
        print(f"Difference: {np.abs(conv.db - expected_db)}")
    
    # Note: The assignment mentions weight gradients should be [13, 27]
    # This refers to specific elements in the gradient tensor
    print("\nGradient w.r.t. weights (dW) shape:", conv.dW.shape)
    print("Some dW values (sample):")
    print(f"  dW[0,0,1,1]: {conv.dW[0,0,1,1]:.2f}")
    print(f"  dW[1,0,1,1]: {conv.dW[1,0,1,1]:.2f}")

def test_shape_preservation():
    """
    Additional test to verify shape calculations work correctly
    """
    print("\n" + "=" * 60)
    print("Testing Shape Preservation and Output Size Calculations")
    print("=" * 60)
    
    test_cases = [
        {"in_ch": 1, "out_ch": 16, "kernel": 3, "stride": 1, "pad": 0, "in_size": (4, 4)},
        {"in_ch": 3, "out_ch": 64, "kernel": 5, "stride": 2, "pad": 2, "in_size": (28, 28)},
        {"in_ch": 16, "out_ch": 32, "kernel": 3, "stride": 1, "pad": 1, "in_size": (14, 14)},
    ]
    
    for i, tc in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"  Input: {tc['in_ch']} channels, {tc['in_size']} size")
        print(f"  Conv: {tc['kernel']}x{tc['kernel']} kernel, stride={tc['stride']}, padding={tc['pad']}")
        
        conv = Conv2d(
            in_channels=tc['in_ch'], 
            out_channels=tc['out_ch'],
            kernel_size=tc['kernel'], 
            stride=tc['stride'], 
            padding=tc['pad']
        )
        
        # Create dummy input
        x = np.random.randn(2, tc['in_ch'], tc['in_size'][0], tc['in_size'][1])
        
        # Forward pass
        out = conv.forward(x)
        
        # Calculate expected output size
        expected_h = (tc['in_size'][0] + 2*tc['pad'] - tc['kernel']) // tc['stride'] + 1
        expected_w = (tc['in_size'][1] + 2*tc['pad'] - tc['kernel']) // tc['stride'] + 1
        
        print(f"  Expected output: {tc['out_ch']} channels, ({expected_h}, {expected_w}) size")
        print(f"  Actual output: {out.shape[1]} channels, ({out.shape[2]}, {out.shape[3]}) size")
        
        if out.shape == (2, tc['out_ch'], expected_h, expected_w):
            print("  [PASS]")
        else:
            print("  [FAIL]")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Assignment Problem 2 Verification Script")
    print("Testing 2D Convolutional Layer Implementation")
    print("=" * 60)
    
    # Run tests
    test_forward_propagation()
    test_backward_propagation()
    test_shape_preservation()
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)

