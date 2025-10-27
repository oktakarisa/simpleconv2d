"""
Problem 10: Calculation of Output Size and Number of Parameters

This script calculates the output size and parameter count for the three
convolutional layer configurations specified in the assignment.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cnn_layers import calculate_output_size, calculate_conv_params

def calculate_scenario(scenario_num, input_h, input_w, in_channels, filter_h, filter_w, 
                       out_channels, stride, padding):
    """
    Calculate output size and parameters for a given scenario
    
    Parameters:
    -----------
    scenario_num : int
        Scenario number (1, 2, or 3)
    input_h, input_w : int
        Input dimensions (height, width)
    in_channels : int
        Number of input channels
    filter_h, filter_w : int
        Filter dimensions (height, width)
    out_channels : int
        Number of output channels
    stride : int
        Stride size
    padding : int
        Padding size
    """
    print(f"\nScenario {scenario_num}:")
    print("=" * 60)
    print(f"Input size: {input_h} x {input_w}, {in_channels} channels")
    print(f"Filter size: {filter_h} x {filter_w}, {out_channels} channels")
    print(f"Stride: {stride}")
    print(f"Padding: {padding}")
    
    # Calculate output size
    out_h, out_w = calculate_output_size(
        input_size=(input_h, input_w),
        filter_size=(filter_h, filter_w),
        stride=stride,
        padding=padding
    )
    
    print(f"\nOutput size: {out_h} x {out_w}, {out_channels} channels")
    
    # Calculate number of parameters
    # Weights: out_channels × in_channels × filter_h × filter_w
    # Biases: out_channels
    num_weights = out_channels * in_channels * filter_h * filter_w
    num_biases = out_channels
    total_params = num_weights + num_biases
    
    print(f"\nParameter Calculations:")
    print(f"  Weights: {out_channels} × {in_channels} × {filter_h} × {filter_w} = {num_weights}")
    print(f"  Biases: {out_channels}")
    print(f"  Total parameters: {total_params:,}")
    
    # Memory estimation (assuming float32, 4 bytes per parameter)
    memory_mb = (total_params * 4) / (1024 * 1024)
    print(f"  Estimated memory (float32): {memory_mb:.4f} MB")
    
    return out_h, out_w, total_params

def main():
    print("=" * 60)
    print("Problem 10: Output Size and Parameter Calculations")
    print("=" * 60)
    
    # Scenario 1
    print("\n" + "*" * 60)
    out_h1, out_w1, params1 = calculate_scenario(
        scenario_num=1,
        input_h=144, input_w=144,
        in_channels=3,
        filter_h=3, filter_w=3,
        out_channels=6,
        stride=1,
        padding=0  # None means no padding
    )
    
    # Verification of calculation
    print("\nManual Verification:")
    print(f"  Output height: (144 + 2×0 - 3) / 1 + 1 = {(144 + 2*0 - 3) // 1 + 1}")
    print(f"  Output width: (144 + 2×0 - 3) / 1 + 1 = {(144 + 2*0 - 3) // 1 + 1}")
    
    # Scenario 2
    print("\n" + "*" * 60)
    out_h2, out_w2, params2 = calculate_scenario(
        scenario_num=2,
        input_h=60, input_w=60,
        in_channels=24,
        filter_h=3, filter_w=3,
        out_channels=48,
        stride=1,
        padding=0
    )
    
    print("\nManual Verification:")
    print(f"  Output height: (60 + 2×0 - 3) / 1 + 1 = {(60 + 2*0 - 3) // 1 + 1}")
    print(f"  Output width: (60 + 2×0 - 3) / 1 + 1 = {(60 + 2*0 - 3) // 1 + 1}")
    
    # Scenario 3 - Special case with stride=2
    print("\n" + "*" * 60)
    out_h3, out_w3, params3 = calculate_scenario(
        scenario_num=3,
        input_h=20, input_w=20,
        in_channels=10,
        filter_h=3, filter_w=3,
        out_channels=20,
        stride=2,
        padding=0
    )
    
    print("\nManual Verification:")
    print(f"  Output height: (20 + 2×0 - 3) / 2 + 1 = {(20 + 2*0 - 3) // 2 + 1}")
    print(f"  Output width: (20 + 2×0 - 3) / 2 + 1 = {(20 + 2*0 - 3) // 2 + 1}")
    
    print("\n" + "*" * 60)
    print("\nNote on Scenario 3:")
    print("  The assignment mentions this is a case where convolution doesn't fit evenly.")
    print("  With stride=2 on a 20×20 input with 3×3 kernel:")
    print("    (20 - 3) / 2 = 17 / 2 = 8.5")
    print("  Since we can't have half a position, the output is 9×9.")
    print("  This means the rightmost and bottom edge pixels aren't fully utilized,")
    print("  which is why such configurations are generally avoided in practice.")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"\nScenario 1: {out_h1}×{out_w1} output, {params1:,} parameters")
    print(f"Scenario 2: {out_h2}×{out_w2} output, {params2:,} parameters")
    print(f"Scenario 3: {out_h3}×{out_w3} output, {params3:,} parameters")
    print(f"\nTotal parameters across all scenarios: {params1 + params2 + params3:,}")
    
    # Additional insights
    print("\n" + "=" * 60)
    print("Key Insights:")
    print("=" * 60)
    print("\n1. Parameter Count Independence:")
    print("   The number of parameters depends only on:")
    print("   - Number of input/output channels")
    print("   - Filter size")
    print("   It does NOT depend on input image size!")
    
    print("\n2. Output Size Calculation:")
    print("   Formula: (Input + 2×Padding - Filter) / Stride + 1")
    print("   This determines the spatial dimensions of the output.")
    
    print("\n3. Edge Handling:")
    print("   When the convolution doesn't divide evenly (Scenario 3),")
    print("   frameworks typically ignore the extra pixels, which can")
    print("   lead to information loss at the edges.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

