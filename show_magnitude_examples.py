#!/usr/bin/env python3
"""
Show specific examples of actual vs predicted magnitudes
"""

import json
import numpy as np

def load_results():
    """Load verification results"""
    with open('magnitude_verification_detailed_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

def show_examples():
    """Show specific examples of magnitude predictions"""
    results = load_results()
    
    actual = np.array(results['actual_magnitudes'])
    predicted = np.array(results['predicted_magnitudes'])
    queries = results['queries']
    errors = np.array(results['errors'])
    
    print("🔍 Magnitude Model Verification Examples")
    print("=" * 50)
    
    # Find best and worst predictions
    best_idx = np.argmin(errors)
    worst_idx = np.argmax(errors)
    
    # Find examples with high and low magnitudes
    high_mag_idx = np.argmax(actual)
    low_mag_idx = np.argmin(actual)
    
    examples = [
        ("Best Prediction", best_idx),
        ("Worst Prediction", worst_idx),
        ("Highest Actual Magnitude", high_mag_idx),
        ("Lowest Actual Magnitude", low_mag_idx)
    ]
    
    for title, idx in examples:
        print(f"\n📊 {title}")
        print(f"   Query: {queries[idx][:100]}...")
        print(f"   Actual Magnitude: {actual[idx]:.4f}")
        print(f"   Predicted Magnitude: {predicted[idx]:.4f}")
        print(f"   Error: {errors[idx]:.4f}")
        print(f"   Error %: {(errors[idx]/actual[idx]*100):.2f}%")
    
    # Show statistics
    print(f"\n📈 Overall Statistics")
    print(f"   Correlation: {np.corrcoef(actual, predicted)[0,1]:.4f}")
    print(f"   Mean Error: {np.mean(errors):.4f}")
    print(f"   Median Error: {np.median(errors):.4f}")
    print(f"   Error Std: {np.std(errors):.4f}")
    
    # Show magnitude ranges
    print(f"\n📏 Magnitude Ranges")
    print(f"   Actual: {actual.min():.4f} - {actual.max():.4f}")
    print(f"   Predicted: {predicted.min():.4f} - {predicted.max():.4f}")
    
    # Show accuracy by magnitude range
    print(f"\n🎯 Accuracy by Magnitude Range")
    low_mask = actual < np.percentile(actual, 33)
    mid_mask = (actual >= np.percentile(actual, 33)) & (actual < np.percentile(actual, 66))
    high_mask = actual >= np.percentile(actual, 66)
    
    for name, mask in [("Low", low_mask), ("Medium", mid_mask), ("High", high_mask)]:
        if np.any(mask):
            avg_error = np.mean(errors[mask])
            avg_actual = np.mean(actual[mask])
            print(f"   {name} magnitude: {avg_error:.4f} error ({avg_error/avg_actual*100:.2f}%)")

if __name__ == "__main__":
    show_examples() 