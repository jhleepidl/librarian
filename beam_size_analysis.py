#!/usr/bin/env python3
"""
Beam Size Performance Analysis Script
"""

import json
import os
from performance_analysis_hf import VectorDBPerformanceAnalyzer

def main():
    """
    Beam Size Performance Analysis Main Function
    """
    # Configuration
    full_db_dir = "vector_db_base"
    data_dir = "../ToolBench/data/test_instruction"
    hf_model_name = "jhleepidl/librarian"  # Hugging Face model name
    BASELINE_THRESHOLD = 0.55
    
    # Check directory existence
    if not os.path.exists(full_db_dir):
        print(f"Error: Full Vector DB directory '{full_db_dir}' not found!")
        return
    
    # Initialize performance analyzer
    print("Initializing VectorDB Performance Analyzer...")
    analyzer = VectorDBPerformanceAnalyzer(full_db_dir, hf_model_name)
    
    # Load test data
    print("Loading test data...")
    test_data_by_group = analyzer.load_test_data(data_dir)
    
    if not test_data_by_group:
        print("No test data found!")
        return
    
    # Beam Size Performance Analysis
    print(f"\n{'='*60}")
    print("BEAM SIZE PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    
    # Set beam size combinations
    # beam_sizes = [2, 3, 5, 8, 10, 15, 20]  # desired beam size combinations
    beam_sizes = [2, 3, 5]
    print(f"Testing beam sizes: {beam_sizes}")
    print(f"Total queries to analyze: {sum(len(data) for data in test_data_by_group.values())}")
    
    # Run beam size performance analysis
    beam_size_results = analyzer.analyze_beam_search_performance(
        test_data_by_group, 
        beam_sizes=beam_sizes, 
        baseline_threshold=BASELINE_THRESHOLD
    )
    
    # Print results
    analyzer.print_beam_search_results(beam_size_results, num_examples=10)
    
    # Save results
    beam_output_file = "beam_size_performance_analysis.json"
    with open(beam_output_file, 'w') as f:
        json.dump(beam_size_results, f, indent=2, default=str)
    
    print(f"\nBeam size analysis results saved to: {beam_output_file}")
    
    # Additional analysis: Find optimal beam size
    print(f"\n{'='*60}")
    print("OPTIMAL BEAM SIZE ANALYSIS")
    print(f"{'='*60}")
    
    best_beam_size = None
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    
    for beam_size in beam_sizes:
        if beam_size in beam_size_results:
            stats = beam_size_results[beam_size]['overall_stats']
            f1_score = stats['f1_mean']
            precision = stats['precision_mean']
            recall = stats['recall_mean']
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_precision = precision
                best_recall = recall
                best_beam_size = beam_size
    
    if best_beam_size:
        print(f"Optimal beam size: {best_beam_size}")
        print(f"Best F1-Score: {best_f1:.4f}")
        print(f"Corresponding Precision: {best_precision:.4f}")
        print(f"Corresponding Recall: {best_recall:.4f}")
        
        # Calculate performance improvement (compared to lowest beam size)
        if beam_sizes and beam_sizes[0] in beam_size_results:
            baseline_f1 = beam_size_results[beam_sizes[0]]['overall_stats']['f1_mean']
            if baseline_f1 > 0:
                improvement = ((best_f1 - baseline_f1) / baseline_f1) * 100
                print(f"Improvement over beam_size={beam_sizes[0]}: {improvement:+.2f}%")
    
    # Computation time estimation (optional)
    print(f"\n{'='*60}")
    print("COMPUTATION TIME ESTIMATION")
    print(f"{'='*60}")
    
    total_queries = sum(len(data) for data in test_data_by_group.values())
    print(f"Total queries processed: {total_queries}")
    print(f"Beam sizes tested: {len(beam_sizes)}")
    print(f"Total beam search runs: {total_queries * len(beam_sizes)}")
    
    # Estimate average API searches per beam size
    print(f"\nEstimated API searches per query (varies by beam size):")
    for beam_size in beam_sizes:
        if beam_size in beam_size_results:
            # Rough estimation (more complex in reality)
            estimated_searches = beam_size * 3  # assume average 3 iterations
            print(f"  Beam size {beam_size}: ~{estimated_searches} API searches per query")

if __name__ == "__main__":
    main() 