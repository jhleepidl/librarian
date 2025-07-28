#!/usr/bin/env python3
"""
Test script to track current_embedding changes for a single query
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from compare_model_performance import ModelPerformanceComparator

def test_single_query():
    """Test residual search with a single query"""
    print("🧪 Testing Residual Search with Single Query")
    print("=" * 60)
    
    # Initialize comparator
    comparator = ModelPerformanceComparator()
    
    # Load models and vector database
    if not comparator.load_models():
        print("❌ Failed to load models. Exiting.")
        return
    
    if not comparator.load_vector_database():
        print("❌ Failed to load vector database. Exiting.")
        return
    
    # Test query
    test_query = "I am a fitness enthusiast and I want to buy a fitness tracker. Can you suggest some top-rated fitness trackers available on Amazon along with their features and prices?"
    
    print(f"📝 Test Query: {test_query}")
    print("=" * 60)
    
    # Test greedy residual search (less conservative)
    print("\n🔍 Testing Greedy Residual Search (Less Conservative):")
    predicted_apis_greedy = comparator.iterative_residual_search_greedy(test_query, max_iterations=5, residual_threshold=0.1)
    
    print(f"\n✅ Greedy Search Results:")
    print(f"   - Predicted APIs: {len(predicted_apis_greedy)}")
    for i, api in enumerate(predicted_apis_greedy):
        print(f"   - {i+1}. {api['tool_name']}/{api['api_name']} (Score: {api['similarity_score']:.4f})")
    
    # Test beam search (less conservative)
    print("\n" + "="*60)
    print("\n🔍 Testing Beam Search (Less Conservative):")
    predicted_apis_beam = comparator.iterative_residual_search(test_query, max_iterations=5, residual_threshold=0.1, beam_size=3)
    
    print(f"\n✅ Beam Search Results:")
    print(f"   - Predicted APIs: {len(predicted_apis_beam)}")
    for i, api in enumerate(predicted_apis_beam):
        print(f"   - {i+1}. {api['tool_name']}/{api['api_name']} (Score: {api['similarity_score']:.4f})")

if __name__ == "__main__":
    test_single_query() 