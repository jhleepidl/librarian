#!/usr/bin/env python3
"""
Test embedding alignment performance of trained query embedding model
Compare query embeddings with sum of relevant API embeddings
"""

import torch
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from train_query_embedding import QueryEmbeddingModel
import torch.nn.functional as F
from typing import List, Dict, Any
import os

# Paths
BEST_MODEL_PATH = "/home/jhlee/librarian/trained_query_model/best_model.pt"
TOOLBENCH_MODEL_NAME = "ToolBench/ToolBench_IR_bert_based_uncased"
TEST_DATA_PATH = "/home/jhlee/librarian/data/test.json"
EVAL_DATA_PATH = "/home/jhlee/librarian/data/eval.json"

class EmbeddingAlignmentTester:
    """
    Test the alignment between query embeddings and relevant API embeddings
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.query_model = None
        self.api_model = None
        
    def load_models(self):
        """Load trained query model and ToolBench API model"""
        print("📦 Loading models...")
        
        # Load trained query embedding model
        self.query_model = QueryEmbeddingModel()
        if os.path.exists(BEST_MODEL_PATH):
            checkpoint = torch.load(BEST_MODEL_PATH, map_location=self.device)
            self.query_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded trained query model from {BEST_MODEL_PATH}")
        else:
            print(f"⚠️  No trained model found at {BEST_MODEL_PATH}")
            return False
        
        self.query_model = self.query_model.to(self.device)
        self.query_model.eval()
        
        # Load ToolBench API model
        self.api_model = SentenceTransformer(TOOLBENCH_MODEL_NAME)
        self.api_model = self.api_model.to(self.device)
        self.api_model.eval()
        print(f"✅ Loaded ToolBench API model: {TOOLBENCH_MODEL_NAME}")
        
        return True
    
    def format_api_text(self, api: Dict[str, Any]) -> str:
        """Format API info into text string"""
        tool_name = api.get('tool_name', '')
        api_name = api.get('api_name', '')
        api_description = api.get('api_description', '')
        category = api.get('category_name', '')
        method = api.get('method', '')
        
        return f"Tool: {tool_name}, API: {api_name}, Category: {category}, Method: {method}, Description: {api_description}"
    
    def get_query_embedding(self, query: str) -> torch.Tensor:
        """Get query embedding using trained model"""
        tokenizer = self.query_model.base_model.tokenizer
        max_length = self.query_model.base_model.get_max_seq_length()
        
        features = tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        features = {k: v.to(self.device) for k, v in features.items()}
        
        with torch.no_grad():
            embedding = self.query_model(features["input_ids"], features["attention_mask"])
            return embedding[0]  # Remove batch dimension
    
    def get_api_embeddings_sum(self, apis: List[Dict[str, Any]]) -> torch.Tensor:
        """Get sum of normalized API embeddings using ToolBench model"""
        if not apis:
            return torch.zeros(768, device=self.device)
        
        api_texts = [self.format_api_text(api) for api in apis]
        
        with torch.no_grad():
            embeddings = self.api_model.encode(api_texts, convert_to_tensor=True, device=self.device)
            # Normalize embeddings before summing
            normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
            return normalized_embeddings.sum(dim=0)  # Sum normalized API embeddings
    
    def calculate_similarity_metrics(self, query_emb: torch.Tensor, api_sum: torch.Tensor) -> Dict[str, float]:
        """Calculate various similarity metrics"""
        # L2 distance
        l2_distance = torch.norm(query_emb - api_sum, p=2).item()
        
        # Cosine similarity
        cosine_sim = F.cosine_similarity(query_emb.unsqueeze(0), api_sum.unsqueeze(0)).item()
        
        # Euclidean distance
        euclidean_dist = torch.dist(query_emb, api_sum, p=2).item()
        
        # Manhattan distance
        manhattan_dist = torch.norm(query_emb - api_sum, p=1).item()
        
        # Normalized L2 distance (by embedding dimension)
        normalized_l2 = l2_distance / np.sqrt(query_emb.shape[0])
        
        return {
            'l2_distance': l2_distance,
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
            'manhattan_distance': manhattan_dist,
            'normalized_l2_distance': normalized_l2
        }
    
    def test_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Test alignment for a single sample"""
        query = sample['query']
        relevant_apis = sample.get('relevant_apis', [])
        
        # Get embeddings
        query_emb = self.get_query_embedding(query)
        api_sum = self.get_api_embeddings_sum(relevant_apis)
        
        # Calculate metrics
        metrics = self.calculate_similarity_metrics(query_emb, api_sum)
        
        return {
            'query': query,
            'num_relevant_apis': len(relevant_apis),
            'metrics': metrics,
            'query_embedding_norm': torch.norm(query_emb).item(),
            'api_sum_norm': torch.norm(api_sum).item()
        }
    
    def test_dataset(self, data_path: str, max_samples: int = None) -> Dict[str, Any]:
        """Test alignment on entire dataset"""
        print(f"🧪 Testing embedding alignment on {data_path}")
        
        # Load dataset
        with open(data_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        if max_samples:
            samples = samples[:max_samples]
        
        print(f"📊 Testing {len(samples)} samples...")
        
        results = []
        for i, sample in enumerate(samples):
            try:
                result = self.test_single_sample(sample)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(samples)} samples")
                    
            except Exception as e:
                print(f"❌ Error processing sample {i}: {e}")
                continue
        
        # Calculate aggregate statistics
        metrics_list = [r['metrics'] for r in results]
        query_norms = [r['query_embedding_norm'] for r in results]
        api_sum_norms = [r['api_sum_norm'] for r in results]
        norm_ratios = [q/a if a > 0 else 0 for q, a in zip(query_norms, api_sum_norms)]
        
        aggregate_stats = {
            'num_samples': len(results),
            'l2_distance': {
                'mean': np.mean([m['l2_distance'] for m in metrics_list]),
                'std': np.std([m['l2_distance'] for m in metrics_list]),
                'min': np.min([m['l2_distance'] for m in metrics_list]),
                'max': np.max([m['l2_distance'] for m in metrics_list])
            },
            'cosine_similarity': {
                'mean': np.mean([m['cosine_similarity'] for m in metrics_list]),
                'std': np.std([m['cosine_similarity'] for m in metrics_list]),
                'min': np.min([m['cosine_similarity'] for m in metrics_list]),
                'max': np.max([m['cosine_similarity'] for m in metrics_list])
            },
            'normalized_l2_distance': {
                'mean': np.mean([m['normalized_l2_distance'] for m in metrics_list]),
                'std': np.std([m['normalized_l2_distance'] for m in metrics_list])
            },
            'query_norm': {
                'mean': np.mean(query_norms),
                'std': np.std(query_norms),
                'min': np.min(query_norms),
                'max': np.max(query_norms)
            },
            'api_sum_norm': {
                'mean': np.mean(api_sum_norms),
                'std': np.std(api_sum_norms),
                'min': np.min(api_sum_norms),
                'max': np.max(api_sum_norms)
            },
            'norm_ratio': {
                'mean': np.mean(norm_ratios),
                'std': np.std(norm_ratios),
                'min': np.min(norm_ratios),
                'max': np.max(norm_ratios)
            }
        }
        
        return {
            'results': results,
            'aggregate_stats': aggregate_stats
        }
    
    def print_detailed_results(self, results: List[Dict[str, Any]], top_k: int = 10):
        """Print detailed results for top-k best and worst alignments"""
        # Sort by cosine similarity (best first)
        sorted_results = sorted(results, key=lambda x: x['metrics']['cosine_similarity'], reverse=True)
        
        print(f"\n🏆 TOP {top_k} BEST ALIGNMENTS:")
        print("=" * 80)
        for i, result in enumerate(sorted_results[:top_k]):
            print(f"{i+1:2d}. Cosine: {result['metrics']['cosine_similarity']:.4f}, "
                  f"L2: {result['metrics']['l2_distance']:.2f}, "
                  f"APIs: {result['num_relevant_apis']}")
            print(f"    Query: {result['query'][:100]}...")
            print(f"    Norms - Query: {result['query_embedding_norm']:.2f}, API_sum: {result['api_sum_norm']:.2f}")
            print()
        
        print(f"\n💥 TOP {top_k} WORST ALIGNMENTS:")
        print("=" * 80)
        for i, result in enumerate(sorted_results[-top_k:]):
            print(f"{i+1:2d}. Cosine: {result['metrics']['cosine_similarity']:.4f}, "
                  f"L2: {result['metrics']['l2_distance']:.2f}, "
                  f"APIs: {result['num_relevant_apis']}")
            print(f"    Query: {result['query'][:100]}...")
            print(f"    Norms - Query: {result['query_embedding_norm']:.2f}, API_sum: {result['api_sum_norm']:.2f}")
            print()
    
    def print_aggregate_stats(self, stats: Dict[str, Any]):
        """Print aggregate statistics"""
        print(f"\n📊 AGGREGATE STATISTICS:")
        print("=" * 50)
        print(f"Total samples: {stats['num_samples']}")
        print()
        
        print("L2 Distance:")
        print(f"  Mean: {stats['l2_distance']['mean']:.4f} ± {stats['l2_distance']['std']:.4f}")
        print(f"  Range: [{stats['l2_distance']['min']:.4f}, {stats['l2_distance']['max']:.4f}]")
        print()
        
        print("Cosine Similarity:")
        print(f"  Mean: {stats['cosine_similarity']['mean']:.4f} ± {stats['cosine_similarity']['std']:.4f}")
        print(f"  Range: [{stats['cosine_similarity']['min']:.4f}, {stats['cosine_similarity']['max']:.4f}]")
        print()
        
        print("Normalized L2 Distance:")
        print(f"  Mean: {stats['normalized_l2_distance']['mean']:.4f} ± {stats['normalized_l2_distance']['std']:.4f}")
        print()
        
        print("Embedding Norms:")
        print(f"  Query embedding norm - Mean: {stats['query_norm']['mean']:.4f} ± {stats['query_norm']['std']:.4f}")
        print(f"  API sum norm - Mean: {stats['api_sum_norm']['mean']:.4f} ± {stats['api_sum_norm']['std']:.4f}")
        print(f"  Norm ratio (query/api_sum) - Mean: {stats['norm_ratio']['mean']:.4f} ± {stats['norm_ratio']['std']:.4f}")


def main():
    """Main testing function"""
    print("🧪 Query Embedding Alignment Test")
    print("=" * 50)
    
    # Initialize tester
    tester = EmbeddingAlignmentTester()
    
    # Load models
    if not tester.load_models():
        print("❌ Failed to load models. Exiting.")
        return
    
    # Test on evaluation dataset
    if os.path.exists(EVAL_DATA_PATH):
        print(f"\n📊 Testing on evaluation dataset...")
        eval_results = tester.test_dataset(EVAL_DATA_PATH, max_samples=100)
        
        # Print results
        tester.print_aggregate_stats(eval_results['aggregate_stats'])
        tester.print_detailed_results(eval_results['results'], top_k=5)
        
        # Save detailed results
        output_path = "/home/jhlee/librarian/embedding_alignment_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Detailed results saved to: {output_path}")
    
    # Test on test dataset
    if os.path.exists(TEST_DATA_PATH):
        print(f"\n📊 Testing on test dataset...")
        test_results = tester.test_dataset(TEST_DATA_PATH, max_samples=50)
        
        # Print results
        tester.print_aggregate_stats(test_results['aggregate_stats'])
        tester.print_detailed_results(test_results['results'], top_k=5)
    
    print("\n✅ Testing completed!")


if __name__ == "__main__":
    main() 