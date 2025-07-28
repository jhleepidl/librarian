#!/usr/bin/env python3
"""
Compare performance between ToolBench model and trained magnitude regression model
"""

import sys
import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Set
import logging
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from tqdm import tqdm
import time
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = "/home/jhlee/librarian/trained_magnitude_model/best_model.pt"
TOOLBENCH_MODEL_NAME = "ToolBench/ToolBench_IR_bert_based_uncased"
VECTOR_DB_PATH = os.path.join(BASE_DIR, "toolbench_database")
TEST_INSTRUCTION_DIR = "/home/jhlee/ToolBench/data/test_instruction"


class MagnitudeRegressionModel(torch.nn.Module):
    """
    Model that learns to output embeddings with magnitude matching relevant API embeddings sum
    """
    
    def __init__(self, base_model_name: str = TOOLBENCH_MODEL_NAME):
        super().__init__()
        # Use ToolBench model as base
        self.base_model = SentenceTransformer(base_model_name)
        
        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Add regression output layer to predict magnitude
        # The base model outputs 768-dim embeddings
        self.regression_head = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1)  # Output single value (magnitude)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get base embeddings
        with torch.no_grad():
            base_output = self.base_model({"input_ids": input_ids, "attention_mask": attention_mask})
            base_embeddings = base_output["sentence_embedding"]
        
        # Predict magnitude
        magnitude = self.regression_head(base_embeddings)
        
        return magnitude.squeeze(-1)  # Remove last dimension


class ModelPerformanceComparator:
    """
    Compare performance between ToolBench model and trained magnitude regression model
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.toolbench_model = None
        self.trained_model = None
        self.vector_db = None
        self.api_metadata = []
        
    def load_models(self):
        """Load both ToolBench and trained models"""
        print("📦 Loading models...")
        
        # Load ToolBench model
        try:
            self.toolbench_model = SentenceTransformer(TOOLBENCH_MODEL_NAME)
            self.toolbench_model = self.toolbench_model.to(self.device)
            self.toolbench_model.eval()
            print(f"✅ ToolBench model loaded")
        except Exception as e:
            print(f"❌ Error loading ToolBench model: {e}")
            return False
        
        # Load trained magnitude regression model
        try:
            # Create trained model without loading base model again
            self.trained_model = MagnitudeRegressionModel()
            # Set the base model to the already loaded ToolBench model
            self.trained_model.base_model = self.toolbench_model
            
            if os.path.exists(BEST_MODEL_PATH):
                checkpoint = torch.load(BEST_MODEL_PATH, map_location=self.device, weights_only=False)
                self.trained_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Trained magnitude regression model loaded from {BEST_MODEL_PATH}")
            else:
                print(f"⚠️  No trained model found at {BEST_MODEL_PATH}")
                return False
            
            self.trained_model = self.trained_model.to(self.device)
            self.trained_model.eval()
        except Exception as e:
            print(f"❌ Error loading trained model: {e}")
            return False
        
        return True
    
    def load_vector_database(self):
        """Load vector database"""
        print("📦 Loading vector database...")
        
        try:
            # Load FAISS index
            faiss_index_path = os.path.join(VECTOR_DB_PATH, "faiss_index.bin")
            self.vector_db = faiss.read_index(faiss_index_path)
            print(f"✅ Loaded FAISS index with {self.vector_db.ntotal} vectors")
            
            # Load metadata
            metadata_path = os.path.join(VECTOR_DB_PATH, "api_info.json")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.api_metadata = json.load(f)
            print(f"✅ Loaded metadata for {len(self.api_metadata)} APIs")
            
            return True
        except Exception as e:
            print(f"❌ Error loading vector database: {e}")
            return False
    
    def get_toolbench_embedding(self, query: str) -> np.ndarray:
        """Get query embedding using ToolBench model"""
        with torch.no_grad():
            embedding = self.toolbench_model.encode([query], convert_to_tensor=True, device=self.device)
            return embedding.cpu().numpy()
    
    def get_trained_embedding(self, query: str) -> np.ndarray:
        """Get query embedding using trained model (ToolBench direction + magnitude prediction)"""
        # Get ToolBench embedding for direction
        with torch.no_grad():
            toolbench_output = self.toolbench_model.encode([query], convert_to_tensor=True, device=self.device)
            toolbench_embedding = toolbench_output
        
        # Get magnitude prediction from trained model using its base_model
        tokenized = self.trained_model.base_model.tokenize([query])
        
        # Move to device
        for key in tokenized:
            if isinstance(tokenized[key], torch.Tensor):
                tokenized[key] = tokenized[key].to(self.device)
        
        # Get base embedding from trained model's base_model
        with torch.no_grad():
            base_output = self.trained_model.base_model(tokenized)
            base_embedding = base_output["sentence_embedding"]
        
        # Predict magnitude using trained model's regression head
        predicted_magnitude = self.trained_model.regression_head(base_embedding)
        
        # Normalize ToolBench embedding and scale by predicted magnitude
        normalized_embedding = F.normalize(toolbench_embedding, p=2, dim=-1)
        scaled_embedding = normalized_embedding * predicted_magnitude.unsqueeze(-1)
        
        return scaled_embedding.detach().cpu().numpy()
    
    def search_similar_apis(self, query_embedding: np.ndarray, similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search for similar APIs using FAISS"""
        # Normalize query embedding for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Search in FAISS
        D, I = self.vector_db.search(query_norm.reshape(1, -1), k=100)
        
        similar_apis = []
        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            if idx < len(self.api_metadata):
                api_info = self.api_metadata[idx]
                # For IndexFlatIP, distance is already similarity score
                similarity_score = distance
                
                if similarity_score >= similarity_threshold:
                    similar_apis.append({
                        **api_info,
                        'similarity_score': similarity_score,
                        'rank': i + 1
                    })
        
        return similar_apis
    
    def iterative_residual_search(self, query: str, max_iterations: int = 10, residual_threshold: float = 0.5, beam_size: int = 3) -> List[Dict[str, Any]]:
        """
        Iterative residual-based search using trained model with beam search
        """
        # Initialize beam with original query embedding
        initial_embedding = self.get_trained_embedding(query)
        initial_magnitude = np.linalg.norm(initial_embedding)
        print(f"  Initial query embedding magnitude: {initial_magnitude:.4f}")
        
        # Beam state: (apis, current_embedding, total_similarity_score)
        beam = [([], initial_embedding, 0.0)]
        
        for iteration in range(max_iterations):
            new_beam = []
            
            for apis, current_embedding, total_score in beam:
                # Search for similar APIs
                similar_apis = self.search_similar_apis(current_embedding, similarity_threshold=0.1)
                
                if not similar_apis:
                    new_beam.append((apis, current_embedding, total_score))
                    continue
                
                # Get top beam_size candidates
                top_candidates = similar_apis[:beam_size]
                
                for candidate_api in top_candidates:
                    # Get the API's embedding
                    api_text = f"Tool: {candidate_api['tool_name']}, API: {candidate_api['api_name']}"
                    api_embedding = self.toolbench_model.encode([api_text], convert_to_tensor=True, device=self.device)
                    api_embedding = F.normalize(api_embedding, p=2, dim=-1).cpu().numpy()
                    
                    # Calculate residual
                    residual = current_embedding - api_embedding
                    residual_norm = np.linalg.norm(residual)
                    
                    # Calculate new total score based on residual magnitude (lower residual = better)
                    residual_score = -residual_norm  # Negative so that lower residual = higher score
                    new_total_score = total_score + residual_score
                    
                    # Add to new beam
                    new_apis = apis + [candidate_api]
                    new_beam.append((new_apis, residual, new_total_score))
                    
                    print(f"  Iteration {iteration + 1}: API={candidate_api['tool_name']}/{candidate_api['api_name']}, "
                          f"Similarity={candidate_api['similarity_score']:.4f}, Residual_norm={residual_norm:.4f}, "
                          f"Current_magnitude={np.linalg.norm(current_embedding):.4f}, Total_score={new_total_score:.4f}")
            
            # Keep top beam_size candidates based on total score
            new_beam.sort(key=lambda x: x[2], reverse=True)
            beam = new_beam[:beam_size]
            
            # Check if all residuals are small enough
            all_small_residuals = all(np.linalg.norm(state[1]) < residual_threshold for state in beam)
            if all_small_residuals:
                print(f"  All residual norms below threshold ({residual_threshold}), stopping")
                break
        
        # Return the best candidate (highest total score)
        if beam:
            best_apis, _, best_score = max(beam, key=lambda x: x[2])
            print(f"  Best beam candidate with total score: {best_score:.4f}")
            return best_apis
        else:
            return []
    
    def iterative_residual_search_greedy(self, query: str, max_iterations: int = 10, residual_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Original greedy iterative residual-based search using trained model
        """
        predicted_apis = []
        current_embedding = self.get_trained_embedding(query)
        initial_magnitude = np.linalg.norm(current_embedding)
        print(f"  Initial query embedding magnitude: {initial_magnitude:.4f}")
        
        for iteration in range(max_iterations):
            # Search for most similar API
            similar_apis = self.search_similar_apis(current_embedding, similarity_threshold=0.1)
            
            if not similar_apis:
                break
            
            # Get the most similar API
            best_api = similar_apis[0]
            predicted_apis.append(best_api)
            
            # Get the API's embedding
            api_text = f"Tool: {best_api['tool_name']}, API: {best_api['api_name']}"
            api_embedding = self.toolbench_model.encode([api_text], convert_to_tensor=True, device=self.device)
            api_embedding = F.normalize(api_embedding, p=2, dim=-1).cpu().numpy()
            
            # Calculate residual
            residual = current_embedding - api_embedding
            residual_norm = np.linalg.norm(residual)
            
            print(f"  Iteration {iteration + 1}: API={best_api['tool_name']}/{best_api['api_name']}, "
                  f"Similarity={best_api['similarity_score']:.4f}, Residual_norm={residual_norm:.4f}, "
                  f"Current_magnitude={np.linalg.norm(current_embedding):.4f}")
            
            # Check if residual is small enough
            if residual_norm < residual_threshold:
                print(f"  Residual norm ({residual_norm:.4f}) below threshold ({residual_threshold}), stopping")
                break
            
            # Update current embedding to residual
            current_embedding = residual
        
        return predicted_apis
    
    def evaluate_predictions(self, predicted_apis: List[Dict[str, Any]], relevant_apis: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate predictions against ground truth"""
        if not predicted_apis:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'num_predicted': 0,
                'num_relevant': len(relevant_apis)
            }
        
        # Create sets for comparison
        predicted_set = {(api['tool_name'], api['api_name']) for api in predicted_apis}
        relevant_set = {(api['tool_name'], api['api_name']) for api in relevant_apis}
        
        # Calculate metrics
        intersection = predicted_set.intersection(relevant_set)
        
        precision = len(intersection) / len(predicted_set) if predicted_set else 0.0
        recall = len(intersection) / len(relevant_set) if relevant_set else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_predicted': len(predicted_apis),
            'num_relevant': len(relevant_apis),
            'num_correct': len(intersection)
        }
    
    def test_toolbench_model(self, test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test ToolBench model performance"""
        print("🔍 Testing ToolBench model...")
        
        results = []
        for i, sample in enumerate(tqdm(test_samples, desc="ToolBench Model")):
            # Safely extract data with fallbacks
            query = sample.get('query', '')
            relevant_apis = sample.get('relevant_apis', [])
            
            if not query:
                print(f"⚠️  Skipping sample {i}: no query found")
                continue
            
            # Get query embedding
            query_embedding = self.get_toolbench_embedding(query)
            
            # Search for similar APIs
            similar_apis = self.search_similar_apis(query_embedding, similarity_threshold=0.5)
            
            # Evaluate
            evaluation = self.evaluate_predictions(similar_apis, relevant_apis)
            evaluation['query'] = query
            evaluation['predicted_apis'] = similar_apis
            evaluation['relevant_apis'] = relevant_apis
            
            results.append(evaluation)
        
        return {
            'model_name': 'ToolBench',
            'results': results,
            'aggregate_metrics': self.calculate_aggregate_metrics(results)
        }
    
    def test_trained_model(self, test_samples: List[Dict[str, Any]], use_beam_search: bool = True) -> Dict[str, Any]:
        """Test trained model performance"""
        print("🔍 Testing trained magnitude regression model...")
        
        results = []
        for i, sample in enumerate(tqdm(test_samples, desc="Trained Model")):
            # Safely extract data with fallbacks
            query = sample.get('query', '')
            relevant_apis = sample.get('relevant_apis', [])
            
            if not query:
                print(f"⚠️  Skipping sample {i}: no query found")
                continue
            
            print(f"\n📝 Query {i+1}: {query}")
            
            # Use beam search or greedy search
            if use_beam_search:
                predicted_apis = self.iterative_residual_search(query, max_iterations=5, residual_threshold=0.3, beam_size=3)
            else:
                predicted_apis = self.iterative_residual_search_greedy(query, max_iterations=5, residual_threshold=0.3)
            
            # Evaluate
            evaluation = self.evaluate_predictions(predicted_apis, relevant_apis)
            evaluation['query'] = query
            evaluation['predicted_apis'] = predicted_apis
            evaluation['relevant_apis'] = relevant_apis
            
            results.append(evaluation)
            
            print(f"  Results: Precision={evaluation['precision']:.3f}, Recall={evaluation['recall']:.3f}, F1={evaluation['f1_score']:.3f}")
        
        return {
            'model_name': 'Trained Magnitude Regression',
            'results': results,
            'aggregate_metrics': self.calculate_aggregate_metrics(results)
        }
    
    def calculate_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate metrics across all results"""
        if not results:
            return {}
        
        total_precision = sum(r['precision'] for r in results)
        total_recall = sum(r['recall'] for r in results)
        total_f1 = sum(r['f1_score'] for r in results)
        
        return {
            'avg_precision': total_precision / len(results),
            'avg_recall': total_recall / len(results),
            'avg_f1_score': total_f1 / len(results),
            'total_samples': len(results)
        }
    
    def print_comparison_results(self, toolbench_results: Dict[str, Any], trained_results: Dict[str, Any]):
        """Print comparison results"""
        print("\n" + "="*80)
        print("📊 MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        # ToolBench results
        print(f"\n🔧 {toolbench_results['model_name']}:")
        metrics = toolbench_results['aggregate_metrics']
        print(f"  Average Precision: {metrics['avg_precision']:.4f}")
        print(f"  Average Recall: {metrics['avg_recall']:.4f}")
        print(f"  Average F1 Score: {metrics['avg_f1_score']:.4f}")
        print(f"  Total Samples: {metrics['total_samples']}")
        
        # Trained model results
        print(f"\n🎯 {trained_results['model_name']}:")
        metrics = trained_results['aggregate_metrics']
        print(f"  Average Precision: {metrics['avg_precision']:.4f}")
        print(f"  Average Recall: {metrics['avg_recall']:.4f}")
        print(f"  Average F1 Score: {metrics['avg_f1_score']:.4f}")
        print(f"  Total Samples: {metrics['total_samples']}")
        
        # Improvement
        toolbench_f1 = toolbench_results['aggregate_metrics']['avg_f1_score']
        trained_f1 = trained_results['aggregate_metrics']['avg_f1_score']
        improvement = ((trained_f1 - toolbench_f1) / toolbench_f1) * 100 if toolbench_f1 > 0 else 0
        
        print(f"\n📈 Improvement in F1 Score: {improvement:+.2f}%")
        
        if improvement > 0:
            print("✅ Trained model performs better!")
        elif improvement < 0:
            print("❌ ToolBench model performs better")
        else:
            print("🤝 Both models perform similarly")
    
    def load_test_instruction_data(self, instruction_dir: str = TEST_INSTRUCTION_DIR) -> List[Dict[str, Any]]:
        """
        Load test instruction data from *_instruction.json files
        """
        print(f"📦 Loading test instruction data from: {instruction_dir}")
        
        if not os.path.exists(instruction_dir):
            print(f"❌ Test instruction directory not found: {instruction_dir}")
            return []
        
        all_samples = []
        
        # Find all *_instruction.json files
        instruction_files = []
        for file in os.listdir(instruction_dir):
            if file.endswith('_instruction.json'):
                instruction_files.append(file)
        
        print(f"📁 Found {len(instruction_files)} instruction files")
        
        for file in tqdm(instruction_files, desc="Loading instruction files"):
            file_path = os.path.join(instruction_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract samples with queries and relevant APIs
                for sample in data:
                    query = sample.get("query", "")
                    api_list = sample.get("api_list", [])
                    relevant_api_keys = set()
                    
                    # Extract relevant APIs
                    for rel in sample.get("relevant APIs", []):
                        if len(rel) == 2:
                            relevant_api_keys.add((rel[0], rel[1]))  # (tool_name, api_name)
                    
                    relevant_apis = [api for api in api_list if (api.get("tool_name"), api.get("api_name")) in relevant_api_keys]
                    
                    if query and relevant_apis:  # Only include samples with both query and relevant APIs
                        all_samples.append({
                            "query": query,
                            "relevant_apis": relevant_apis,
                            "api_list": api_list,
                            "query_id": sample.get("query_id", None)
                        })
                            
            except Exception as e:
                print(f"⚠️  Error loading {file}: {e}")
                continue
        
        print(f"✅ Loaded {len(all_samples)} samples from test instruction data")
        return all_samples
    
    def run_comparison(self, max_samples: int = None):
        """Run the complete comparison"""
        print("🚀 Starting model performance comparison...")
        
        # Load models
        if not self.load_models():
            print("❌ Failed to load models")
            return
        
        # Load vector database
        if not self.load_vector_database():
            print("❌ Failed to load vector database")
            return
        
        # Load test data
        test_samples = self.load_test_instruction_data()
        
        if not test_samples:
            print("❌ No test samples found")
            return
        
        # Limit samples if specified
        if max_samples:
            test_samples = test_samples[:max_samples]
            print(f"📊 Using {len(test_samples)} test samples")
        
        # Test ToolBench model
        toolbench_results = self.test_toolbench_model(test_samples)
        
        # Test trained model
        trained_results = self.test_trained_model(test_samples, use_beam_search=True)
        
        # Print comparison
        self.print_comparison_results(toolbench_results, trained_results)
        
        # Save results
        results = {
            'toolbench_results': toolbench_results,
            'trained_results': trained_results,
            'comparison_timestamp': time.time()
        }
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'dtype'):  # numpy array or scalar
                return obj.item() if obj.size == 1 else obj.tolist()
            else:
                return obj
        
        results = convert_numpy_types(results)
        
        output_path = os.path.join(BASE_DIR, "magnitude_model_comparison_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_path}")


def main():
    """Main comparison function"""
    comparator = ModelPerformanceComparator()
    
    # Run comparison with limited samples for testing
    comparator.run_comparison(max_samples=50)


if __name__ == "__main__":
    main() 