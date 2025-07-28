#!/usr/bin/env python3
"""
Compare performance between ToolBench model and trained query embedding model
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
from train_query_embedding import QueryEmbeddingModel
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import shutil

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = "/home/jhlee/librarian/trained_query_model_unnormalized/best_model.pt"
TOOLBENCH_MODEL_NAME = "ToolBench/ToolBench_IR_bert_based_uncased"
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db_toolbench")
TEST_INSTRUCTION_DIR = "/home/jhlee/ToolBench/data/test_instruction"
SAMPLE_SIZE = 20   # Number of queries to sample (reduced for faster testing)
NUM_SAMPLES = 1    # Number of different samples to test (reduced for faster testing)


class ModelPerformanceComparator:
    """
    Compare performance between ToolBench model and trained query embedding model
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.toolbench_model = None
        self.trained_model = None
        self.vector_db = None
        self.api_metadata = []
        
    def load_models(self, load_trained_model: bool = True):
        """Load ToolBench model and optionally trained model"""
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
        
        # Load trained query embedding model (optional)
        if load_trained_model:
            try:
                self.trained_model = QueryEmbeddingModel()
                if os.path.exists(BEST_MODEL_PATH):
                    checkpoint = torch.load(BEST_MODEL_PATH, map_location=self.device)
                    self.trained_model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✅ Trained query model loaded from {BEST_MODEL_PATH}")
                else:
                    print(f"⚠️  No trained model found at {BEST_MODEL_PATH}")
                    return False
                
                self.trained_model = self.trained_model.to(self.device)
                self.trained_model.eval()
            except Exception as e:
                print(f"❌ Error loading trained model: {e}")
                return False
        else:
            print("⏭️  Skipping trained model loading")
        
        return True
    
    def build_sample_vector_database(self, sample_queries: List[Dict[str, Any]], sample_id: int) -> bool:
        """Build vector database for a specific sample"""
        print(f"🔨 Building vector database for sample {sample_id}...")
        
        # Create sample-specific directory
        sample_db_path = os.path.join(BASE_DIR, f"vector_db_sample_{sample_id}")
        os.makedirs(sample_db_path, exist_ok=True)
        
        # Collect all unique APIs from the sample
        all_apis = set()
        for query_data in sample_queries:
            for api in query_data.get('api_list', []):
                api_id = (api.get('tool_name', ''), api.get('api_name', ''))
                all_apis.add(api_id)
        
        # Convert to list and create metadata
        api_list = []
        for tool_name, api_name in all_apis:
            # Find the API in the original data
            for query_data in sample_queries:
                for api in query_data.get('api_list', []):
                    if api.get('tool_name') == tool_name and api.get('api_name') == api_name:
                        api_list.append(api)
                        break
                else:
                    continue
                break
        
        print(f"📊 Found {len(api_list)} unique APIs for sample {sample_id}")
        
        # Generate embeddings for all APIs
        api_embeddings = []
        api_metadata = []
        
        for api in tqdm(api_list, desc=f"Generating embeddings for sample {sample_id}"):
            api_text = f"{api['tool_name']} {api['api_name']} {api['api_description']}"
            
            # Get embedding using ToolBench model
            with torch.no_grad():
                embedding = self.toolbench_model.encode([api_text], convert_to_tensor=True, device=self.device)
                # Normalize for cosine similarity
                normalized_embedding = F.normalize(embedding, p=2, dim=-1)
                api_embeddings.append(normalized_embedding.cpu().numpy())
                api_metadata.append(api)
        
        # Convert to numpy array
        api_embeddings = np.vstack(api_embeddings)
        
        # Build FAISS index
        dimension = api_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(api_embeddings.astype('float32'))
        
        # Save index and metadata
        faiss_index_path = os.path.join(sample_db_path, "api_embeddings_toolbench.index")
        metadata_path = os.path.join(sample_db_path, "api_metadata_toolbench.pkl")
        
        faiss.write_index(index, faiss_index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(api_metadata, f)
        
        print(f"✅ Built vector database for sample {sample_id}: {len(api_embeddings)} APIs")
        return True
    
    def load_sample_vector_database(self, sample_id: int) -> bool:
        """Load vector database for a specific sample"""
        print(f"📦 Loading vector database for sample {sample_id}...")
        
        sample_db_path = os.path.join(BASE_DIR, f"vector_db_sample_{sample_id}")
        
        try:
            # Load FAISS index
            faiss_index_path = os.path.join(sample_db_path, "api_embeddings_toolbench.index")
            self.vector_db = faiss.read_index(faiss_index_path)
            print(f"✅ Loaded FAISS index with {self.vector_db.ntotal} vectors")
            
            # Load metadata
            metadata_path = os.path.join(sample_db_path, "api_metadata_toolbench.pkl")
            with open(metadata_path, 'rb') as f:
                self.api_metadata = pickle.load(f)
            print(f"✅ Loaded metadata for {len(self.api_metadata)} APIs")
            
            return True
        except Exception as e:
            print(f"❌ Error loading vector database for sample {sample_id}: {e}")
            return False
    
    def load_vector_database(self):
        """Load the full ToolBench vector database"""
        print("📦 Loading full ToolBench vector database...")
        
        try:
            # Load FAISS index
            faiss_index_path = os.path.join(VECTOR_DB_PATH, "api_embeddings_toolbench.index")
            self.vector_db = faiss.read_index(faiss_index_path)
            print(f"✅ Loaded FAISS index with {self.vector_db.ntotal} vectors")
            
            # Load metadata
            metadata_path = os.path.join(VECTOR_DB_PATH, "api_metadata_toolbench.pkl")
            with open(metadata_path, 'rb') as f:
                self.api_metadata = pickle.load(f)
            print(f"✅ Loaded metadata for {len(self.api_metadata)} APIs")
            
            return True
        except Exception as e:
            print(f"❌ Error loading vector database: {e}")
            return False
    
    def get_toolbench_embedding(self, query: str) -> np.ndarray:
        """Get query embedding using ToolBench model"""
        with torch.no_grad():
            embedding = self.toolbench_model.encode([query], convert_to_tensor=True, device=self.device)
            # Normalize for cosine similarity
            normalized_embedding = F.normalize(embedding, p=2, dim=-1)
            return normalized_embedding.cpu().numpy()
    
    def get_trained_embedding(self, query: str) -> np.ndarray:
        """Get query embedding using trained model"""
        tokenizer = self.trained_model.base_model.tokenizer
        max_length = self.trained_model.base_model.get_max_seq_length()
        
        features = tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        features = {k: v.to(self.device) for k, v in features.items()}
        
        with torch.no_grad():
            embedding = self.trained_model(features["input_ids"], features["attention_mask"])
            return embedding.cpu().numpy()
    
    def search_similar_apis(self, query_embedding: np.ndarray, similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search for similar APIs in vector database"""
        # Search in vector database
        # Normalize query embedding for cosine similarity
        query_norm = np.linalg.norm(query_embedding, axis=-1, keepdims=True)
        query_embedding_normalized = query_embedding / (query_norm + 1e-8)
        scores, indices = self.vector_db.search(query_embedding_normalized, k=100)  # Get top 100
        
        # Filter by similarity threshold
        similar_apis = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= similarity_threshold:
                api_info = self.api_metadata[idx].copy()
                api_info['similarity_score'] = float(score)
                similar_apis.append(api_info)
        
        return similar_apis
    
    def iterative_residual_search(self, query: str, max_iterations: int = 10, residual_threshold: float = 0.5, beam_size: int = 3) -> List[Dict[str, Any]]:
        """
        Iterative residual-based search using trained model with beam search
        """
        # Initialize beam with original query embedding
        initial_embedding = self.get_trained_embedding(query)
        
        # Beam state: (apis, current_embedding, total_similarity_score, is_active)
        beam = [([], initial_embedding, 0.0, True)]
        
        for iteration in range(max_iterations):
            new_beam = []
            
            for apis, current_embedding, total_score, is_active in beam:
                # Skip inactive beams
                if not is_active:
                    new_beam.append((apis, current_embedding, total_score, is_active))
                    continue
                
                # Normalize current embedding for search
                current_norm = np.linalg.norm(current_embedding, axis=-1, keepdims=True)
                normalized_embedding = current_embedding / (current_norm + 1e-8)
                # Search for similar APIs
                similar_apis = self.search_similar_apis(normalized_embedding, similarity_threshold=0.1)
                
                if not similar_apis:
                    # No more APIs found, mark beam as inactive
                    new_beam.append((apis, current_embedding, total_score, False))
                    continue
                
                # Get top beam_size candidates
                top_candidates = similar_apis[:beam_size]
                beam_has_valid_candidate = False
                
                for candidate_api in top_candidates:
                    # Get the API's embedding
                    api_text = f"{candidate_api['tool_name']} {candidate_api['api_name']} {candidate_api['api_description']}"
                    api_embedding = self.toolbench_model.encode([api_text], convert_to_tensor=True, device=self.device)
                    api_embedding = F.normalize(api_embedding, p=2, dim=-1).cpu().numpy()
                    
                    # Calculate residual
                    residual = current_embedding - api_embedding
                    residual_norm = np.linalg.norm(residual)
                    current_norm = np.linalg.norm(current_embedding)
                    
                    # Check if residual magnitude is increasing (bad sign)
                    if residual_norm > current_norm:
                        continue
                    
                    # Valid candidate found
                    beam_has_valid_candidate = True
                    
                    # Calculate new total score (higher similarity = better)
                    new_total_score = total_score + candidate_api['similarity_score']
                    
                    # Add to new beam
                    new_apis = apis + [candidate_api]
                    new_beam.append((new_apis, residual, new_total_score, True))
                
                # If no valid candidates found, mark beam as inactive
                if not beam_has_valid_candidate:
                    new_beam.append((apis, current_embedding, total_score, False))
            
            # Keep top beam_size candidates based on total score
            new_beam.sort(key=lambda x: x[2], reverse=True)
            beam = new_beam[:beam_size]
            
            # Check if all beams are inactive
            all_inactive = all(not state[3] for state in beam)
            if all_inactive:
                break
            
            # Check if all residuals are small enough
            active_beams = [state for state in beam if state[3]]
            if active_beams:
                all_small_residuals = all(np.linalg.norm(state[1]) < residual_threshold for state in active_beams)
                if all_small_residuals:
                    break
        
        # Return the beam with smallest residual magnitude
        if beam:
            # Sort by residual magnitude (smallest first)
            beam.sort(key=lambda x: np.linalg.norm(x[1]))
            best_apis, best_residual, _, _ = beam[0]
            return best_apis
        else:
            return []
    
    def iterative_residual_search_greedy(self, query: str, max_iterations: int = 10, residual_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Original greedy iterative residual-based search using trained model
        """
        predicted_apis = []
        current_embedding = self.get_trained_embedding(query)
        
        for iteration in range(max_iterations):
            # Search for most similar API
            similar_apis = self.search_similar_apis(current_embedding, similarity_threshold=0.1)
            
            if not similar_apis:
                break
            
            # Get the most similar API
            best_api = similar_apis[0]
            predicted_apis.append(best_api)
            
            # Get the API's embedding
            api_text = f"{best_api['tool_name']} {best_api['api_name']} {best_api['api_description']}"
            api_embedding = self.toolbench_model.encode([api_text], convert_to_tensor=True, device=self.device)
            api_embedding = F.normalize(api_embedding, p=2, dim=-1).cpu().numpy()
            
            # Calculate residual
            residual = current_embedding - api_embedding
            residual_norm = np.linalg.norm(residual)
            current_norm = np.linalg.norm(current_embedding)
            
            # Check if residual magnitude is increasing (bad sign)
            if residual_norm > current_norm:
                break
            
            # Check if residual is small enough
            if residual_norm < residual_threshold:
                break
            
            # Update current embedding to residual
            current_embedding = residual
        
        return predicted_apis
    
    def iterative_residual_search_with_relevant_apis_sum(self, query: str, relevant_apis: List[Dict[str, Any]], max_iterations: int = 10, residual_threshold: float = 0.5, beam_size: int = 3) -> List[Dict[str, Any]]:
        """
        Iterative residual-based search using relevant APIs embedding sum with beam search
        """
        # Initialize beam with relevant APIs embedding sum
        initial_embedding = self.get_relevant_apis_embedding_sum(relevant_apis)
        
        # Beam state: (apis, current_embedding, total_similarity_score, is_active)
        beam = [([], initial_embedding, 0.0, True)]
        
        for iteration in range(max_iterations):
            new_beam = []
            
            for apis, current_embedding, total_score, is_active in beam:
                # Skip inactive beams
                if not is_active:
                    new_beam.append((apis, current_embedding, total_score, is_active))
                    continue
                
                # Normalize current embedding for search
                current_norm = np.linalg.norm(current_embedding, axis=-1, keepdims=True)
                normalized_embedding = current_embedding / (current_norm + 1e-8)
                # Search for similar APIs
                similar_apis = self.search_similar_apis(normalized_embedding, similarity_threshold=0.1)
                
                if not similar_apis:
                    # No more APIs found, mark beam as inactive
                    new_beam.append((apis, current_embedding, total_score, False))
                    continue
                
                # Get top beam_size candidates
                top_candidates = similar_apis[:beam_size]
                beam_has_valid_candidate = False
                
                for candidate_api in top_candidates:
                    # Get the API's embedding
                    api_text = f"{candidate_api['tool_name']} {candidate_api['api_name']} {candidate_api['api_description']}"
                    api_embedding = self.toolbench_model.encode([api_text], convert_to_tensor=True, device=self.device)
                    api_embedding = F.normalize(api_embedding, p=2, dim=-1).cpu().numpy()
                    
                    # Calculate residual
                    residual = current_embedding - api_embedding
                    residual_norm = np.linalg.norm(residual)
                    current_norm = np.linalg.norm(current_embedding)
                    
                    # Check if residual magnitude is increasing (bad sign)
                    if residual_norm > current_norm:
                        continue
                    
                    # Valid candidate found
                    beam_has_valid_candidate = True
                    
                    # Calculate new total score (higher similarity = better)
                    new_total_score = total_score + candidate_api['similarity_score']
                    
                    # Add to new beam
                    new_apis = apis + [candidate_api]
                    new_beam.append((new_apis, residual, new_total_score, True))
                
                # If no valid candidates found, mark beam as inactive
                if not beam_has_valid_candidate:
                    new_beam.append((apis, current_embedding, total_score, False))
            
            # Keep top beam_size candidates based on total score
            new_beam.sort(key=lambda x: x[2], reverse=True)
            beam = new_beam[:beam_size]
            
            # Check if all beams are inactive
            all_inactive = all(not state[3] for state in beam)
            if all_inactive:
                break
            
            # Check if all residuals are small enough
            active_beams = [state for state in beam if state[3]]
            if active_beams:
                all_small_residuals = all(np.linalg.norm(state[1]) < residual_threshold for state in active_beams)
                if all_small_residuals:
                    break
        
        # Return the beam with smallest residual magnitude
        if beam:
            # Sort by residual magnitude (smallest first)
            beam.sort(key=lambda x: np.linalg.norm(x[1]))
            best_apis, best_residual, _, _ = beam[0]
            return best_apis
        else:
            return []
    
    def iterative_residual_search_greedy_with_relevant_apis_sum(self, query: str, relevant_apis: List[Dict[str, Any]], max_iterations: int = 10, residual_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Greedy iterative residual-based search using relevant APIs embedding sum
        """
        predicted_apis = []
        current_embedding = self.get_relevant_apis_embedding_sum(relevant_apis)
        
        for iteration in range(max_iterations):
            # Search for most similar API
            similar_apis = self.search_similar_apis(current_embedding, similarity_threshold=0.1)
            
            if not similar_apis:
                break
            
            # Get the most similar API
            best_api = similar_apis[0]
            predicted_apis.append(best_api)
            
            # Get the API's embedding
            api_text = f"{best_api['tool_name']} {best_api['api_name']} {best_api['api_description']}"
            api_embedding = self.toolbench_model.encode([api_text], convert_to_tensor=True, device=self.device)
            api_embedding = F.normalize(api_embedding, p=2, dim=-1).cpu().numpy()
            
            # Calculate residual
            residual = current_embedding - api_embedding
            residual_norm = np.linalg.norm(residual)
            current_norm = np.linalg.norm(current_embedding)
            
            # Check if residual magnitude is increasing (bad sign)
            if residual_norm > current_norm:
                break
            
            # Check if residual is small enough
            if residual_norm < residual_threshold:
                break
            
            # Update current embedding to residual
            current_embedding = residual
        
        return predicted_apis
    
    def evaluate_predictions(self, predicted_apis: List[Dict[str, Any]], relevant_apis: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate predictions against ground truth"""
        # Create sets of predicted and relevant APIs
        predicted_set = set()
        for api in predicted_apis:
            api_id = (api['tool_name'], api['api_name'])
            predicted_set.add(api_id)
        
        relevant_set = set()
        for api in relevant_apis:
            api_id = (api.get('tool_name', ''), api.get('api_name', ''))
            relevant_set.add(api_id)
        
        # Calculate metrics
        if len(predicted_set) == 0 and len(relevant_set) == 0:
            precision = recall = f1 = 1.0
        elif len(predicted_set) == 0:
            precision = recall = f1 = 0.0
        elif len(relevant_set) == 0:
            precision = recall = f1 = 0.0
        else:
            intersection = predicted_set.intersection(relevant_set)
            precision = len(intersection) / len(predicted_set) if len(predicted_set) > 0 else 0
            recall = len(intersection) / len(relevant_set) if len(relevant_set) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predicted_count': len(predicted_set),
            'relevant_count': len(relevant_set),
            'intersection_count': len(predicted_set.intersection(relevant_set))
        }
    
    def test_toolbench_model(self, test_samples: List[Dict[str, Any]], sample_id: int) -> Dict[str, Any]:
        """Test ToolBench model performance"""
        print(f"\n🧪 Testing ToolBench model for sample {sample_id}...")
        
        results = []
        for i, sample in enumerate(tqdm(test_samples, desc=f"Testing ToolBench model (sample {sample_id})")):
            query = sample['query']
            relevant_apis = sample.get('relevant_apis', [])
            
            # Get query embedding using ToolBench model
            query_embedding = self.get_toolbench_embedding(query)
            
            # Search for similar APIs
            similar_apis = self.search_similar_apis(query_embedding, similarity_threshold=0.5)
            
            # Evaluate predictions
            metrics = self.evaluate_predictions(similar_apis, relevant_apis)
            
            results.append({
                'query': query,
                'predicted_apis': similar_apis,
                'relevant_apis': relevant_apis,
                'metrics': metrics
            })
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(results)
        
        return {
            'results': results,
            'aggregate_metrics': aggregate_metrics
        }
    
    def test_trained_model(self, test_samples: List[Dict[str, Any]], sample_id: int, use_beam_search: bool = True) -> Dict[str, Any]:
        """Test performance using relevant APIs embedding sum instead of trained model"""
        search_method = "beam search" if use_beam_search else "greedy search"
        print(f"\n🧪 Testing relevant APIs embedding sum (iterative residual {search_method}) for sample {sample_id}...")
        
        results = []
        for i, sample in enumerate(tqdm(test_samples, desc=f"Testing relevant APIs sum ({search_method}, sample {sample_id})")):
            query = sample['query']
            relevant_apis = sample.get('relevant_apis', [])
            
            if not relevant_apis:
                # Skip samples without relevant APIs
                continue
            
            # Use iterative residual search with relevant APIs embedding sum
            if use_beam_search:
                predicted_apis = self.iterative_residual_search_with_relevant_apis_sum(query, relevant_apis, beam_size=3)
            else:
                predicted_apis = self.iterative_residual_search_greedy_with_relevant_apis_sum(query, relevant_apis)
            
            # Evaluate predictions
            metrics = self.evaluate_predictions(predicted_apis, relevant_apis)
            
            # Get relevant APIs embedding sum for analysis
            relevant_apis_embedding_sum = self.get_relevant_apis_embedding_sum(relevant_apis)
            
            results.append({
                'query': query,
                'predicted_apis': predicted_apis,
                'relevant_apis': relevant_apis,
                'metrics': metrics,
                'relevant_apis_embedding_sum': relevant_apis_embedding_sum.tolist()
            })
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(results)
        
        return {
            'results': results,
            'aggregate_metrics': aggregate_metrics
        }
    
    def calculate_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate metrics from individual results"""
        metrics_list = [r['metrics'] for r in results]
        
        return {
            'precision': np.mean([m['precision'] for m in metrics_list]),
            'recall': np.mean([m['recall'] for m in metrics_list]),
            'f1_score': np.mean([m['f1_score'] for m in metrics_list]),
            'avg_predicted_count': np.mean([m['predicted_count'] for m in metrics_list]),
            'avg_relevant_count': np.mean([m['relevant_count'] for m in metrics_list]),
            'avg_intersection_count': np.mean([m['intersection_count'] for m in metrics_list])
        }
    
    def print_comparison_results(self, toolbench_results: Dict[str, Any], trained_results: Dict[str, Any], sample_id: int):
        """Print comparison results"""
        print(f"\n" + "="*80)
        print(f"📊 MODEL PERFORMANCE COMPARISON - SAMPLE {sample_id}")
        print("="*80)
        
        toolbench_metrics = toolbench_results['aggregate_metrics']
        trained_metrics = trained_results['aggregate_metrics']
        
        print(f"{'Metric':<20} {'ToolBench':<15} {'Relevant APIs Sum':<15} {'Difference':<15}")
        print("-" * 80)
        print(f"{'Precision':<20} {toolbench_metrics['precision']:<15.4f} {trained_metrics['precision']:<15.4f} {trained_metrics['precision'] - toolbench_metrics['precision']:<15.4f}")
        print(f"{'Recall':<20} {toolbench_metrics['recall']:<15.4f} {trained_metrics['recall']:<15.4f} {trained_metrics['recall'] - toolbench_metrics['recall']:<15.4f}")
        print(f"{'F1 Score':<20} {toolbench_metrics['f1_score']:<15.4f} {trained_metrics['f1_score']:<15.4f} {trained_metrics['f1_score'] - toolbench_metrics['f1_score']:<15.4f}")
        print(f"{'Avg Predicted':<20} {toolbench_metrics['avg_predicted_count']:<15.2f} {trained_metrics['avg_predicted_count']:<15.2f} {trained_metrics['avg_predicted_count'] - toolbench_metrics['avg_predicted_count']:<15.2f}")
        print(f"{'Avg Relevant':<20} {toolbench_metrics['avg_relevant_count']:<15.2f} {trained_metrics['avg_relevant_count']:<15.2f} {'N/A':<15}")
        print(f"{'Avg Intersection':<20} {toolbench_metrics['avg_intersection_count']:<15.2f} {trained_metrics['avg_intersection_count']:<15.2f} {trained_metrics['avg_intersection_count'] - toolbench_metrics['avg_intersection_count']:<15.2f}")
        
        print(f"\n" + "="*80)
        print(f"🏆 WINNER ANALYSIS - SAMPLE {sample_id}")
        print("="*80)
        
        if trained_metrics['f1_score'] > toolbench_metrics['f1_score']:
            print("🎉 Relevant APIs embedding sum wins in F1 Score!")
        elif trained_metrics['f1_score'] < toolbench_metrics['f1_score']:
            print("🎉 ToolBench model wins in F1 Score!")
        else:
            print("🤝 Models are tied in F1 Score!")
        
        if trained_metrics['precision'] > toolbench_metrics['precision']:
            print("🎯 Relevant APIs embedding sum has higher precision!")
        elif trained_metrics['precision'] < toolbench_metrics['precision']:
            print("🎯 ToolBench model has higher precision!")
        
        if trained_metrics['recall'] > toolbench_metrics['recall']:
            print("📈 Relevant APIs embedding sum has higher recall!")
        elif trained_metrics['recall'] < toolbench_metrics['recall']:
            print("📈 ToolBench model has higher recall!")
    
    def print_relevant_apis_sum_comparison(self, toolbench_results: Dict[str, Any], relevant_apis_sum_results: Dict[str, Any], ideal_results: Dict[str, Any], sample_id: int):
        """Print comparison results for relevant APIs sum experiment"""
        print(f"\n" + "="*80)
        print(f"📊 RELEVANT APIS SUM EXPERIMENT - SAMPLE {sample_id}")
        print("="*80)
        
        toolbench_metrics = toolbench_results['aggregate_metrics']
        relevant_apis_sum_metrics = relevant_apis_sum_results['aggregate_metrics']
        ideal_metrics = ideal_results['aggregate_metrics']
        
        print(f"{'Metric':<25} {'ToolBench':<15} {'Relevant Sum':<15} {'Ideal':<15}")
        print("-" * 80)
        print(f"{'Precision':<25} {toolbench_metrics['precision']:<15.4f} {relevant_apis_sum_metrics['precision']:<15.4f} {ideal_metrics['precision']:<15.4f}")
        print(f"{'Recall':<25} {toolbench_metrics['recall']:<15.4f} {relevant_apis_sum_metrics['recall']:<15.4f} {ideal_metrics['recall']:<15.4f}")
        print(f"{'F1 Score':<25} {toolbench_metrics['f1_score']:<15.4f} {relevant_apis_sum_metrics['f1_score']:<15.4f} {ideal_metrics['f1_score']:<15.4f}")
        print(f"{'Avg Predicted':<25} {toolbench_metrics['avg_predicted_count']:<15.2f} {relevant_apis_sum_metrics['avg_predicted_count']:<15.2f} {ideal_metrics['avg_predicted_count']:<15.2f}")
        print(f"{'Avg Relevant':<25} {toolbench_metrics['avg_relevant_count']:<15.2f} {relevant_apis_sum_metrics['avg_relevant_count']:<15.2f} {ideal_metrics['avg_relevant_count']:<15.2f}")
        print(f"{'Avg Intersection':<25} {toolbench_metrics['avg_intersection_count']:<15.2f} {relevant_apis_sum_metrics['avg_intersection_count']:<15.2f} {ideal_metrics['avg_intersection_count']:<15.2f}")
        
        print(f"\n" + "="*80)
        print(f"🏆 RELEVANT APIS SUM ANALYSIS - SAMPLE {sample_id}")
        print("="*80)
        
        # Compare ToolBench vs Relevant APIs Sum
        if relevant_apis_sum_metrics['f1_score'] > toolbench_metrics['f1_score']:
            print("🎉 Relevant APIs sum performs better than ToolBench query embedding!")
        elif relevant_apis_sum_metrics['f1_score'] < toolbench_metrics['f1_score']:
            print("🎉 ToolBench query embedding performs better than relevant APIs sum!")
        else:
            print("🤝 Both approaches perform similarly!")
        
        # Compare with ideal performance
        ideal_f1 = ideal_metrics['f1_score']
        toolbench_f1 = toolbench_metrics['f1_score']
        relevant_sum_f1 = relevant_apis_sum_metrics['f1_score']
        
        print(f"📊 Performance gap to ideal:")
        print(f"   ToolBench: {ideal_f1 - toolbench_f1:.4f} ({(toolbench_f1/ideal_f1)*100:.1f}% of ideal)")
        print(f"   Relevant Sum: {ideal_f1 - relevant_sum_f1:.4f} ({(relevant_sum_f1/ideal_f1)*100:.1f}% of ideal)")
        
        if relevant_sum_f1 > toolbench_f1:
            print("🎯 Relevant APIs sum reduces the gap to ideal performance!")
        elif relevant_sum_f1 < toolbench_f1:
            print("🎯 ToolBench query embedding is closer to ideal performance!")
        else:
            print("🤝 Both approaches have similar gap to ideal performance!")
    
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
    
    def create_sample_queries(self, all_samples: List[Dict[str, Any]], sample_id: int) -> List[Dict[str, Any]]:
        """Create a sample of queries for testing"""
        # Set random seed for reproducible sampling
        random.seed(42 + sample_id)
        
        # Sample SAMPLE_SIZE queries
        if len(all_samples) <= SAMPLE_SIZE:
            sample_queries = all_samples
        else:
            sample_queries = random.sample(all_samples, SAMPLE_SIZE)
        
        print(f"📊 Created sample {sample_id} with {len(sample_queries)} queries")
        return sample_queries
    
    def run_full_comparison(self):
        """Run complete performance comparison using full ToolBench vector database"""
        print("🚀 Starting Full Model Performance Comparison")
        print("=" * 50)
        
        # Load all test data from ToolBench instruction files
        all_samples = self.load_test_instruction_data()
        
        if not all_samples:
            print("❌ No test samples found. Exiting.")
            return
        
        print(f"📊 Total available samples: {len(all_samples)}")
        
        # Load full vector database
        if not self.load_vector_database():
            print("❌ Failed to load vector database. Exiting.")
            return
        
        # Create multiple samples for testing
        all_results = []
        
        for sample_id in range(1, NUM_SAMPLES + 1):
            print(f"\n" + "="*60)
            print(f"🧪 PROCESSING SAMPLE {sample_id}/{NUM_SAMPLES}")
            print("="*60)
            
            # Create sample queries
            sample_queries = self.create_sample_queries(all_samples, sample_id)
            
            # Test ToolBench model
            toolbench_results = self.test_toolbench_model(sample_queries, sample_id)
            
            # Test trained model with beam search
            trained_beam_results = self.test_trained_model(sample_queries, sample_id, use_beam_search=True)
            
            # Test trained model with greedy search
            trained_greedy_results = self.test_trained_model(sample_queries, sample_id, use_beam_search=False)
            
            # Print comparison
            self.print_comparison_results(toolbench_results, trained_beam_results, sample_id)
            
            # Store results
            all_results.append({
                'sample_id': sample_id,
                'toolbench_results': toolbench_results,
                'trained_beam_results': trained_beam_results,
                'trained_greedy_results': trained_greedy_results
            })
        
        # Print overall summary
        self.print_overall_summary(all_results)
        
        # Save detailed results
        output_path = "/home/jhlee/librarian/full_model_comparison_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Detailed results saved to: {output_path}")
    
    def print_overall_summary(self, all_results: List[Dict[str, Any]]):
        """Print overall summary across all samples"""
        print("\n" + "="*80)
        print("📊 OVERALL SUMMARY ACROSS ALL SAMPLES")
        print("="*80)
        
        toolbench_f1_scores = []
        relevant_apis_beam_f1_scores = []
        relevant_apis_greedy_f1_scores = []
        
        for result in all_results:
            toolbench_f1_scores.append(result['toolbench_results']['aggregate_metrics']['f1_score'])
            relevant_apis_beam_f1_scores.append(result['trained_beam_results']['aggregate_metrics']['f1_score'])
            relevant_apis_greedy_f1_scores.append(result['trained_greedy_results']['aggregate_metrics']['f1_score'])
        
        print(f"{'Model':<25} {'Avg F1':<15} {'Std F1':<15} {'Min F1':<15} {'Max F1':<15}")
        print("-" * 80)
        print(f"{'ToolBench':<25} {np.mean(toolbench_f1_scores):<15.4f} {np.std(toolbench_f1_scores):<15.4f} {np.min(toolbench_f1_scores):<15.4f} {np.max(toolbench_f1_scores):<15.4f}")
        print(f"{'Relevant APIs (Beam)':<25} {np.mean(relevant_apis_beam_f1_scores):<15.4f} {np.std(relevant_apis_beam_f1_scores):<15.4f} {np.min(relevant_apis_beam_f1_scores):<15.4f} {np.max(relevant_apis_beam_f1_scores):<15.4f}")
        print(f"{'Relevant APIs (Greedy)':<25} {np.mean(relevant_apis_greedy_f1_scores):<15.4f} {np.std(relevant_apis_greedy_f1_scores):<15.4f} {np.min(relevant_apis_greedy_f1_scores):<15.4f} {np.max(relevant_apis_greedy_f1_scores):<15.4f}")
        
        print(f"\n" + "="*80)
        print("🏆 OVERALL WINNER ANALYSIS")
        print("="*80)
        
        avg_toolbench_f1 = np.mean(toolbench_f1_scores)
        avg_relevant_apis_beam_f1 = np.mean(relevant_apis_beam_f1_scores)
        avg_relevant_apis_greedy_f1 = np.mean(relevant_apis_greedy_f1_scores)
        
        best_model = max([avg_toolbench_f1, avg_relevant_apis_beam_f1, avg_relevant_apis_greedy_f1])
        
        if best_model == avg_toolbench_f1:
            print("🎉 ToolBench model has the best average F1 score!")
        elif best_model == avg_relevant_apis_beam_f1:
            print("🎉 Relevant APIs embedding sum with beam search has the best average F1 score!")
        else:
            print("🎉 Relevant APIs embedding sum with greedy search has the best average F1 score!")
        
        print(f"ToolBench avg F1: {avg_toolbench_f1:.4f}")
        print(f"Relevant APIs (Beam) avg F1: {avg_relevant_apis_beam_f1:.4f}")
        print(f"Relevant APIs (Greedy) avg F1: {avg_relevant_apis_greedy_f1:.4f}")

    def test_faiss_vs_list_performance(self, test_samples: List[Dict[str, Any]], sample_id: int) -> Dict[str, Any]:
        """Compare FAISS vs List-based search performance"""
        print(f"\n🔍 Comparing FAISS vs List-based search for sample {sample_id}...")
        
        # Test FAISS performance
        print("Testing FAISS search...")
        faiss_start = time.time()
        faiss_results = self.test_toolbench_model(test_samples, sample_id)
        faiss_end = time.time()
        faiss_time = faiss_end - faiss_start
        
        # Test List-based performance
        print("Testing List-based search...")
        list_start = time.time()
        list_results = self.test_list_based_search(test_samples, sample_id)
        list_end = time.time()
        list_time = list_end - list_start
        
        # Compare results
        faiss_metrics = faiss_results['aggregate_metrics']
        list_metrics = list_results['aggregate_metrics']
        
        print(f"\n" + "="*80)
        print(f"🔍 FAISS vs LIST-BASED SEARCH COMPARISON - SAMPLE {sample_id}")
        print("="*80)
        
        print(f"{'Metric':<20} {'FAISS':<15} {'List':<15} {'Difference':<15}")
        print("-" * 80)
        print(f"{'Precision':<20} {faiss_metrics['precision']:<15.4f} {list_metrics['precision']:<15.4f} {list_metrics['precision'] - faiss_metrics['precision']:<15.4f}")
        print(f"{'Recall':<20} {faiss_metrics['recall']:<15.4f} {list_metrics['recall']:<15.4f} {list_metrics['recall'] - faiss_metrics['recall']:<15.4f}")
        print(f"{'F1 Score':<20} {faiss_metrics['f1_score']:<15.4f} {list_metrics['f1_score']:<15.4f} {list_metrics['f1_score'] - faiss_metrics['f1_score']:<15.4f}")
        print(f"{'Avg Predicted':<20} {faiss_metrics['avg_predicted_count']:<15.2f} {list_metrics['avg_predicted_count']:<15.2f} {list_metrics['avg_predicted_count'] - faiss_metrics['avg_predicted_count']:<15.2f}")
        print(f"{'Search Time (s)':<20} {faiss_time:<15.4f} {list_time:<15.4f} {list_time - faiss_time:<15.4f}")
        
        print(f"\n" + "="*80)
        print(f"🏆 PERFORMANCE ANALYSIS - SAMPLE {sample_id}")
        print("="*80)
        
        if faiss_time < list_time:
            print(f"⚡ FAISS is {list_time/faiss_time:.2f}x faster than List-based search!")
        else:
            print(f"⚡ List-based search is {faiss_time/list_time:.2f}x faster than FAISS!")
        
        if faiss_metrics['f1_score'] > list_metrics['f1_score']:
            print("🎯 FAISS has better F1 score!")
        elif faiss_metrics['f1_score'] < list_metrics['f1_score']:
            print("🎯 List-based search has better F1 score!")
        else:
            print("🤝 Both methods have the same F1 score!")
        
        return {
            'faiss_results': faiss_results,
            'list_results': list_results,
            'faiss_time': faiss_time,
            'list_time': list_time,
            'speedup': list_time / faiss_time if faiss_time > 0 else float('inf')
        }

    def compare_embedding_accuracy(self, test_samples: List[Dict[str, Any]], sample_id: int) -> Dict[str, Any]:
        """Compare embedding accuracy between trained model and ToolBench model"""
        print(f"\n🔍 Comparing embedding accuracy for sample {sample_id}...")
        
        # Build API list for this sample
        all_apis = set()
        for query_data in test_samples:
            for api in query_data.get('api_list', []):
                api_id = (api.get('tool_name', ''), api.get('api_name', ''))
                all_apis.add(api_id)
        
        # Convert to list and create metadata
        api_list = []
        for tool_name, api_name in all_apis:
            for query_data in test_samples:
                for api in query_data.get('api_list', []):
                    if api.get('tool_name') == tool_name and api.get('api_name') == api_name:
                        api_list.append(api)
                        break
                else:
                    continue
                break
        
        # Generate API embeddings using both models
        print("Generating API embeddings using ToolBench model...")
        toolbench_api_embeddings = []
        for api in tqdm(api_list, desc="ToolBench API embeddings"):
            api_text = f"{api['tool_name']} {api['api_name']} {api['api_description']}"
            with torch.no_grad():
                embedding = self.toolbench_model.encode([api_text], convert_to_tensor=True, device=self.device)
                normalized_embedding = F.normalize(embedding, p=2, dim=-1)
                toolbench_api_embeddings.append(normalized_embedding.cpu().numpy())
        
        print("Generating API embeddings using trained model...")
        trained_api_embeddings = []
        tokenizer = self.trained_model.base_model.tokenizer
        max_length = self.trained_model.base_model.get_max_seq_length()
        
        for api in tqdm(api_list, desc="Trained API embeddings"):
            api_text = f"{api['tool_name']} {api['api_name']} {api['api_description']}"
            features = tokenizer(
                [api_text],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            features = {k: v.to(self.device) for k, v in features.items()}
            
            with torch.no_grad():
                embedding = self.trained_model(features["input_ids"], features["attention_mask"])
                trained_api_embeddings.append(embedding.cpu().numpy())
        
        # Convert to numpy arrays
        toolbench_api_embeddings = np.vstack(toolbench_api_embeddings)
        trained_api_embeddings = np.vstack(trained_api_embeddings)
        
        print(f"📊 Generated embeddings for {len(api_list)} APIs")
        
        # Test queries
        results = []
        
        for i, sample in enumerate(tqdm(test_samples, desc=f"Testing embedding accuracy (sample {sample_id})")):
            query = sample['query']
            relevant_apis = sample.get('relevant_apis', [])
            
            if not relevant_apis:
                continue
            
            # Get query embeddings using both models
            toolbench_query_embedding = self.get_toolbench_embedding(query)
            trained_query_embedding = self.get_trained_embedding(query)
            
            # 1. Cosine similarity between query and each relevant API
            toolbench_cosine_scores = []
            trained_cosine_scores = []
            
            for relevant_api in relevant_apis:
                # Find the API in our list
                api_idx = None
                for j, api in enumerate(api_list):
                    if (api['tool_name'] == relevant_api['tool_name'] and 
                        api['api_name'] == relevant_api['api_name']):
                        api_idx = j
                        break
                
                if api_idx is not None:
                    # ToolBench cosine similarity
                    toolbench_similarity = np.dot(
                        toolbench_query_embedding.flatten(), 
                        toolbench_api_embeddings[api_idx].flatten()
                    )
                    toolbench_cosine_scores.append(toolbench_similarity)
                    
                    # Trained cosine similarity
                    trained_similarity = np.dot(
                        trained_query_embedding.flatten(), 
                        trained_api_embeddings[api_idx].flatten()
                    )
                    trained_cosine_scores.append(trained_similarity)
            
            # 2. L2 distance between query embedding and sum of relevant API embeddings
            toolbench_l2_scores = []
            trained_l2_scores = []
            
            if relevant_apis:
                # Sum of relevant API embeddings (unnormalized)
                toolbench_relevant_sum = np.zeros_like(toolbench_query_embedding)
                trained_relevant_sum = np.zeros_like(trained_query_embedding)
                
                for relevant_api in relevant_apis:
                    api_idx = None
                    for j, api in enumerate(api_list):
                        if (api['tool_name'] == relevant_api['tool_name'] and 
                            api['api_name'] == relevant_api['api_name']):
                            api_idx = j
                            break
                    
                    if api_idx is not None:
                        toolbench_relevant_sum += toolbench_api_embeddings[api_idx]
                        trained_relevant_sum += trained_api_embeddings[api_idx]
                
                # L2 distance
                toolbench_l2_distance = np.linalg.norm(toolbench_query_embedding - toolbench_relevant_sum)
                trained_l2_distance = np.linalg.norm(trained_query_embedding - trained_relevant_sum)
                
                toolbench_l2_scores.append(toolbench_l2_distance)
                trained_l2_scores.append(trained_l2_distance)
            
            # 2-1. Global ranking: find most similar APIs in global pool
            toolbench_global_rankings = []
            trained_global_rankings = []
            
            for relevant_api in relevant_apis:
                # Find the API in our list
                api_idx = None
                for j, api in enumerate(api_list):
                    if (api['tool_name'] == relevant_api['tool_name'] and 
                        api['api_name'] == relevant_api['api_name']):
                        api_idx = j
                        break
                
                if api_idx is not None:
                    # Calculate similarities with all APIs
                    toolbench_similarities = []
                    trained_similarities = []
                    
                    for j in range(len(api_list)):
                        toolbench_sim = np.dot(
                            toolbench_query_embedding.flatten(), 
                            toolbench_api_embeddings[j].flatten()
                        )
                        trained_sim = np.dot(
                            trained_query_embedding.flatten(), 
                            trained_api_embeddings[j].flatten()
                        )
                        
                        toolbench_similarities.append((toolbench_sim, j))
                        trained_similarities.append((trained_sim, j))
                    
                    # Sort by similarity
                    toolbench_similarities.sort(key=lambda x: x[0], reverse=True)
                    trained_similarities.sort(key=lambda x: x[0], reverse=True)
                    
                    # Find rank of relevant API
                    toolbench_rank = None
                    trained_rank = None
                    
                    for rank, (_, idx) in enumerate(toolbench_similarities):
                        if idx == api_idx:
                            toolbench_rank = rank + 1
                            break
                    
                    for rank, (_, idx) in enumerate(trained_similarities):
                        if idx == api_idx:
                            trained_rank = rank + 1
                            break
                    
                    if toolbench_rank is not None:
                        toolbench_global_rankings.append(toolbench_rank)
                    if trained_rank is not None:
                        trained_global_rankings.append(trained_rank)
            
            results.append({
                'query': query,
                'relevant_apis': relevant_apis,
                'toolbench_cosine_scores': [float(x) for x in toolbench_cosine_scores],
                'trained_cosine_scores': [float(x) for x in trained_cosine_scores],
                'toolbench_l2_scores': [float(x) for x in toolbench_l2_scores],
                'trained_l2_scores': [float(x) for x in trained_l2_scores],
                'toolbench_global_rankings': [int(x) for x in toolbench_global_rankings],
                'trained_global_rankings': [int(x) for x in trained_global_rankings]
            })
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_embedding_accuracy_metrics(results)
        
        return {
            'results': results,
            'aggregate_metrics': aggregate_metrics
        }
    
    def calculate_embedding_accuracy_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate metrics for embedding accuracy comparison"""
        toolbench_cosine_all = []
        trained_cosine_all = []
        toolbench_l2_all = []
        trained_l2_all = []
        toolbench_rankings_all = []
        trained_rankings_all = []
        
        for result in results:
            toolbench_cosine_all.extend(result['toolbench_cosine_scores'])
            trained_cosine_all.extend(result['trained_cosine_scores'])
            toolbench_l2_all.extend(result['toolbench_l2_scores'])
            trained_l2_all.extend(result['trained_l2_scores'])
            toolbench_rankings_all.extend(result['toolbench_global_rankings'])
            trained_rankings_all.extend(result['trained_global_rankings'])
        
        return {
            'toolbench_avg_cosine': float(np.mean(toolbench_cosine_all) if toolbench_cosine_all else 0.0),
            'trained_avg_cosine': float(np.mean(trained_cosine_all) if trained_cosine_all else 0.0),
            'toolbench_std_cosine': float(np.std(toolbench_cosine_all) if toolbench_cosine_all else 0.0),
            'trained_std_cosine': float(np.std(trained_cosine_all) if trained_cosine_all else 0.0),
            'toolbench_avg_l2': float(np.mean(toolbench_l2_all) if toolbench_l2_all else 0.0),
            'trained_avg_l2': float(np.mean(trained_l2_all) if trained_l2_all else 0.0),
            'toolbench_std_l2': float(np.std(toolbench_l2_all) if toolbench_l2_all else 0.0),
            'trained_std_l2': float(np.std(trained_l2_all) if trained_l2_all else 0.0),
            'toolbench_avg_ranking': float(np.mean(toolbench_rankings_all) if toolbench_rankings_all else 0.0),
            'trained_avg_ranking': float(np.mean(trained_rankings_all) if trained_rankings_all else 0.0),
            'toolbench_median_ranking': float(np.median(toolbench_rankings_all) if toolbench_rankings_all else 0.0),
            'trained_median_ranking': float(np.median(trained_rankings_all) if trained_rankings_all else 0.0),
            'toolbench_top1_count': int(sum(1 for rank in toolbench_rankings_all if rank == 1)),
            'trained_top1_count': int(sum(1 for rank in trained_rankings_all if rank == 1)),
            'toolbench_top5_count': int(sum(1 for rank in toolbench_rankings_all if rank <= 5)),
            'trained_top5_count': int(sum(1 for rank in trained_rankings_all if rank <= 5)),
            'toolbench_top10_count': int(sum(1 for rank in toolbench_rankings_all if rank <= 10)),
            'trained_top10_count': int(sum(1 for rank in trained_rankings_all if rank <= 10)),
            'total_relevant_apis': int(len(toolbench_rankings_all))
        }
    
    def print_embedding_accuracy_comparison(self, results: Dict[str, Any], sample_id: int):
        """Print embedding accuracy comparison results"""
        metrics = results['aggregate_metrics']
        
        print(f"\n" + "="*80)
        print(f"🔍 EMBEDDING ACCURACY COMPARISON - SAMPLE {sample_id}")
        print("="*80)
        
        print(f"{'Metric':<25} {'ToolBench':<15} {'Trained':<15} {'Difference':<15}")
        print("-" * 80)
        
        # Cosine similarity metrics
        print(f"{'Avg Cosine Similarity':<25} {metrics['toolbench_avg_cosine']:<15.4f} {metrics['trained_avg_cosine']:<15.4f} {metrics['trained_avg_cosine'] - metrics['toolbench_avg_cosine']:<15.4f}")
        print(f"{'Std Cosine Similarity':<25} {metrics['toolbench_std_cosine']:<15.4f} {metrics['trained_std_cosine']:<15.4f} {metrics['trained_std_cosine'] - metrics['toolbench_std_cosine']:<15.4f}")
        
        # L2 distance metrics
        print(f"{'Avg L2 Distance':<25} {metrics['toolbench_avg_l2']:<15.4f} {metrics['trained_avg_l2']:<15.4f} {metrics['trained_avg_l2'] - metrics['toolbench_avg_l2']:<15.4f}")
        print(f"{'Std L2 Distance':<25} {metrics['toolbench_std_l2']:<15.4f} {metrics['trained_std_l2']:<15.4f} {metrics['trained_std_l2'] - metrics['toolbench_std_l2']:<15.4f}")
        
        # Ranking metrics
        print(f"{'Avg Global Ranking':<25} {metrics['toolbench_avg_ranking']:<15.2f} {metrics['trained_avg_ranking']:<15.2f} {metrics['trained_avg_ranking'] - metrics['toolbench_avg_ranking']:<15.2f}")
        print(f"{'Median Global Ranking':<25} {metrics['toolbench_median_ranking']:<15.2f} {metrics['trained_median_ranking']:<15.2f} {metrics['trained_median_ranking'] - metrics['toolbench_median_ranking']:<15.2f}")
        
        # Top-k metrics
        total_apis = metrics['total_relevant_apis']
        print(f"{'Top-1 Accuracy':<25} {metrics['toolbench_top1_count']}/{total_apis:<14} {metrics['trained_top1_count']}/{total_apis:<14} {metrics['trained_top1_count'] - metrics['toolbench_top1_count']:<15}")
        print(f"{'Top-5 Accuracy':<25} {metrics['toolbench_top5_count']}/{total_apis:<14} {metrics['trained_top5_count']}/{total_apis:<14} {metrics['trained_top5_count'] - metrics['toolbench_top5_count']:<15}")
        print(f"{'Top-10 Accuracy':<25} {metrics['toolbench_top10_count']}/{total_apis:<14} {metrics['trained_top10_count']}/{total_apis:<14} {metrics['trained_top10_count'] - metrics['toolbench_top10_count']:<15}")
        
        print(f"\n" + "="*80)
        print(f"🏆 EMBEDDING ACCURACY ANALYSIS - SAMPLE {sample_id}")
        print("="*80)
        
        # Cosine similarity analysis
        if metrics['toolbench_avg_cosine'] > metrics['trained_avg_cosine']:
            print("🎯 ToolBench model has higher average cosine similarity!")
        elif metrics['toolbench_avg_cosine'] < metrics['trained_avg_cosine']:
            print("🎯 Trained model has higher average cosine similarity!")
        else:
            print("🤝 Both models have the same average cosine similarity!")
        
        # L2 distance analysis
        if metrics['toolbench_avg_l2'] < metrics['trained_avg_l2']:
            print("🎯 ToolBench model has lower average L2 distance!")
        elif metrics['toolbench_avg_l2'] > metrics['trained_avg_l2']:
            print("🎯 Trained model has lower average L2 distance!")
        else:
            print("🤝 Both models have the same average L2 distance!")
        
        # Global ranking analysis
        if metrics['toolbench_avg_ranking'] < metrics['trained_avg_ranking']:
            print("🎯 ToolBench model has better global ranking!")
        elif metrics['toolbench_avg_ranking'] > metrics['trained_avg_ranking']:
            print("🎯 Trained model has better global ranking!")
        else:
            print("🤝 Both models have the same global ranking!")
        
        # Top-k analysis
        toolbench_top1_rate = metrics['toolbench_top1_count'] / total_apis if total_apis > 0 else 0
        trained_top1_rate = metrics['trained_top1_count'] / total_apis if total_apis > 0 else 0
        
        if toolbench_top1_rate > trained_top1_rate:
            print(f"🎯 ToolBench model has higher top-1 accuracy ({toolbench_top1_rate:.2%} vs {trained_top1_rate:.2%})!")
        elif toolbench_top1_rate < trained_top1_rate:
            print(f"🎯 Trained model has higher top-1 accuracy ({trained_top1_rate:.2%} vs {toolbench_top1_rate:.2%})!")
        else:
            print("🤝 Both models have the same top-1 accuracy!")


    


    def get_relevant_apis_embedding_sum(self, relevant_apis: List[Dict[str, Any]]) -> np.ndarray:
        """Get sum of relevant APIs embeddings using ToolBench model"""
        if not relevant_apis:
            # Return zero embedding if no relevant APIs
            return np.zeros(768)  # Assuming ToolBench model produces 768-dim embeddings
        
        # Generate embeddings for all relevant APIs
        api_embeddings = []
        for api in relevant_apis:
            api_text = f"{api['tool_name']} {api['api_name']} {api['api_description']}"
            with torch.no_grad():
                embedding = self.toolbench_model.encode([api_text], convert_to_tensor=True, device=self.device)
                # Normalize for cosine similarity
                normalized_embedding = F.normalize(embedding, p=2, dim=-1)
                api_embeddings.append(normalized_embedding.cpu().numpy())
        
        # Sum all embeddings
        if api_embeddings:
            embedding_sum = np.sum(api_embeddings, axis=0)
            return embedding_sum
        else:
            return np.zeros(768)
    
    def test_relevant_apis_sum_model(self, test_samples: List[Dict[str, Any]], sample_id: int) -> Dict[str, Any]:
        """Test performance using sum of relevant APIs embeddings instead of query embedding"""
        print(f"\n🧪 Testing relevant APIs sum model for sample {sample_id}...")
        
        results = []
        for i, sample in enumerate(tqdm(test_samples, desc=f"Testing relevant APIs sum model (sample {sample_id})")):
            query = sample['query']
            relevant_apis = sample.get('relevant_apis', [])
            
            if not relevant_apis:
                # Skip samples without relevant APIs
                continue
            
            # Get sum of relevant APIs embeddings
            relevant_apis_embedding_sum = self.get_relevant_apis_embedding_sum(relevant_apis)
            
            # Search for similar APIs using the sum embedding
            similar_apis = self.search_similar_apis(relevant_apis_embedding_sum, similarity_threshold=0.5)
            
            # Evaluate predictions
            metrics = self.evaluate_predictions(similar_apis, relevant_apis)
            
            results.append({
                'query': query,
                'predicted_apis': similar_apis,
                'relevant_apis': relevant_apis,
                'metrics': metrics,
                'relevant_apis_embedding_sum': relevant_apis_embedding_sum.tolist()
            })
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(results)
        
        return {
            'results': results,
            'aggregate_metrics': aggregate_metrics
        }
    
    def test_ideal_performance_model(self, test_samples: List[Dict[str, Any]], sample_id: int) -> Dict[str, Any]:
        """Test ideal performance using perfect knowledge of relevant APIs"""
        print(f"\n🧪 Testing ideal performance model for sample {sample_id}...")
        
        results = []
        for i, sample in enumerate(tqdm(test_samples, desc=f"Testing ideal performance model (sample {sample_id})")):
            query = sample['query']
            relevant_apis = sample.get('relevant_apis', [])
            
            if not relevant_apis:
                # Skip samples without relevant APIs
                continue
            
            # For ideal performance, we directly return the relevant APIs
            # This represents the upper bound of performance
            predicted_apis = []
            for api in relevant_apis:
                api_info = api.copy()
                api_info['similarity_score'] = 1.0  # Perfect similarity
                predicted_apis.append(api_info)
            
            # Evaluate predictions
            metrics = self.evaluate_predictions(predicted_apis, relevant_apis)
            
            results.append({
                'query': query,
                'predicted_apis': predicted_apis,
                'relevant_apis': relevant_apis,
                'metrics': metrics
            })
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(results)
        
        return {
            'results': results,
            'aggregate_metrics': aggregate_metrics
        }

    def get_api_sum_norm_with_toolbench_direction(self, query: str, relevant_apis: List[Dict[str, Any]]) -> np.ndarray:
        """Get embedding with API sum norm and ToolBench model direction"""
        if not relevant_apis:
            # Return zero embedding if no relevant APIs
            return np.zeros(768)
        
        # Get ToolBench model direction (normalized)
        toolbench_embedding = self.get_toolbench_embedding(query)
        toolbench_direction = toolbench_embedding / (np.linalg.norm(toolbench_embedding) + 1e-8)
        
        # Get API embedding sum norm
        api_embedding_sum = self.get_relevant_apis_embedding_sum(relevant_apis)
        api_sum_norm = np.linalg.norm(api_embedding_sum)
        
        # Combine: ToolBench direction * API sum norm
        combined_embedding = toolbench_direction * api_sum_norm
        
        return combined_embedding
    
    def iterative_residual_search_with_combined_embedding(self, query: str, relevant_apis: List[Dict[str, Any]], max_iterations: int = 10, residual_threshold: float = 0.5, beam_size: int = 3) -> List[Dict[str, Any]]:
        """
        Iterative residual-based search using combined embedding (ToolBench direction + API sum norm)
        """
        # Initialize with combined embedding
        initial_embedding = self.get_api_sum_norm_with_toolbench_direction(query, relevant_apis)
        
        # Beam state: (apis, current_embedding, total_similarity_score, is_active)
        beam = [([], initial_embedding, 0.0, True)]
        
        for iteration in range(max_iterations):
            new_beam = []
            
            for apis, current_embedding, total_score, is_active in beam:
                # Skip inactive beams
                if not is_active:
                    new_beam.append((apis, current_embedding, total_score, is_active))
                    continue
                
                # Normalize current embedding for search
                current_norm = np.linalg.norm(current_embedding, axis=-1, keepdims=True)
                normalized_embedding = current_embedding / (current_norm + 1e-8)
                # Search for similar APIs
                similar_apis = self.search_similar_apis(normalized_embedding, similarity_threshold=0.1)
                
                if not similar_apis:
                    # No more APIs found, mark beam as inactive
                    new_beam.append((apis, current_embedding, total_score, False))
                    continue
                
                # Get top beam_size candidates
                top_candidates = similar_apis[:beam_size]
                beam_has_valid_candidate = False
                
                for candidate_api in top_candidates:
                    # Get the API's embedding
                    api_text = f"{candidate_api['tool_name']} {candidate_api['api_name']} {candidate_api['api_description']}"
                    api_embedding = self.toolbench_model.encode([api_text], convert_to_tensor=True, device=self.device)
                    api_embedding = F.normalize(api_embedding, p=2, dim=-1).cpu().numpy()
                    
                    # Calculate residual
                    residual = current_embedding - api_embedding
                    residual_norm = np.linalg.norm(residual)
                    current_norm = np.linalg.norm(current_embedding)
                    
                    # Check if residual magnitude is increasing (bad sign)
                    if residual_norm > current_norm:
                        continue
                    
                    # Valid candidate found
                    beam_has_valid_candidate = True
                    
                    # Calculate new total score (higher similarity = better)
                    new_total_score = total_score + candidate_api['similarity_score']
                    
                    # Add to new beam
                    new_apis = apis + [candidate_api]
                    new_beam.append((new_apis, residual, new_total_score, True))
                
                # If no valid candidates found, mark beam as inactive
                if not beam_has_valid_candidate:
                    new_beam.append((apis, current_embedding, total_score, False))
            
            # Keep top beam_size candidates based on total score
            new_beam.sort(key=lambda x: x[2], reverse=True)
            beam = new_beam[:beam_size]
            
            # Check if all beams are inactive
            all_inactive = all(not state[3] for state in beam)
            if all_inactive:
                break
            
            # Check if all residuals are small enough
            active_beams = [state for state in beam if state[3]]
            if active_beams:
                all_small_residuals = all(np.linalg.norm(state[1]) < residual_threshold for state in active_beams)
                if all_small_residuals:
                    break
        
        # Return the beam with smallest residual magnitude
        if beam:
            # Sort by residual magnitude (smallest first)
            beam.sort(key=lambda x: np.linalg.norm(x[1]))
            best_apis, best_residual, _, _ = beam[0]
            return best_apis
        else:
            return []
    
    def iterative_residual_search_greedy_with_combined_embedding(self, query: str, relevant_apis: List[Dict[str, Any]], max_iterations: int = 10, residual_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Greedy iterative residual-based search using combined embedding (ToolBench direction + API sum norm)
        """
        predicted_apis = []
        current_embedding = self.get_api_sum_norm_with_toolbench_direction(query, relevant_apis)
        
        for iteration in range(max_iterations):
            # Search for most similar API
            similar_apis = self.search_similar_apis(current_embedding, similarity_threshold=0.1)
            
            if not similar_apis:
                break
            
            # Get the most similar API
            best_api = similar_apis[0]
            predicted_apis.append(best_api)
            
            # Get the API's embedding
            api_text = f"{best_api['tool_name']} {best_api['api_name']} {best_api['api_description']}"
            api_embedding = self.toolbench_model.encode([api_text], convert_to_tensor=True, device=self.device)
            api_embedding = F.normalize(api_embedding, p=2, dim=-1).cpu().numpy()
            
            # Calculate residual
            residual = current_embedding - api_embedding
            residual_norm = np.linalg.norm(residual)
            current_norm = np.linalg.norm(current_embedding)
            
            # Check if residual magnitude is increasing (bad sign)
            if residual_norm > current_norm:
                break
            
            # Check if residual is small enough
            if residual_norm < residual_threshold:
                break
            
            # Update current embedding to residual
            current_embedding = residual
        
        return predicted_apis
    
    def test_combined_embedding_model(self, test_samples: List[Dict[str, Any]], sample_id: int, use_beam_search: bool = True) -> Dict[str, Any]:
        """Test performance using combined embedding (ToolBench direction + API sum norm) with iterative search"""
        search_method = "beam search" if use_beam_search else "greedy search"
        print(f"\n🧪 Testing combined embedding model (iterative {search_method}) for sample {sample_id}...")
        
        results = []
        for i, sample in enumerate(tqdm(test_samples, desc=f"Testing combined embedding model ({search_method}, sample {sample_id})")):
            query = sample['query']
            relevant_apis = sample.get('relevant_apis', [])
            
            if not relevant_apis:
                # Skip samples without relevant APIs
                continue
            
            # Use iterative residual search with combined embedding
            if use_beam_search:
                predicted_apis = self.iterative_residual_search_with_combined_embedding(query, relevant_apis, beam_size=3)
            else:
                predicted_apis = self.iterative_residual_search_greedy_with_combined_embedding(query, relevant_apis)
            
            # Evaluate predictions
            metrics = self.evaluate_predictions(predicted_apis, relevant_apis)
            
            # Get combined embedding for analysis
            combined_embedding = self.get_api_sum_norm_with_toolbench_direction(query, relevant_apis)
            
            results.append({
                'query': query,
                'predicted_apis': predicted_apis,
                'relevant_apis': relevant_apis,
                'metrics': metrics,
                'combined_embedding': combined_embedding.tolist()
            })
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(results)
        
        return {
            'results': results,
            'aggregate_metrics': aggregate_metrics
        }


def main():
    """Main comparison function"""
    print("🧪 ToolBench vs Relevant APIs Embedding Sum Performance Comparison")
    print("=" * 70)
    
    # Initialize comparator
    comparator = ModelPerformanceComparator()
    
    # Load only ToolBench model (no trained model needed for this comparison)
    if not comparator.load_models(load_trained_model=False):
        print("❌ Failed to load ToolBench model. Exiting.")
        return
    
    # Run full comparison using complete ToolBench vector database
    print(f"\n🔍 Running comparison: ToolBench query embedding vs Relevant APIs embedding sum...")
    comparator.run_full_comparison()
    
    print("\n✅ Comparison completed!")


def run_relevant_apis_sum_experiment():
    """Run experiment comparing query embedding vs relevant APIs embedding sum"""
    print("🧪 Relevant APIs Sum Experiment")
    print("=" * 50)
    
    # Initialize comparator
    comparator = ModelPerformanceComparator()
    
    # Load only ToolBench model (no trained model needed)
    if not comparator.load_models(load_trained_model=False):
        print("❌ Failed to load ToolBench model. Exiting.")
        return
    
    # Load all test data from ToolBench instruction files
    all_samples = comparator.load_test_instruction_data()
    
    if not all_samples:
        print("❌ No test samples found. Exiting.")
        return
    
    print(f"📊 Total available samples: {len(all_samples)}")
    
    # Load full vector database
    if not comparator.load_vector_database():
        print("❌ Failed to load vector database. Exiting.")
        return
    
    # Create multiple samples for testing
    all_results = []
    
    for sample_id in range(1, NUM_SAMPLES + 1):
        print(f"\n" + "="*60)
        print(f"🧪 PROCESSING SAMPLE {sample_id}/{NUM_SAMPLES}")
        print("="*60)
        
        # Create sample queries
        sample_queries = comparator.create_sample_queries(all_samples, sample_id)
        
        # Test ToolBench model (query embedding)
        toolbench_results = comparator.test_toolbench_model(sample_queries, sample_id)
        
        # Test relevant APIs sum model
        relevant_apis_sum_results = comparator.test_relevant_apis_sum_model(sample_queries, sample_id)
        
        # Test ideal performance model
        ideal_results = comparator.test_ideal_performance_model(sample_queries, sample_id)
        
        # Print comparison
        comparator.print_relevant_apis_sum_comparison(toolbench_results, relevant_apis_sum_results, ideal_results, sample_id)
        
        # Store results
        all_results.append({
            'sample_id': sample_id,
            'toolbench_results': toolbench_results,
            'relevant_apis_sum_results': relevant_apis_sum_results,
            'ideal_results': ideal_results
        })
    
    # Print overall summary
    print_relevant_apis_sum_overall_summary(all_results)
    
    # Save detailed results
    output_path = "/home/jhlee/librarian/relevant_apis_sum_experiment_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Detailed results saved to: {output_path}")


def print_relevant_apis_sum_overall_summary(all_results: List[Dict[str, Any]]):
    """Print overall summary for relevant APIs sum experiment"""
    print("\n" + "="*80)
    print("📊 OVERALL SUMMARY - RELEVANT APIS SUM EXPERIMENT")
    print("="*80)
    
    toolbench_f1_scores = []
    relevant_apis_sum_f1_scores = []
    ideal_f1_scores = []
    
    for result in all_results:
        toolbench_f1_scores.append(result['toolbench_results']['aggregate_metrics']['f1_score'])
        relevant_apis_sum_f1_scores.append(result['relevant_apis_sum_results']['aggregate_metrics']['f1_score'])
        ideal_f1_scores.append(result['ideal_results']['aggregate_metrics']['f1_score'])
    
    print(f"{'Model':<25} {'Avg F1':<15} {'Std F1':<15} {'Min F1':<15} {'Max F1':<15}")
    print("-" * 80)
    print(f"{'ToolBench (Query)':<25} {np.mean(toolbench_f1_scores):<15.4f} {np.std(toolbench_f1_scores):<15.4f} {np.min(toolbench_f1_scores):<15.4f} {np.max(toolbench_f1_scores):<15.4f}")
    print(f"{'Relevant APIs Sum':<25} {np.mean(relevant_apis_sum_f1_scores):<15.4f} {np.std(relevant_apis_sum_f1_scores):<15.4f} {np.min(relevant_apis_sum_f1_scores):<15.4f} {np.max(relevant_apis_sum_f1_scores):<15.4f}")
    print(f"{'Ideal Performance':<25} {np.mean(ideal_f1_scores):<15.4f} {np.std(ideal_f1_scores):<15.4f} {np.min(ideal_f1_scores):<15.4f} {np.max(ideal_f1_scores):<15.4f}")
    
    print(f"\n" + "="*80)
    print("🏆 OVERALL EXPERIMENT ANALYSIS")
    print("="*80)
    
    avg_toolbench_f1 = np.mean(toolbench_f1_scores)
    avg_relevant_apis_sum_f1 = np.mean(relevant_apis_sum_f1_scores)
    avg_ideal_f1 = np.mean(ideal_f1_scores)
    
    print(f"📊 Average Performance Gap to Ideal:")
    print(f"   ToolBench (Query): {avg_ideal_f1 - avg_toolbench_f1:.4f} ({(avg_toolbench_f1/avg_ideal_f1)*100:.1f}% of ideal)")
    print(f"   Relevant APIs Sum: {avg_ideal_f1 - avg_relevant_apis_sum_f1:.4f} ({(avg_relevant_apis_sum_f1/avg_ideal_f1)*100:.1f}% of ideal)")
    
    if avg_relevant_apis_sum_f1 > avg_toolbench_f1:
        print("🎉 Relevant APIs sum performs better than query embedding on average!")
        improvement = ((avg_relevant_apis_sum_f1 - avg_toolbench_f1) / avg_toolbench_f1) * 100
        print(f"📈 Improvement: {improvement:.2f}%")
    elif avg_relevant_apis_sum_f1 < avg_toolbench_f1:
        print("🎉 Query embedding performs better than relevant APIs sum on average!")
        degradation = ((avg_toolbench_f1 - avg_relevant_apis_sum_f1) / avg_toolbench_f1) * 100
        print(f"📉 Degradation: {degradation:.2f}%")
    else:
        print("🤝 Both approaches perform similarly on average!")
    
    print(f"\n🔍 Key Insights:")
    print(f"   • Query embedding vs Relevant APIs sum comparison")
    print(f"   • Performance gap to ideal (upper bound)")
    print(f"   • Whether using ground truth API embeddings helps")
    print(f"   • Model's ability to learn query-to-API mapping")


def run_three_methods_comparison():
    """Run experiment comparing three methods: query embedding, API sum, and combined embedding"""
    print("🧪 Three Methods Comparison Experiment")
    print("=" * 50)
    
    # Initialize comparator
    comparator = ModelPerformanceComparator()
    
    # Load only ToolBench model (no trained model needed)
    if not comparator.load_models(load_trained_model=False):
        print("❌ Failed to load ToolBench model. Exiting.")
        return
    
    # Load all test data from ToolBench instruction files
    all_samples = comparator.load_test_instruction_data()
    
    if not all_samples:
        print("❌ No test samples found. Exiting.")
        return
    
    print(f"📊 Total available samples: {len(all_samples)}")
    
    # Load full vector database
    if not comparator.load_vector_database():
        print("❌ Failed to load vector database. Exiting.")
        return
    
    # Create multiple samples for testing
    all_results = []
    
    for sample_id in range(1, NUM_SAMPLES + 1):
        print(f"\n" + "="*60)
        print(f"🧪 PROCESSING SAMPLE {sample_id}/{NUM_SAMPLES}")
        print("="*60)
        
        # Create sample queries
        sample_queries = comparator.create_sample_queries(all_samples, sample_id)
        
        # Method 1: ToolBench query embedding
        toolbench_results = comparator.test_toolbench_model(sample_queries, sample_id)
        
        # Method 2: API embedding sum
        api_sum_results = comparator.test_relevant_apis_sum_model(sample_queries, sample_id)
        
        # Method 3: Combined embedding with beam search
        combined_beam_results = comparator.test_combined_embedding_model(sample_queries, sample_id, use_beam_search=True)
        
        # Method 4: Combined embedding with greedy search
        combined_greedy_results = comparator.test_combined_embedding_model(sample_queries, sample_id, use_beam_search=False)
        
        # Test ideal performance model
        ideal_results = comparator.test_ideal_performance_model(sample_queries, sample_id)
        
        # Print comparison
        print_three_methods_comparison(toolbench_results, api_sum_results, combined_beam_results, combined_greedy_results, ideal_results, sample_id)
        
        # Store results
        all_results.append({
            'sample_id': sample_id,
            'toolbench_results': toolbench_results,
            'api_sum_results': api_sum_results,
            'combined_beam_results': combined_beam_results,
            'combined_greedy_results': combined_greedy_results,
            'ideal_results': ideal_results
        })
    
    # Print overall summary
    print_three_methods_overall_summary(all_results)
    
    # Save detailed results
    output_path = "/home/jhlee/librarian/three_methods_comparison_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Detailed results saved to: {output_path}")


def print_three_methods_comparison(toolbench_results: Dict[str, Any], api_sum_results: Dict[str, Any], 
                                 combined_beam_results: Dict[str, Any], combined_greedy_results: Dict[str, Any],
                                 ideal_results: Dict[str, Any], sample_id: int):
    """Print comparison results for three methods experiment"""
    print(f"\n" + "="*80)
    print(f"📊 THREE METHODS COMPARISON - SAMPLE {sample_id}")
    print("="*80)
    
    toolbench_metrics = toolbench_results['aggregate_metrics']
    api_sum_metrics = api_sum_results['aggregate_metrics']
    combined_beam_metrics = combined_beam_results['aggregate_metrics']
    combined_greedy_metrics = combined_greedy_results['aggregate_metrics']
    ideal_metrics = ideal_results['aggregate_metrics']
    
    print(f"{'Metric':<20} {'Query':<12} {'API Sum':<12} {'Combined(Beam)':<15} {'Combined(Greedy)':<15} {'Ideal':<12}")
    print("-" * 80)
    print(f"{'Precision':<20} {toolbench_metrics['precision']:<12.4f} {api_sum_metrics['precision']:<12.4f} {combined_beam_metrics['precision']:<15.4f} {combined_greedy_metrics['precision']:<15.4f} {ideal_metrics['precision']:<12.4f}")
    print(f"{'Recall':<20} {toolbench_metrics['recall']:<12.4f} {api_sum_metrics['recall']:<12.4f} {combined_beam_metrics['recall']:<15.4f} {combined_greedy_metrics['recall']:<15.4f} {ideal_metrics['recall']:<12.4f}")
    print(f"{'F1 Score':<20} {toolbench_metrics['f1_score']:<12.4f} {api_sum_metrics['f1_score']:<12.4f} {combined_beam_metrics['f1_score']:<15.4f} {combined_greedy_metrics['f1_score']:<15.4f} {ideal_metrics['f1_score']:<12.4f}")
    print(f"{'Avg Predicted':<20} {toolbench_metrics['avg_predicted_count']:<12.2f} {api_sum_metrics['avg_predicted_count']:<12.2f} {combined_beam_metrics['avg_predicted_count']:<15.2f} {combined_greedy_metrics['avg_predicted_count']:<15.2f} {ideal_metrics['avg_predicted_count']:<12.2f}")
    print(f"{'Avg Relevant':<20} {toolbench_metrics['avg_relevant_count']:<12.2f} {api_sum_metrics['avg_relevant_count']:<12.2f} {combined_beam_metrics['avg_relevant_count']:<15.2f} {combined_greedy_metrics['avg_relevant_count']:<15.2f} {ideal_metrics['avg_relevant_count']:<12.2f}")
    print(f"{'Avg Intersection':<20} {toolbench_metrics['avg_intersection_count']:<12.2f} {api_sum_metrics['avg_intersection_count']:<12.2f} {combined_beam_metrics['avg_intersection_count']:<15.2f} {combined_greedy_metrics['avg_intersection_count']:<15.2f} {ideal_metrics['avg_intersection_count']:<12.2f}")
    
    print(f"\n" + "="*80)
    print(f"🏆 THREE METHODS ANALYSIS - SAMPLE {sample_id}")
    print("="*80)
    
    # Find best performing method
    methods = [
        ("Query Embedding", toolbench_metrics['f1_score']),
        ("API Sum", api_sum_metrics['f1_score']),
        ("Combined (Beam)", combined_beam_metrics['f1_score']),
        ("Combined (Greedy)", combined_greedy_metrics['f1_score'])
    ]
    
    best_method = max(methods, key=lambda x: x[1])
    print(f"🥇 Best performing method: {best_method[0]} (F1: {best_method[1]:.4f})")
    
    # Compare with ideal
    ideal_f1 = ideal_metrics['f1_score']
    print(f"📊 Performance gap to ideal:")
    for method_name, f1_score in methods:
        gap = ideal_f1 - f1_score
        percentage = (f1_score / ideal_f1) * 100
        print(f"   {method_name}: {gap:.4f} ({percentage:.1f}% of ideal)")
    
    # Compare beam vs greedy for combined method
    beam_f1 = combined_beam_metrics['f1_score']
    greedy_f1 = combined_greedy_metrics['f1_score']
    
    if beam_f1 > greedy_f1:
        print(f"🎯 Beam search performs better than greedy for combined method!")
        improvement = ((beam_f1 - greedy_f1) / greedy_f1) * 100
        print(f"📈 Improvement: {improvement:.2f}%")
    elif beam_f1 < greedy_f1:
        print(f"🎯 Greedy search performs better than beam for combined method!")
        degradation = ((greedy_f1 - beam_f1) / beam_f1) * 100
        print(f"📉 Degradation: {degradation:.2f}%")
    else:
        print("🤝 Beam and greedy perform similarly for combined method!")


def print_three_methods_overall_summary(all_results: List[Dict[str, Any]]):
    """Print overall summary for three methods comparison"""
    print("\n" + "="*80)
    print("📊 OVERALL SUMMARY - THREE METHODS COMPARISON")
    print("="*80)
    
    toolbench_f1_scores = []
    api_sum_f1_scores = []
    combined_beam_f1_scores = []
    combined_greedy_f1_scores = []
    ideal_f1_scores = []
    
    for result in all_results:
        toolbench_f1_scores.append(result['toolbench_results']['aggregate_metrics']['f1_score'])
        api_sum_f1_scores.append(result['api_sum_results']['aggregate_metrics']['f1_score'])
        combined_beam_f1_scores.append(result['combined_beam_results']['aggregate_metrics']['f1_score'])
        combined_greedy_f1_scores.append(result['combined_greedy_results']['aggregate_metrics']['f1_score'])
        ideal_f1_scores.append(result['ideal_results']['aggregate_metrics']['f1_score'])
    
    print(f"{'Method':<20} {'Avg F1':<12} {'Std F1':<12} {'Min F1':<12} {'Max F1':<12}")
    print("-" * 80)
    print(f"{'Query Embedding':<20} {np.mean(toolbench_f1_scores):<12.4f} {np.std(toolbench_f1_scores):<12.4f} {np.min(toolbench_f1_scores):<12.4f} {np.max(toolbench_f1_scores):<12.4f}")
    print(f"{'API Sum':<20} {np.mean(api_sum_f1_scores):<12.4f} {np.std(api_sum_f1_scores):<12.4f} {np.min(api_sum_f1_scores):<12.4f} {np.max(api_sum_f1_scores):<12.4f}")
    print(f"{'Combined (Beam)':<20} {np.mean(combined_beam_f1_scores):<12.4f} {np.std(combined_beam_f1_scores):<12.4f} {np.min(combined_beam_f1_scores):<12.4f} {np.max(combined_beam_f1_scores):<12.4f}")
    print(f"{'Combined (Greedy)':<20} {np.mean(combined_greedy_f1_scores):<12.4f} {np.std(combined_greedy_f1_scores):<12.4f} {np.min(combined_greedy_f1_scores):<12.4f} {np.max(combined_greedy_f1_scores):<12.4f}")
    print(f"{'Ideal Performance':<20} {np.mean(ideal_f1_scores):<12.4f} {np.std(ideal_f1_scores):<12.4f} {np.min(ideal_f1_scores):<12.4f} {np.max(ideal_f1_scores):<12.4f}")
    
    print(f"\n" + "="*80)
    print("🏆 OVERALL EXPERIMENT ANALYSIS")
    print("="*80)
    
    avg_toolbench_f1 = np.mean(toolbench_f1_scores)
    avg_api_sum_f1 = np.mean(api_sum_f1_scores)
    avg_combined_beam_f1 = np.mean(combined_beam_f1_scores)
    avg_combined_greedy_f1 = np.mean(combined_greedy_f1_scores)
    avg_ideal_f1 = np.mean(ideal_f1_scores)
    
    # Find best method
    methods = [
        ("Query Embedding", avg_toolbench_f1),
        ("API Sum", avg_api_sum_f1),
        ("Combined (Beam)", avg_combined_beam_f1),
        ("Combined (Greedy)", avg_combined_greedy_f1)
    ]
    
    best_method = max(methods, key=lambda x: x[1])
    print(f"🥇 Best performing method: {best_method[0]} (F1: {best_method[1]:.4f})")
    
    print(f"📊 Average Performance Gap to Ideal:")
    for method_name, f1_score in methods:
        gap = avg_ideal_f1 - f1_score
        percentage = (f1_score / avg_ideal_f1) * 100
        print(f"   {method_name}: {gap:.4f} ({percentage:.1f}% of ideal)")
    
    # Compare beam vs greedy
    if avg_combined_beam_f1 > avg_combined_greedy_f1:
        print(f"🎯 Beam search performs better than greedy for combined method!")
        improvement = ((avg_combined_beam_f1 - avg_combined_greedy_f1) / avg_combined_greedy_f1) * 100
        print(f"📈 Improvement: {improvement:.2f}%")
    elif avg_combined_beam_f1 < avg_combined_greedy_f1:
        print(f"🎯 Greedy search performs better than beam for combined method!")
        degradation = ((avg_combined_greedy_f1 - avg_combined_beam_f1) / avg_combined_beam_f1) * 100
        print(f"📉 Degradation: {degradation:.2f}%")
    else:
        print("🤝 Beam and greedy perform similarly for combined method!")
    
    print(f"\n🔍 Key Insights:")
    print(f"   • Query embedding vs API sum vs Combined embedding comparison")
    print(f"   • Beam search vs Greedy search for iterative methods")
    print(f"   • Performance gap to ideal (upper bound)")
    print(f"   • Effectiveness of combining query direction with API magnitude")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "relevant_apis_sum":
        # Run relevant APIs sum experiment
        run_relevant_apis_sum_experiment()
    elif len(sys.argv) > 1 and sys.argv[1] == "three_methods":
        # Run three methods comparison experiment
        run_three_methods_comparison()
    else:
        # Run original full comparison
        main() 