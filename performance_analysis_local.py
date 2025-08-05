import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple
import faiss
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoConfig, AutoTokenizer

class DirectionFocusedQueryEmbeddingModel(nn.Module):
    """Dynamic direction-focused model with query-specific scale prediction"""
    
    def __init__(self, model_name: str, dropout: float = 0.1):
        super(DirectionFocusedQueryEmbeddingModel, self).__init__()
        
        # Single encoder with standard dropout
        config = AutoConfig.from_pretrained(model_name)
        config.attention_probs_dropout_prob = dropout
        config.hidden_dropout_prob = dropout
        
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        
        # Dynamic scale prediction network - can predict up to infinity
        self.scale_predictor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            # Use Softplus + 1 instead of Sigmoid to predict all positive values >= 1
            # Softplus(x) + 1 = log(1 + exp(x)) + 1
            # This allows prediction from 1 to infinity
        )
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        """Forward pass with dynamic scale prediction"""
        # Get encoder outputs
        outputs = self.encoder(
            input_ids, 
            attention_mask, 
            token_type_ids, 
            output_hidden_states=True, 
            return_dict=True
        )
        
        # Use CLS token
        cls_token = outputs.last_hidden_state[:, 0]  # [batch, 768]
        
        # Predict dynamic scale factor for each query (1 ~ infinity)
        raw_scale = self.scale_predictor(cls_token)  # [batch, 1]
        dynamic_scale = F.softplus(raw_scale) + 1.0  # [batch, 1] -> minimum value 1.0
        
        # Apply scaling to match target magnitudes
        # Expand dynamic_scale to [batch, 1] for broadcasting
        scaled_embedding = dynamic_scale.squeeze(-1).unsqueeze(-1) * F.normalize(cls_token, p=2, dim=1)
        
        return scaled_embedding

class VectorDBPerformanceAnalyzer:
    def __init__(self, full_db_dir: str, trained_model_path: str = None):
        """
        Class for Vector DB performance analysis (using Dynamic Direction-focused model)
        
        Args:
            full_db_dir: Full information Vector DB directory (fixed to Full text DB)
            trained_model_path: Path to trained Dynamic Direction-focused model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load baseline ToolBench model (for API embeddings)
        print("Loading ToolBench's trained retriever model...")
        self.model = SentenceTransformer('ToolBench/ToolBench_IR_bert_based_uncased').to(self.device)
        
        # Load trained Dynamic Direction-focused model (for query embeddings)
        if trained_model_path and os.path.exists(trained_model_path):
            print(f"Loading trained Dynamic Direction-focused model from {trained_model_path}...")
            try:
                # Use weights_only=False for PyTorch 2.6+ compatibility
                checkpoint = torch.load(trained_model_path, map_location=self.device, weights_only=False)
                
                # Use model name from training script
                model_name = "ToolBench/ToolBench_IR_bert_based_uncased"
                print(f"Using model_name: {model_name}")
                
                # Initialize Dynamic Direction-focused model
                self.trained_model = DirectionFocusedQueryEmbeddingModel(model_name)
                
                # Load state dict
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    self.trained_model.load_state_dict(state_dict)
                else:
                    raise ValueError("Could not find model_state_dict in checkpoint")
                
                self.trained_model.to(self.device)
                self.trained_model.eval()
                
                # Load tokenizer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                except:
                    print(f"Warning: Could not load tokenizer for {model_name}, using bert-base-uncased")
                    self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                
                print("Trained Dynamic Direction-focused model loaded successfully!")
            except Exception as e:
                print(f"Error loading trained model: {e}")
                raise RuntimeError(f"Failed to load trained model from {trained_model_path}. Error: {e}")
        else:
            raise FileNotFoundError(f"Trained model path not found: {trained_model_path}")
        
        # Load Faiss index
        index_path = f"{full_db_dir}/faiss_index.bin"
        self.index = faiss.read_index(index_path)
        print(f"Loaded Faiss index with {self.index.ntotal} vectors")
        
        # Load metadata
        metadata_path = f"{full_db_dir}/metadata.pkl"
        with open(metadata_path, 'rb') as f:
            self.documents = pickle.load(f)
        print(f"Loaded {len(self.documents)} documents")
        
        # Load model info
        model_info_path = f"{full_db_dir}/model_info.json"
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        print(f"Model info: {self.model_info}")
        
    def get_query_embedding(self, query: str, normalize: bool = True) -> np.ndarray:
        """Convert query text to embedding (using Dynamic Direction-focused model)"""
        if self.trained_model is not None:
            # Use trained Dynamic Direction-focused model
            with torch.no_grad():
                # Tokenization
                encoding = self.tokenizer(
                    query,
                    truncation=True,
                    padding='max_length',
                    max_length=256,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                token_type_ids = encoding['token_type_ids'].to(self.device)
                
                # Model inference
                embedding = self.trained_model(
                    input_ids, 
                    attention_mask, 
                    token_type_ids
                )

                # Move to CPU and convert to numpy
                embedding = embedding.cpu().numpy()
                if normalize:
                    faiss.normalize_L2(embedding)
                return embedding[0]  # Remove first dimension
        else:
            # Use baseline ToolBench model
            embedding = self.model.encode([query], convert_to_tensor=True)
            embedding = embedding.cpu().numpy()
            if normalize:
                faiss.normalize_L2(embedding)
            return embedding[0]
    
    def get_query_embedding_for_iterative(self, query: str) -> np.ndarray:
        """Generate unnormalized query embedding for iterative search"""
        return self.get_query_embedding(query, normalize=False)
        
    def get_embedding_by_index(self, idx: int) -> np.ndarray:
        """Extract embedding from Faiss index by specific index"""
        try:
            # Extract embedding directly from Faiss index
            embedding = self.index.reconstruct(int(idx))
            return embedding
        except:
            # Fallback: generate embedding from document text
            doc = self.documents[idx]
            embedding = self.model.encode([doc['text']], convert_to_tensor=True)
            embedding = embedding.cpu().numpy()
            faiss.normalize_L2(embedding)
            return embedding[0]
        
    def get_api_embedding(self, tool_name: str, api_name: str) -> np.ndarray:
        """
        Function to get embedding for specific API
        Generate embedding using ToolBench model with full text
        
        Args:
            tool_name: Tool name
            api_name: API name
            
        Returns:
            API embedding vector
        """
        # Find actual document for the API in Vector DB (using full text)
        for doc in self.documents:
            if (doc['metadata']['tool_name'] == tool_name and 
                doc['metadata']['api_name'] == api_name):
                # Generate embedding using ToolBench model from full text
                embedding = self.model.encode([doc['text']], convert_to_tensor=True)
                embedding = embedding.cpu().numpy()
                faiss.normalize_L2(embedding)
                return embedding[0]  # Remove first dimension
        
        # Fallback if API not found
        print(f"Warning: API {tool_name}_{api_name} not found in Vector DB, using fallback")
        api_text = f"{tool_name} {api_name}"
        embedding = self.model.encode([api_text], convert_to_tensor=True)
        embedding = embedding.cpu().numpy()
        faiss.normalize_L2(embedding)
        return embedding[0]  # Remove first dimension
    
    def calculate_threshold(self, test_data: List[Dict[str, Any]]) -> float:
        """
        Calculate threshold as minimum similarity between relevant APIs and queries from all test data
        Save calculated threshold to file for reuse
        
        Args:
            test_data: Test data list
            
        Returns:
            threshold value
        """
        threshold_file = "threshold_cache_direction_focused_dynamic.json"
        
        # Check if cached threshold exists
        if os.path.exists(threshold_file):
            try:
                with open(threshold_file, 'r') as f:
                    cached_data = json.load(f)
                    if 'threshold' in cached_data:
                        print(f"Using cached threshold: {cached_data['threshold']:.4f}")
                        return cached_data['threshold']
            except:
                pass
        
        print("Calculating threshold from all test data...")
        min_similarity = float('inf')
        
        for test_item in test_data:
            query = test_item['query']
            relevant_apis = test_item['relevant_apis']
            
            # Generate query embedding
            query_embedding = self.get_query_embedding(query)
            
            # Calculate similarity with each relevant API
            for tool_name, api_name in relevant_apis:
                api_embedding = self.get_api_embedding(tool_name, api_name)
                similarity = np.dot(query_embedding, api_embedding)
                min_similarity = min(min_similarity, similarity)
        
        print(f"Calculated threshold: {min_similarity:.4f}")
        
        # Save threshold
        with open(threshold_file, 'w') as f:
            json.dump({'threshold': float(min_similarity)}, f, indent=2)
        
        return float(min_similarity)
    
    def iterative_greedy_search(self, query: str, relevant_apis: List[List[str]], 
                               threshold: float, remove_duplicates: bool = True, 
                               track_termination: bool = False) -> List[Dict[str, Any]]:
        """
        API search using iterative greedy approach (using trained Dynamic Direction-focused model)
        
        Args:
            query: search query
            relevant_apis: relevant API list [[tool_name, api_name], ...]
            threshold: similarity threshold
            remove_duplicates: whether to remove duplicate APIs (default: True)
            track_termination: whether to track termination conditions (default: False)
            
        Returns:
            list of found APIs
        """
        # 1. Generate query embedding using trained model (unnormalized for residual, normalized for search)
        query_embedding_unnorm = self.get_query_embedding_for_iterative(query)
        query_embedding_norm = self.get_query_embedding(query, normalize=True)
        
        # 2. Iterative greedy search
        found_apis = []
        current_query_unnorm = query_embedding_unnorm.copy()  # for residual calculation
        current_query_norm = query_embedding_norm.copy()      # for search
        initial_norm = np.linalg.norm(current_query_unnorm)
        
        # Set for duplicate removal (only used when remove_duplicates=True)
        found_api_keys = set() if remove_duplicates else None
        
        # Track termination conditions
        termination_reason = None
        iteration_count = 0
        
        while True:
            iteration_count += 1
            
            # Search top1 from Vector DB using current query embedding (use normalized)
            current_query_reshaped = current_query_norm.reshape(1, -1)
            scores, indices = self.index.search(current_query_reshaped, 1)

           
            # Check top1 result
            if indices[0][0] == -1:  # invalid index
                if track_termination:
                    termination_reason = "invalid_index"
                break
                
            score = scores[0][0]
            idx = indices[0][0]
            
            # Check termination conditions
            if score < threshold:
                if track_termination:
                    termination_reason = "score_below_threshold"
                break
                
            # Check current query embedding norm (based on unnormalized)
            current_norm = np.linalg.norm(current_query_unnorm)
            if current_norm < 0.5:
                if track_termination:
                    termination_reason = "query_norm_too_small"
                break
            
            # Get API information
            doc = self.documents[idx]
            doc_embedding = self.get_embedding_by_index(idx)
            
            # Check duplicates (when remove_duplicates=True)
            if remove_duplicates:
                api_key = f"{doc['metadata']['tool_name']}_{doc['metadata']['api_name']}"
                if api_key in found_api_keys:
                    # Found duplicate API, calculate residual and continue
                    residual = current_query_unnorm - doc_embedding
                    residual_norm = np.linalg.norm(residual)
                    
                    # Check if residual norm increases
                    if residual_norm > current_norm:
                        break
                    
                    # Update query embedding
                    current_query_unnorm = residual
                    if residual_norm > 0:
                        current_query_norm = residual / residual_norm
                    else:
                        break
                    continue  # move to next iteration
            
            # Calculate residual (remove found API embedding from unnormalized query)
            residual = current_query_unnorm - doc_embedding
            residual_norm = np.linalg.norm(residual)
            
            # Check if residual norm increases
            if residual_norm > current_norm:
                if track_termination:
                    termination_reason = "residual_norm_increased"
                break
            
            # Add API
            best_api = {
                'tool_name': doc['metadata']['tool_name'],
                'api_name': doc['metadata']['api_name'],
                'score': float(score)
            }
            found_apis.append(best_api)
            
            # Add to duplicate tracking set (when remove_duplicates=True)
            if remove_duplicates:
                api_key = f"{doc['metadata']['tool_name']}_{doc['metadata']['api_name']}"
                found_api_keys.add(api_key)
            
            # Update query embedding (don't normalize)
            current_query_unnorm = residual
            
            # Also update normalized query (normalize residual)
            if residual_norm > 0:
                current_query_norm = residual / residual_norm
            else:
                if track_termination:
                    termination_reason = "residual_zero"
                break  # terminate if residual is zero
        
        if track_termination:
            return {
                'apis': found_apis,
                'termination_reason': termination_reason,
                'iteration_count': iteration_count
            }
        else:
            return found_apis
    
    def beam_search_iterative(self, query: str, relevant_apis: List[List[str]], 
                            threshold: float, beam_size: int = 5, 
                            track_termination: bool = False) -> List[Dict[str, Any]]:
        """
        API search using beam search with iterative approach (including duplicate API combination merge)
        
        Args:
            query: search query
            relevant_apis: relevant API list [[tool_name, api_name], ...]
            threshold: similarity threshold
            beam_size: beam size (number of candidates to maintain)
            track_termination: whether to track termination conditions (default: False)
            
        Returns:
            list of found APIs (from the best beam)
        """
        # 1. Generate query embedding (unnormalized for residual, normalized for search)
        query_embedding_unnorm = self.get_query_embedding_for_iterative(query)
        query_embedding_norm = self.get_query_embedding(query, normalize=True)
        
        # 2. Initialize beam: each beam is (api_list, current_query_unnorm, current_query_norm, residual_norm)
        initial_residual_norm = np.linalg.norm(query_embedding_unnorm)
        beams = [([], query_embedding_unnorm.copy(), query_embedding_norm.copy(), initial_residual_norm)]  # (API list, unnorm query, norm query, residual norm)
        
        # Track termination conditions
        termination_reason = None
        iteration_count = 0
        
        while True:
            iteration_count += 1
            new_beams = []
            # Track API combinations already selected from all beams
            all_existing_combinations = set()
            
            # Search for each beam
            for api_list, current_query_unnorm, current_query_norm, current_residual_norm in beams:
                # Convert current beam's API combination to set
                current_api_set = frozenset([f"{api['tool_name']}_{api['api_name']}" for api in api_list])
                
                # Search top beam_size from Vector DB using current query embedding (use normalized)
                current_query_reshaped = current_query_norm.reshape(1, -1)
                scores, indices = self.index.search(current_query_reshaped, beam_size)
                
                # Create new beam for each search result
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:  # invalid index
                        continue
                        
                    # Check termination conditions
                    if score < threshold:
                        continue
                    
                    # Check current query embedding norm (based on unnormalized)
                    current_norm = np.linalg.norm(current_query_unnorm)
                    if current_norm < 0.5:
                        continue
                    
                    # Get API information
                    doc = self.documents[idx]
                    
                    # Add new API
                    new_api = {
                        'tool_name': doc['metadata']['tool_name'],
                        'api_name': doc['metadata']['api_name'],
                        'score': float(score)
                    }
                    
                    # Check duplicate API within current beam
                    api_key = f"{new_api['tool_name']}_{new_api['api_name']}"
                    if any(f"{api['tool_name']}_{api['api_name']}" == api_key for api in api_list):
                        continue
                    
                    # Create new API combination
                    new_api_list = api_list + [new_api]
                    new_api_set = frozenset([f"{api['tool_name']}_{api['api_name']}" for api in new_api_list])
                    
                    # Check if combination already selected in other beam (remove duplicates between beams)
                    if new_api_set in all_existing_combinations:
                        continue
                    
                    # Calculate residual (expensive operation) - only execute when not duplicate
                    doc_embedding = self.get_embedding_by_index(idx)
                    residual = current_query_unnorm - doc_embedding
                    residual_norm = np.linalg.norm(residual)
                    
                    # Check if residual norm increases
                    if residual_norm > current_norm:
                        continue
                    
                    # Calculate normalized residual
                    if residual_norm > 0:
                        new_query_norm = residual / residual_norm
                    else:
                        continue  # skip if residual is zero
                    
                    new_beams.append((new_api_list, residual, new_query_norm, residual_norm))
                    
                    # Add newly added combination to tracking set
                    all_existing_combinations.add(new_api_set)
            
            # Exit if no new beams
            if not new_beams:
                if track_termination:
                    termination_reason = "no_new_beams"
                break
            
            # Keep beam_size beams with smallest residual norm
            new_beams.sort(key=lambda x: x[3])  # sort by residual norm (ascending)
            beams = new_beams[:beam_size]
            
            # Check if all beams reached termination conditions
            all_terminated = True
            for api_list, current_query_unnorm, current_query_norm, current_residual_norm in beams:
                current_norm = np.linalg.norm(current_query_unnorm)
                if current_norm >= 0.5:
                    all_terminated = False
                    break
            
            if all_terminated:
                if track_termination:
                    termination_reason = "all_beams_terminated"
                break
        
        # Return best beam (select beam with smallest residual norm)
        if beams:
            best_beam = min(beams, key=lambda x: x[3])  # beam with smallest residual norm
            if track_termination:
                return {
                    'apis': best_beam[0],
                    'termination_reason': termination_reason,
                    'iteration_count': iteration_count
                }
            else:
                return best_beam[0]  # return only API list
        else:
            if track_termination:
                return {
                    'apis': [],
                    'termination_reason': 'no_beams_available',
                    'iteration_count': iteration_count
                }
            else:
                return []
    
    def _merge_duplicate_beams(self, beams: List[Tuple]) -> List[Tuple]:
        """
        Function to merge beams with duplicate API combinations (process as set regardless of order)
        
        Args:
            beams: beam list [(api_list, current_query_unnorm, current_query_norm, residual_norm), ...]
            
        Returns:
            merged beam list (unique API combinations with duplicates removed)
        """
        if not beams:
            return beams
        
        # Create key by making API combination into set (order independent)
        beam_dict = {}
        
        for api_list, current_query_unnorm, current_query_norm, residual_norm in beams:
            # Make API combination into set to process order independently
            api_set = frozenset([f"{api['tool_name']}_{api['api_name']}" for api in api_list])
            
            if api_set not in beam_dict:
                # Add if new combination (use first discovered beam)
                beam_dict[api_set] = (api_list, current_query_unnorm, current_query_norm, residual_norm)
            # Ignore if already exists (same combination with different order)
        
        # Convert dictionary to beam list
        merged_beams = list(beam_dict.values())
        
        return merged_beams
    
    def load_test_data(self, data_dir: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load query and relevant APIs information from test_instruction JSON files
        Load G1, G2, G3 data separately
        
        Args:
            data_dir: test_instruction folder path
            
        Returns:
            Dictionary of test data separated by G1, G2, G3
        """
        print("Loading test data from instruction JSON files...")
        
        json_files = [
            'G1_instruction.json',
            'G2_instruction.json', 
            'G3_instruction.json'
        ]
        
        data_by_group = {}
        
        for file_name in json_files:
            file_path = os.path.join(data_dir, file_name)
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, skipping...")
                continue
                
            print(f"Loading {file_name}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            file_data = []
            for item in data:
                if 'query' in item and 'relevant APIs' in item:
                    test_item = {
                        'query': item['query'],
                        'relevant_apis': item['relevant APIs'],
                        'query_id': item.get('query_id', 'unknown'),
                        'source_file': file_name
                    }
                    file_data.append(test_item)
            
            # Extract group name (G1, G2, G3)
            group_name = file_name.split('_')[0]
            data_by_group[group_name] = file_data
            
            print(f"  Loaded {len(file_data)} queries from {file_name}")
        
        # Print overall statistics
        total_queries = sum(len(data) for data in data_by_group.values())
        print(f"\nTotal loaded: {total_queries} test queries")
        for group, data in data_by_group.items():
            if data:
                query_lengths = [len(x['query']) for x in data]
                print(f"  {group}: {len(data)} queries, length range: {min(query_lengths)} - {max(query_lengths)} characters")
        
        return data_by_group
    
    def evaluate_search_performance(self, query: str, relevant_apis: List[List[str]], 
                                  threshold: float = None, baseline_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Evaluate search performance for single query (compare three methods)
        
        Args:
            query: search query
            relevant_apis: relevant API list [[tool_name, api_name], ...]
            threshold: similarity threshold (for iterative/beam search)
            baseline_threshold: threshold for baseline search (default: 0.6)
            
        Returns:
            performance evaluation results
        """
        # 1. Baseline search using existing ToolBench model
        query_embedding_baseline = self.model.encode([query], convert_to_tensor=True)
        query_embedding_baseline = query_embedding_baseline.cpu().numpy()
        faiss.normalize_L2(query_embedding_baseline)
        
        # Use threshold for baseline
        
        # Efficient threshold-based search
        baseline_results = []
        search_k = 100  # initial search range
        max_search_k = min(1000, self.index.ntotal)  # maximum search range
        
        while search_k <= max_search_k:
            scores, indices = self.index.search(query_embedding_baseline, search_k)
            
            # Add only results that satisfy baseline threshold from current search results
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1: continue
                if score < baseline_threshold:
                    continue
                    
                # Check if result already added
                api_key = f"{self.documents[idx]['metadata']['tool_name']}_{self.documents[idx]['metadata']['api_name']}"
                if not any(f"{result['tool_name']}_{result['api_name']}" == api_key for result in baseline_results):
                    doc = self.documents[idx]
                    baseline_results.append({
                        'tool_name': doc['metadata']['tool_name'],
                        'api_name': doc['metadata']['api_name'],
                        'score': float(score)
                    })
            
            # Stop if last result score is below baseline threshold
            if len(scores[0]) > 0 and scores[0][-1] < baseline_threshold:
                break
                
            # Double search range
            search_k = min(search_k * 2, max_search_k)
        
        # 2. Iterative greedy search using trained model (with duplicate removal)
        iterative_results = []
        if threshold is not None:
            iterative_results = self.iterative_greedy_search(query, relevant_apis, threshold, remove_duplicates=True)
        
        # 3. Beam search iterative search
        beam_results = []
        if threshold is not None:
            beam_results = self.beam_search_iterative(query, relevant_apis, threshold, beam_size=3)
        
        # Convert relevant APIs to set
        relevant_set = set()
        for tool_name, api_name in relevant_apis:
            relevant_set.add(f"{tool_name}_{api_name}")
        
        # Convert found APIs to sets
        baseline_found = set()
        for result in baseline_results:
            baseline_found.add(f"{result['tool_name']}_{result['api_name']}")
        
        iterative_found = set()
        for result in iterative_results:
            iterative_found.add(f"{result['tool_name']}_{result['api_name']}")
        
        beam_found = set()
        for result in beam_results:
            beam_found.add(f"{result['tool_name']}_{result['api_name']}")
        
        # Calculate performance metrics
        baseline_hits = len(relevant_set & baseline_found)
        iterative_hits = len(relevant_set & iterative_found)
        beam_hits = len(relevant_set & beam_found)
        
        baseline_precision = baseline_hits / len(baseline_found) if baseline_found else 0
        iterative_precision = iterative_hits / len(iterative_found) if iterative_found else 0
        beam_precision = beam_hits / len(beam_found) if beam_found else 0
        
        baseline_recall = baseline_hits / len(relevant_set) if relevant_set else 0
        iterative_recall = iterative_hits / len(relevant_set) if relevant_set else 0
        beam_recall = beam_hits / len(relevant_set) if relevant_set else 0
        
        baseline_f1 = 2 * (baseline_precision * baseline_recall) / (baseline_precision + baseline_recall) if (baseline_precision + baseline_recall) > 0 else 0
        iterative_f1 = 2 * (iterative_precision * iterative_recall) / (iterative_precision + iterative_recall) if (iterative_precision + iterative_recall) > 0 else 0
        beam_f1 = 2 * (beam_precision * beam_recall) / (beam_precision + beam_recall) if (beam_precision + beam_recall) > 0 else 0
        
        return {
            'query': query,
            'relevant_apis': relevant_apis,
            'baseline_results': baseline_results,
            'iterative_results': iterative_results,
            'beam_results': beam_results,
            'baseline_hits': baseline_hits,
            'iterative_hits': iterative_hits,
            'beam_hits': beam_hits,
            'baseline_precision': baseline_precision,
            'iterative_precision': iterative_precision,
            'beam_precision': beam_precision,
            'baseline_recall': baseline_recall,
            'iterative_recall': iterative_recall,
            'beam_recall': beam_recall,
            'baseline_f1': baseline_f1,
            'iterative_f1': iterative_f1,
            'beam_f1': beam_f1
        }
    
    def analyze_performance(self, test_data_by_group: Dict[str, List[Dict[str, Any]]], baseline_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Performance analysis for test data separated by G1, G2, G3
        
        Args:
            test_data_by_group: Dictionary of test data separated by G1, G2, G3
            baseline_threshold: threshold for baseline search (default: 0.6)
            
        Returns:
            Overall performance analysis results (by group + comprehensive)
        """
        # Combine all data to calculate threshold
        all_test_data = []
        for group_data in test_data_by_group.values():
            all_test_data.extend(group_data)
        
        print(f"\nCalculating threshold from all {len(all_test_data)} queries...")
        threshold = self.calculate_threshold(all_test_data)
        
        # Performance analysis by group
        group_results = {}
        all_results = []
        
        for group_name, test_data in test_data_by_group.items():
            print(f"\nAnalyzing performance for {group_name} ({len(test_data)} queries)...")
            
            results = []
            baseline_precisions = []
            iterative_precisions = []
            beam_precisions = []
            baseline_recalls = []
            iterative_recalls = []
            beam_recalls = []
            baseline_f1s = []
            iterative_f1s = []
            beam_f1s = []
            
            for i, test_item in enumerate(test_data):
                if i % 50 == 0:
                    print(f"Processing {group_name} query {i+1}/{len(test_data)}...")
                
                result = self.evaluate_search_performance(
                    test_item['query'], 
                    test_item['relevant_apis'], 
                    threshold,
                    baseline_threshold
                )
                results.append(result)
                all_results.append(result)
                
                baseline_precisions.append(result['baseline_precision'])
                iterative_precisions.append(result['iterative_precision'])
                beam_precisions.append(result['beam_precision'])
                baseline_recalls.append(result['baseline_recall'])
                iterative_recalls.append(result['iterative_recall'])
                beam_recalls.append(result['beam_recall'])
                baseline_f1s.append(result['baseline_f1'])
                iterative_f1s.append(result['iterative_f1'])
                beam_f1s.append(result['beam_f1'])
            
            # Calculate group statistics
            group_stats = {
                'total_queries': len(test_data),
                'threshold': threshold,
                'baseline_precision_mean': np.mean(baseline_precisions),
                'iterative_precision_mean': np.mean(iterative_precisions),
                'beam_precision_mean': np.mean(beam_precisions),
                'baseline_recall_mean': np.mean(baseline_recalls),
                'iterative_recall_mean': np.mean(iterative_recalls),
                'beam_recall_mean': np.mean(beam_recalls),
                'baseline_f1_mean': np.mean(baseline_f1s),
                'iterative_f1_mean': np.mean(iterative_f1s),
                'beam_f1_mean': np.mean(beam_f1s),
                'baseline_precision_std': np.std(baseline_precisions),
                'iterative_precision_std': np.std(iterative_precisions),
                'beam_precision_std': np.std(beam_precisions),
                'baseline_recall_std': np.std(baseline_recalls),
                'iterative_recall_std': np.std(iterative_recalls),
                'beam_recall_std': np.std(beam_recalls),
                'baseline_f1_std': np.std(baseline_f1s),
                'iterative_f1_std': np.std(iterative_f1s),
                'beam_f1_std': np.std(beam_f1s)
            }
            
            group_results[group_name] = {
                'individual_results': results,
                'overall_stats': group_stats
            }
        
        # Calculate comprehensive statistics
        print(f"\nCalculating overall statistics for all {len(all_results)} queries...")
        all_baseline_precisions = [r['baseline_precision'] for r in all_results]
        all_iterative_precisions = [r['iterative_precision'] for r in all_results]
        all_beam_precisions = [r['beam_precision'] for r in all_results]
        all_baseline_recalls = [r['baseline_recall'] for r in all_results]
        all_iterative_recalls = [r['iterative_recall'] for r in all_results]
        all_beam_recalls = [r['beam_recall'] for r in all_results]
        all_baseline_f1s = [r['baseline_f1'] for r in all_results]
        all_iterative_f1s = [r['iterative_f1'] for r in all_results]
        all_beam_f1s = [r['beam_f1'] for r in all_results]
        
        overall_stats = {
            'total_queries': len(all_results),
            'threshold': threshold,
            'baseline_precision_mean': np.mean(all_baseline_precisions),
            'iterative_precision_mean': np.mean(all_iterative_precisions),
            'beam_precision_mean': np.mean(all_beam_precisions),
            'baseline_recall_mean': np.mean(all_baseline_recalls),
            'iterative_recall_mean': np.mean(all_iterative_recalls),
            'beam_recall_mean': np.mean(all_beam_recalls),
            'baseline_f1_mean': np.mean(all_baseline_f1s),
            'iterative_f1_mean': np.mean(all_iterative_f1s),
            'beam_f1_mean': np.mean(all_beam_f1s),
            'baseline_precision_std': np.std(all_baseline_precisions),
            'iterative_precision_std': np.std(all_iterative_precisions),
            'beam_precision_std': np.std(all_beam_precisions),
            'baseline_recall_std': np.std(all_baseline_recalls),
            'iterative_recall_std': np.std(all_iterative_recalls),
            'beam_recall_std': np.std(all_beam_recalls),
            'baseline_f1_std': np.std(all_baseline_f1s),
            'iterative_f1_std': np.std(all_iterative_f1s),
            'beam_f1_std': np.std(all_beam_f1s)
        }
        
        return {
            'group_results': group_results,
            'overall_stats': overall_stats,
            'all_individual_results': all_results
        }
    
    def print_detailed_results(self, analysis_result: Dict[str, Any], 
                             num_examples: int = 5, baseline_threshold: float = 0.6):
        """
        Print detailed results (by group + comprehensive)
        
        Args:
            analysis_result: analysis results
            num_examples: number of examples to print
            baseline_threshold: threshold for baseline search
        """
        group_results = analysis_result['group_results']
        overall_stats = analysis_result['overall_stats']
        all_results = analysis_result['all_individual_results']
        
        print("\n" + "=" * 80)
        print("VECTOR DB PERFORMANCE ANALYSIS RESULTS (Dynamic Scale Infinite Model)")
        print("=" * 80)
        
        print(f"\nTotal queries analyzed: {overall_stats['total_queries']}")
        
        # Print performance by group
        print(f"\n{'='*60}")
        print("GROUP-BY-GROUP PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        
        for group_name, group_data in group_results.items():
            stats = group_data['overall_stats']
            print(f"\n{group_name} ({stats['total_queries']} queries):")
            print(f"{'Metric':<15} {'Baseline':<15} {'Iterative':<15} {'Beam Search':<15}")
            print("-" * 90)
            print(f"{'Precision':<15} {stats['baseline_precision_mean']:.4f}±{stats['baseline_precision_std']:.4f}  {stats['iterative_precision_mean']:.4f}±{stats['iterative_precision_std']:.4f}  {stats['beam_precision_mean']:.4f}±{stats['beam_precision_std']:.4f}")
            print(f"{'Recall':<15} {stats['baseline_recall_mean']:.4f}±{stats['baseline_recall_std']:.4f}  {stats['iterative_recall_mean']:.4f}±{stats['iterative_recall_std']:.4f}  {stats['beam_recall_mean']:.4f}±{stats['beam_recall_std']:.4f}")
            print(f"{'F1-Score':<15} {stats['baseline_f1_mean']:.4f}±{stats['baseline_f1_std']:.4f}  {stats['iterative_f1_mean']:.4f}±{stats['iterative_f1_std']:.4f}  {stats['beam_f1_mean']:.4f}±{stats['beam_f1_std']:.4f}")
            
            # Best performing method for each group
            methods = ['Baseline', 'Iterative', 'Beam Search']
            f1_scores = [stats['baseline_f1_mean'], stats['iterative_f1_mean'], stats['beam_f1_mean']]
            best_method = methods[np.argmax(f1_scores)]
            print(f"Best method for {group_name}: {best_method} (F1: {max(f1_scores):.4f})")
        
        # Print comprehensive performance
        print(f"\n{'='*60}")
        print("OVERALL PERFORMANCE COMPARISON (ALL GROUPS)")
        print(f"{'='*60}")
        
        print(f"\n{'Metric':<15} {'Baseline':<15} {'Iterative':<15} {'Beam Search':<15}")
        print("-" * 90)
        print(f"{'Precision':<15} {overall_stats['baseline_precision_mean']:.4f}±{overall_stats['baseline_precision_std']:.4f}  {overall_stats['iterative_precision_mean']:.4f}±{overall_stats['iterative_precision_std']:.4f}  {overall_stats['beam_precision_mean']:.4f}±{overall_stats['beam_precision_std']:.4f}")
        print(f"{'Recall':<15} {overall_stats['baseline_recall_mean']:.4f}±{overall_stats['baseline_recall_std']:.4f}  {overall_stats['iterative_recall_mean']:.4f}±{overall_stats['iterative_recall_std']:.4f}  {overall_stats['beam_recall_mean']:.4f}±{overall_stats['beam_recall_std']:.4f}")
        print(f"{'F1-Score':<15} {overall_stats['baseline_f1_mean']:.4f}±{overall_stats['baseline_f1_std']:.4f}  {overall_stats['iterative_f1_mean']:.4f}±{overall_stats['iterative_f1_std']:.4f}  {overall_stats['beam_f1_mean']:.4f}±{overall_stats['beam_f1_std']:.4f}")
        
        print(f"\nThreshold used:")
        print(f"  - Baseline: {baseline_threshold:.4f}")
        print(f"  - Iterative/Beam Search: {overall_stats['threshold']:.4f}")
        
        # Find best performing method overall
        methods = ['Baseline', 'Iterative', 'Beam Search']
        f1_scores = [overall_stats['baseline_f1_mean'], overall_stats['iterative_f1_mean'], overall_stats['beam_f1_mean']]
        best_method = methods[np.argmax(f1_scores)]
        print(f"\nBest performing method overall: {best_method} (F1: {max(f1_scores):.4f})")
        
        # Top performing examples
        print(f"\n{'='*60}")
        print("TOP PERFORMING EXAMPLES (ACROSS ALL GROUPS)")
        print(f"{'='*60}")
        
        # Sort by F1 score
        sorted_results = sorted(all_results, key=lambda x: max(x['baseline_f1'], x['iterative_f1'], x['beam_f1']), reverse=True)
        
        for i in range(min(num_examples, len(sorted_results))):
            result = sorted_results[i]
            print(f"\nExample {i+1}:")
            print(f"Query: {result['query'][:100]}...")
            print(f"Relevant APIs: {result['relevant_apis']}")
            print(f"Baseline - Hits: {result['baseline_hits']}, Precision: {result['baseline_precision']:.3f}, Recall: {result['baseline_recall']:.3f}, F1: {result['baseline_f1']:.3f}")
            print(f"Iterative - Hits: {result['iterative_hits']}, Precision: {result['iterative_precision']:.3f}, Recall: {result['iterative_recall']:.3f}, F1: {result['iterative_f1']:.3f}")
            print(f"Beam Search - Hits: {result['beam_hits']}, Precision: {result['beam_precision']:.3f}, Recall: {result['beam_recall']:.3f}, F1: {result['beam_f1']:.3f}")

    def analyze_beam_search_performance(self, test_data_by_group: Dict[str, List[Dict[str, Any]]], 
                                      beam_sizes: List[int] = [2, 3, 5, 8, 10], 
                                      baseline_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Performance analysis of beam search with various beam sizes
        
        Args:
            test_data_by_group: Dictionary of test data separated by G1, G2, G3
            beam_sizes: List of beam sizes to test
            baseline_threshold: threshold for baseline search (default: 0.6)
            
        Returns:
            Performance analysis results by beam size
        """
        # Combine all data to calculate threshold
        all_test_data = []
        for group_data in test_data_by_group.values():
            all_test_data.extend(group_data)
        
        print(f"\nCalculating threshold from all {len(all_test_data)} queries...")
        threshold = self.calculate_threshold(all_test_data)
        
        # Performance analysis by beam size
        beam_size_results = {}
        
        for beam_size in beam_sizes:
            print(f"\nAnalyzing beam search performance with beam_size={beam_size}...")
            
            all_results = []
            precisions = []
            recalls = []
            f1s = []
            
            # Combine all group data for processing
            all_test_items = []
            for group_name, test_data in test_data_by_group.items():
                all_test_items.extend(test_data)
            
            for i, test_item in enumerate(all_test_items):
                if i % 50 == 0:
                    print(f"Processing query {i+1}/{len(all_test_items)} with beam_size={beam_size}...")
                
                # Execute only beam search
                beam_results = self.beam_search_iterative(
                    test_item['query'], 
                    test_item['relevant_apis'], 
                    threshold, 
                    beam_size=beam_size
                )
                
                # Convert relevant APIs to set
                relevant_set = set()
                for tool_name, api_name in test_item['relevant_apis']:
                    relevant_set.add(f"{tool_name}_{api_name}")
                
                # Convert found APIs to set
                beam_found = set()
                for result in beam_results:
                    beam_found.add(f"{result['tool_name']}_{result['api_name']}")
                
                # Calculate performance metrics
                beam_hits = len(relevant_set & beam_found)
                beam_precision = beam_hits / len(beam_found) if beam_found else 0
                beam_recall = beam_hits / len(relevant_set) if relevant_set else 0
                beam_f1 = 2 * (beam_precision * beam_recall) / (beam_precision + beam_recall) if (beam_precision + beam_recall) > 0 else 0
                
                result = {
                    'query': test_item['query'],
                    'relevant_apis': test_item['relevant_apis'],
                    'beam_results': beam_results,
                    'beam_hits': beam_hits,
                    'beam_precision': beam_precision,
                    'beam_recall': beam_recall,
                    'beam_f1': beam_f1,
                    'beam_size': beam_size
                }
                
                all_results.append(result)
                precisions.append(beam_precision)
                recalls.append(beam_recall)
                f1s.append(beam_f1)
            
            # Calculate statistics by beam size
            beam_size_stats = {
                'beam_size': beam_size,
                'total_queries': len(all_results),
                'threshold': threshold,
                'precision_mean': np.mean(precisions),
                'recall_mean': np.mean(recalls),
                'f1_mean': np.mean(f1s),
                'precision_std': np.std(precisions),
                'recall_std': np.std(recalls),
                'f1_std': np.std(f1s),
                'precision_median': np.median(precisions),
                'recall_median': np.median(recalls),
                'f1_median': np.median(f1s)
            }
            
            beam_size_results[beam_size] = {
                'individual_results': all_results,
                'overall_stats': beam_size_stats
            }
        
        return beam_size_results
    
    def print_beam_search_results(self, beam_size_results: Dict[str, Any], 
                                num_examples: int = 3):
        """
        Print beam search performance analysis results
        
        Args:
            beam_size_results: analysis results by beam size
            num_examples: number of examples to print
        """
        print("\n" + "=" * 80)
        print("BEAM SEARCH PERFORMANCE ANALYSIS (Different Beam Sizes)")
        print("=" * 80)
        
        # Performance comparison table by beam size
        print(f"\n{'Beam Size':<12} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Queries':<10}")
        print("-" * 80)
        
        beam_sizes = sorted(beam_size_results.keys())
        for beam_size in beam_sizes:
            stats = beam_size_results[beam_size]['overall_stats']
            print(f"{beam_size:<12} {stats['precision_mean']:.4f}±{stats['precision_std']:.4f}  {stats['recall_mean']:.4f}±{stats['recall_std']:.4f}  {stats['f1_mean']:.4f}±{stats['f1_std']:.4f}  {stats['total_queries']:<10}")
        
        # Find best performing beam size
        best_beam_size = max(beam_sizes, key=lambda x: beam_size_results[x]['overall_stats']['f1_mean'])
        best_f1 = beam_size_results[best_beam_size]['overall_stats']['f1_mean']
        print(f"\nBest performing beam size: {best_beam_size} (F1: {best_f1:.4f})")
        
        # Detailed analysis by each beam size
        print(f"\n{'='*60}")
        print("DETAILED ANALYSIS BY BEAM SIZE")
        print(f"{'='*60}")
        
        for beam_size in beam_sizes:
            stats = beam_size_results[beam_size]['overall_stats']
            print(f"\nBeam Size {beam_size}:")
            print(f"  Precision: {stats['precision_mean']:.4f} ± {stats['precision_std']:.4f} (median: {stats['precision_median']:.4f})")
            print(f"  Recall: {stats['recall_mean']:.4f} ± {stats['recall_std']:.4f} (median: {stats['recall_median']:.4f})")
            print(f"  F1-Score: {stats['f1_mean']:.4f} ± {stats['f1_std']:.4f} (median: {stats['f1_median']:.4f})")
            print(f"  Total queries: {stats['total_queries']}")
        
        # Top performing examples (based on best performing beam size)
        print(f"\n{'='*60}")
        print(f"TOP PERFORMING EXAMPLES (Beam Size {best_beam_size})")
        print(f"{'='*60}")
        
        best_results = beam_size_results[best_beam_size]['individual_results']
        sorted_results = sorted(best_results, key=lambda x: x['beam_f1'], reverse=True)
        
        for i in range(min(num_examples, len(sorted_results))):
            result = sorted_results[i]
            print(f"\nExample {i+1}:")
            print(f"Query: {result['query'][:100]}...")
            print(f"Relevant APIs: {result['relevant_apis']}")
            print(f"Beam Search Results: {len(result['beam_results'])} APIs found")
            print(f"Performance - Hits: {result['beam_hits']}, Precision: {result['beam_precision']:.3f}, Recall: {result['beam_recall']:.3f}, F1: {result['beam_f1']:.3f}")
            
            # Print found APIs
            if result['beam_results']:
                print("Found APIs:")
                for api in result['beam_results']:
                    print(f"  - {api['tool_name']}_{api['api_name']} (score: {api['score']:.3f})")



def main():
    """
    Main function - Compare three methods (Baseline, Greedy, Beam Search)
    """
    # Configuration
    full_db_dir = "vector_db_base"
    data_dir = "../ToolBench/data/test_instruction"
    trained_model_path = "trained_query_model/best_model.pt"
    BASELINE_THRESHOLD=0.55
    
    # Check directory existence
    if not os.path.exists(full_db_dir):
        print(f"Error: Full Vector DB directory '{full_db_dir}' not found!")
        return
    
    # Initialize performance analyzer (using trained Dynamic Direction-focused model)
    analyzer = VectorDBPerformanceAnalyzer(full_db_dir, trained_model_path)
    
    # Load test data (all data)
    test_data_by_group = analyzer.load_test_data(data_dir)
    
    if not test_data_by_group:
        print("No test data found!")
        return
    
    # Run three-method comparison analysis
    print("Starting Three-Method Performance Comparison...")
    analysis_result = analyzer.analyze_performance(test_data_by_group, baseline_threshold=BASELINE_THRESHOLD)
    
    # Print results
    analyzer.print_detailed_results(analysis_result, num_examples=10, baseline_threshold=BASELINE_THRESHOLD)
    
    # Save results
    output_file = "three_method_performance_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(analysis_result, f, indent=2, default=str)
    
    print(f"\nThree-method comparison results saved to: {output_file}")
    
    # Additional statistical analysis
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    overall_stats = analysis_result['overall_stats']
    
    print(f"Total queries analyzed: {overall_stats['total_queries']}")
    print(f"Threshold used:")
    print(f"  - Baseline: {BASELINE_THRESHOLD:.4f}")
    print(f"  - Iterative/Beam Search: {overall_stats['threshold']:.4f}")
    
    print(f"\nOverall Performance:")
    print(f"{'Method':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-" * 75)
    print(f"{'Baseline':<15} {overall_stats['baseline_precision_mean']:<15.4f} {overall_stats['baseline_recall_mean']:<15.4f} {overall_stats['baseline_f1_mean']:<15.4f}")
    print(f"{'Greedy':<15} {overall_stats['iterative_precision_mean']:<15.4f} {overall_stats['iterative_recall_mean']:<15.4f} {overall_stats['iterative_f1_mean']:<15.4f}")
    print(f"{'Beam Search':<15} {overall_stats['beam_precision_mean']:<15.4f} {overall_stats['beam_recall_mean']:<15.4f} {overall_stats['beam_f1_mean']:<15.4f}")
    
    # Find best performing method
    methods = ['Baseline', 'Greedy', 'Beam Search']
    f1_scores = [overall_stats['baseline_f1_mean'], overall_stats['iterative_f1_mean'], overall_stats['beam_f1_mean']]
    best_method = methods[np.argmax(f1_scores)]
    print(f"\nBest performing method: {best_method} (F1: {max(f1_scores):.4f})")
    
    # Calculate performance improvements
    baseline_f1 = overall_stats['baseline_f1_mean']
    greedy_f1 = overall_stats['iterative_f1_mean']
    beam_f1 = overall_stats['beam_f1_mean']
    
    if baseline_f1 > 0:
        greedy_improvement = ((greedy_f1 - baseline_f1) / baseline_f1) * 100
        beam_improvement = ((beam_f1 - baseline_f1) / baseline_f1) * 100
        print(f"\nImprovement over Baseline:")
        print(f"  Greedy: {greedy_improvement:+.2f}%")
        print(f"  Beam Search: {beam_improvement:+.2f}%")
    
    if greedy_f1 > 0:
        beam_vs_greedy = ((beam_f1 - greedy_f1) / greedy_f1) * 100
        print(f"  Beam Search vs Greedy: {beam_vs_greedy:+.2f}%")

if __name__ == "__main__":
    main() 