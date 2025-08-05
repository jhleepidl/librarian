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
from tqdm import tqdm

class DirectionFocusedQueryEmbeddingModel(nn.Module):
    """Dynamic direction-focused model with query-specific scale prediction"""
    
    def __init__(self, model_name: str, dropout: float = 0.1):
        super(DirectionFocusedQueryEmbeddingModel, self).__init__()
        
        # Single encoder with standard dropout
        config = AutoConfig.from_pretrained(model_name)
        config.attention_probs_dropout_prob = dropout
        config.hidden_dropout_prob = dropout
        
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        
        # Dynamic scale prediction network
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
        
        # Predict dynamic scale factor for each query
        raw_scale = self.scale_predictor(cls_token)  # [batch, 1]
        dynamic_scale = F.softplus(raw_scale) + 1.0  # [batch, 1] -> minimum value 1.0
        
        # Apply scaling to match target magnitudes
        scaled_embedding = dynamic_scale.squeeze(-1).unsqueeze(-1) * F.normalize(cls_token, p=2, dim=1)
        
        return scaled_embedding

class MultiVectorDBPerformanceAnalyzer:
    def __init__(self, vector_db_dirs: List[str], hf_model_name: str = "jhleepidl/librarian"):
        """
        Multi-Vector DB performance analyzer using Hugging Face Dynamic Direction-focused model
        
        Args:
            vector_db_dirs: List of Vector DB directories
            hf_model_name: Hugging Face model name (default: jhleepidl/librarian)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vector_db_dirs = vector_db_dirs
        
        # Load ToolBench model for API embeddings
        print("Loading ToolBench's trained retriever model...")
        self.model = SentenceTransformer('ToolBench/ToolBench_IR_bert_based_uncased').to(self.device)
        
        # Load Hugging Face Dynamic Direction-focused model for query embeddings
        print(f"Loading Hugging Face Dynamic Direction-focused model: {hf_model_name}...")
        try:
            # Base model name (ToolBench based)
            base_model_name = "ToolBench/ToolBench_IR_bert_based_uncased"
            
            # Initialize Dynamic Direction-focused model
            self.trained_model = DirectionFocusedQueryEmbeddingModel(base_model_name)
            
            # Load model weights from Hugging Face
            print(f"Loading model weights from Hugging Face: {hf_model_name}")
            
            # Load model state dictionary from Hugging Face
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            
            # Try safetensors file first
            try:
                model_path = hf_hub_download(
                    repo_id=hf_model_name,
                    filename="model.safetensors"
                )
                print(f"Loading from safetensors: {model_path}")
                # Load on CPU then move to GPU
                state_dict = load_file(model_path, device="cpu")
                # Move to GPU
                if self.device.type == 'cuda':
                    state_dict = {k: v.to(self.device) for k, v in state_dict.items()}
            except Exception as e:
                print(f"Failed to load from safetensors: {e}")
                # If safetensors not available, try pytorch_model.bin
                try:
                    model_path = hf_hub_download(
                        repo_id=hf_model_name,
                        filename="pytorch_model.bin"
                    )
                    print(f"Loading from pytorch_model.bin: {model_path}")
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    # Extract state dict (support various formats)
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint  # Direct state dict
                except Exception as e2:
                    print(f"Failed to load model weights: {e2}")
                    raise
            
            # Load weights into model
            self.trained_model.load_state_dict(state_dict)
            
            self.trained_model.to(self.device)
            self.trained_model.eval()
            
            # Load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            except:
                print(f"Warning: Could not load tokenizer for {base_model_name}, using bert-base-uncased")
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            
            print("Hugging Face Dynamic Direction-focused model loaded successfully!")
        except Exception as e:
            print(f"Error loading Hugging Face model: {e}")
            print("Trying alternative loading method...")
            try:
                # Alternative: Load config.json first to verify model structure
                from huggingface_hub import hf_hub_download
                
                # Load config.json
                config_path = hf_hub_download(
                    repo_id=hf_model_name,
                    filename="config.json"
                )
                
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                print(f"Model config: {config}")
                
                # Base model name (ToolBench based)
                base_model_name = "ToolBench/ToolBench_IR_bert_based_uncased"
                
                # Initialize Dynamic Direction-focused model
                self.trained_model = DirectionFocusedQueryEmbeddingModel(base_model_name)
                
                # Try safetensors file again (load on CPU)
                model_path = hf_hub_download(
                    repo_id=hf_model_name,
                    filename="model.safetensors"
                )
                print(f"Loading from safetensors (alternative): {model_path}")
                state_dict = load_file(model_path, device="cpu")
                
                # Move to GPU
                if self.device.type == 'cuda':
                    state_dict = {k: v.to(self.device) for k, v in state_dict.items()}
                
                # Load weights into model
                self.trained_model.load_state_dict(state_dict)
                
                self.trained_model.to(self.device)
                self.trained_model.eval()
                
                # Load tokenizer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                except:
                    print(f"Warning: Could not load tokenizer for {base_model_name}, using bert-base-uncased")
                    self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                
                print("Hugging Face model loaded successfully using alternative method!")
            except Exception as e2:
                print(f"Error with alternative loading method: {e2}")
                print("Warning: Using baseline only (no iterative/beam search)")
                self.trained_model = None
                self.tokenizer = None
        
        # Load Vector DBs
        self.vector_dbs = {}
        for db_dir in vector_db_dirs:
            if os.path.exists(db_dir):
                print(f"Loading Vector DB from {db_dir}...")
                try:
                    # Load Faiss index
                    index_path = f"{db_dir}/faiss_index.bin"
                    index = faiss.read_index(index_path)
                    
                    # Load metadata
                    metadata_path = f"{db_dir}/metadata.pkl"
                    with open(metadata_path, 'rb') as f:
                        documents = pickle.load(f)
                    
                    # Load model info
                    model_info_path = f"{db_dir}/model_info.json"
                    with open(model_info_path, 'r') as f:
                        model_info = json.load(f)
                    
                    self.vector_dbs[db_dir] = {
                        'index': index,
                        'documents': documents,
                        'model_info': model_info
                    }
                    
                    print(f"  Loaded {len(documents)} documents, {index.ntotal} vectors")
                except Exception as e:
                    print(f"  Error loading {db_dir}: {e}")
            else:
                print(f"Warning: Vector DB directory {db_dir} not found")
        
        print(f"Loaded {len(self.vector_dbs)} Vector DBs")
    
    def get_query_embedding(self, query: str, normalize: bool = True) -> np.ndarray:
        """Convert query text to embedding using Dynamic Direction-focused model"""
        if self.trained_model is not None:
            # Use trained Dynamic Direction-focused model
            with torch.no_grad():
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
                
                embedding = self.trained_model(
                    input_ids, 
                    attention_mask, 
                    token_type_ids
                )

                embedding = embedding.cpu().numpy()
                if normalize:
                    faiss.normalize_L2(embedding)
                return embedding[0]
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
        
    def get_embedding_by_index(self, idx: int, db_dir: str) -> np.ndarray:
        """Extract embedding by index from specific Vector DB"""
        try:
            index = self.vector_dbs[db_dir]['index']
            embedding = index.reconstruct(int(idx))
            return embedding
        except:
            # If failed, generate embedding from document text
            documents = self.vector_dbs[db_dir]['documents']
            doc = documents[idx]
            embedding = self.model.encode([doc['text']], convert_to_tensor=True)
            embedding = embedding.cpu().numpy()
            faiss.normalize_L2(embedding)
            return embedding[0]
    
    def calculate_threshold(self, test_data: List[Dict[str, Any]], db_dir: str) -> float:
        """Use existing threshold_cache_direction_focused_hf.json file (avoid duplicate calculation)"""
        threshold_file = "threshold_cache_direction_focused_hf.json"
        
        # Check if saved threshold exists (use existing file)
        if os.path.exists(threshold_file):
            try:
                with open(threshold_file, 'r') as f:
                    cached_data = json.load(f)
                    if 'threshold' in cached_data:
                        print(f"Using existing threshold from {threshold_file}: {cached_data['threshold']:.4f}")
                        return cached_data['threshold']
            except Exception as e:
                print(f"Error reading existing threshold file: {e}")
        
        # Calculate new threshold only if existing file not found
        print(f"Existing threshold file not found. Calculating threshold from vector_db_base...")
        min_similarity = float('inf')
        
        # Calculate only from vector_db_base
        base_db_dir = "vector_db_base"
        if base_db_dir not in self.vector_dbs:
            raise ValueError("vector_db_base not found in loaded Vector DBs")
        
        for test_item in test_data:
            query = test_item['query']
            relevant_apis = test_item['relevant_apis']
            
            # Generate query embedding
            query_embedding = self.get_query_embedding(query)
            
            # Calculate similarity with each relevant API
            for tool_name, api_name in relevant_apis:
                # Find corresponding API in vector_db_base
                documents = self.vector_dbs[base_db_dir]['documents']
                for doc in documents:
                    if (doc['metadata']['tool_name'] == tool_name and 
                        doc['metadata']['api_name'] == api_name):
                        # Generate embedding using ToolBench model with full text
                        embedding = self.model.encode([doc['text']], convert_to_tensor=True)
                        embedding = embedding.cpu().numpy()
                        faiss.normalize_L2(embedding)
                        api_embedding = embedding[0]
                        similarity = np.dot(query_embedding, api_embedding)
                        min_similarity = min(min_similarity, similarity)
                        break
        
        print(f"Calculated new threshold: {min_similarity:.4f}")
        
        # Save threshold (same format as existing file)
        with open(threshold_file, 'w') as f:
            json.dump({'threshold': float(min_similarity)}, f, indent=2)
        
        return float(min_similarity)
    
    def iterative_greedy_search(self, query: str, relevant_apis: List[List[str]], 
                               threshold: float, db_dir: str, remove_duplicates: bool = True) -> List[Dict[str, Any]]:
        """API search using iterative greedy approach"""
        # 1. Generate query embedding using trained model
        query_embedding_unnorm = self.get_query_embedding_for_iterative(query)
        query_embedding_norm = self.get_query_embedding(query, normalize=True)
        
        # 2. Iterative greedy search
        found_apis = []
        current_query_unnorm = query_embedding_unnorm.copy()
        current_query_norm = query_embedding_norm.copy()
        
        # Set for duplicate removal
        found_api_keys = set() if remove_duplicates else None
        
        index = self.vector_dbs[db_dir]['index']
        documents = self.vector_dbs[db_dir]['documents']
        
        while True:
            # Search top1 from Vector DB using current query embedding
            current_query_reshaped = current_query_norm.reshape(1, -1)
            scores, indices = index.search(current_query_reshaped, 1)
            
            # Check top1 result
            if indices[0][0] == -1:
                break
                
            score = scores[0][0]
            idx = indices[0][0]
            
            # Check termination condition
            if score < threshold:
                break
                
            # Check current query embedding norm
            current_norm = np.linalg.norm(current_query_unnorm)
            if current_norm < 0.5:
                break
            
            # Get API information
            doc = documents[idx]
            doc_embedding = self.get_embedding_by_index(idx, db_dir)
            
            # Check duplicates
            if remove_duplicates:
                api_key = f"{doc['metadata']['tool_name']}_{doc['metadata']['api_name']}"
                if api_key in found_api_keys:
                    # Found duplicate API, calculate residual and continue
                    residual = current_query_unnorm - doc_embedding
                    residual_norm = np.linalg.norm(residual)
                    
                    if residual_norm > current_norm:
                        break
                    
                    current_query_unnorm = residual
                    if residual_norm > 0:
                        current_query_norm = residual / residual_norm
                    else:
                        break
                    continue
            
            # Calculate residual
            residual = current_query_unnorm - doc_embedding
            residual_norm = np.linalg.norm(residual)
            
            if residual_norm > current_norm:
                break
            
            # Add API
            best_api = {
                'tool_name': doc['metadata']['tool_name'],
                'api_name': doc['metadata']['api_name'],
                'score': float(score)
            }
            found_apis.append(best_api)
            
            # Add to duplicate tracking set
            if remove_duplicates:
                api_key = f"{doc['metadata']['tool_name']}_{doc['metadata']['api_name']}"
                found_api_keys.add(api_key)
            
            # Update query embedding
            current_query_unnorm = residual
            
            if residual_norm > 0:
                current_query_norm = residual / residual_norm
            else:
                break
        
        return found_apis
    
    def beam_search_iterative(self, query: str, relevant_apis: List[List[str]], 
                            threshold: float, db_dir: str, beam_size: int = 5) -> List[Dict[str, Any]]:
        """API search using beam search with iterative approach"""
        # 1. Generate query embedding
        query_embedding_unnorm = self.get_query_embedding_for_iterative(query)
        query_embedding_norm = self.get_query_embedding(query, normalize=True)
        
        # 2. Initialize beam
        initial_residual_norm = np.linalg.norm(query_embedding_unnorm)
        beams = [([], query_embedding_unnorm.copy(), query_embedding_norm.copy(), initial_residual_norm)]
        
        index = self.vector_dbs[db_dir]['index']
        documents = self.vector_dbs[db_dir]['documents']
        
        while True:
            new_beams = []
            all_existing_combinations = set()
            
            # Search for each beam
            for api_list, current_query_unnorm, current_query_norm, current_residual_norm in beams:
                # Search top beam_size from Vector DB using current query embedding
                current_query_reshaped = current_query_norm.reshape(1, -1)
                scores, indices = index.search(current_query_reshaped, beam_size)
                
                # Create new beam for each search result
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:
                        continue
                        
                    if score < threshold:
                        continue
                    
                    current_norm = np.linalg.norm(current_query_unnorm)
                    if current_norm < 0.5:
                        continue
                    
                    # Get API information
                    doc = documents[idx]
                    
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
                    
                    # Check if combination already selected in other beam
                    if new_api_set in all_existing_combinations:
                        continue
                    
                    # Calculate residual
                    doc_embedding = self.get_embedding_by_index(idx, db_dir)
                    residual = current_query_unnorm - doc_embedding
                    residual_norm = np.linalg.norm(residual)
                    
                    if residual_norm > current_norm:
                        continue
                    
                    if residual_norm > 0:
                        new_query_norm = residual / residual_norm
                    else:
                        continue
                    
                    new_beams.append((new_api_list, residual, new_query_norm, residual_norm))
                    all_existing_combinations.add(new_api_set)
            
            # Exit if no new beams
            if not new_beams:
                break
            
            # Keep beam_size beams with smallest residual norm
            new_beams.sort(key=lambda x: x[3])
            beams = new_beams[:beam_size]
            
            # Check if all beams reached termination condition
            all_terminated = True
            for api_list, current_query_unnorm, current_query_norm, current_residual_norm in beams:
                current_norm = np.linalg.norm(current_query_unnorm)
                if current_norm >= 0.5:
                    all_terminated = False
                    break
            
            if all_terminated:
                break
        
        # Return best beam
        if beams:
            best_beam = min(beams, key=lambda x: x[3])
            return best_beam[0]
        else:
            return []
    
    def load_test_data(self, data_dir: str) -> List[Dict[str, Any]]:
        """Load query and relevant APIs information from test_instruction JSON files"""
        print("Loading test data from instruction JSON files...")
        
        json_files = [
            'G1_instruction.json',
            'G2_instruction.json', 
            'G3_instruction.json'
        ]
        
        all_test_data = []
        
        for file_name in json_files:
            file_path = os.path.join(data_dir, file_name)
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, skipping...")
                continue
                
            print(f"Loading {file_name}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data:
                if 'query' in item and 'relevant APIs' in item:
                    test_item = {
                        'query': item['query'],
                        'relevant_apis': item['relevant APIs'],
                        'query_id': item.get('query_id', 'unknown'),
                        'source_file': file_name
                    }
                    all_test_data.append(test_item)
            
            print(f"  Loaded {len([x for x in all_test_data if x['source_file'] == file_name])} queries from {file_name}")
        
        print(f"\nTotal loaded: {len(all_test_data)} test queries")
        return all_test_data
    
    def evaluate_search_performance(self, query: str, relevant_apis: List[List[str]], 
                                  db_dir: str, threshold: float = None, 
                                  baseline_threshold: float = 0.6) -> Dict[str, Any]:
        """Evaluate search performance for single query (compare three methods)"""
        index = self.vector_dbs[db_dir]['index']
        documents = self.vector_dbs[db_dir]['documents']
        
        # 1. Baseline search using ToolBench model
        query_embedding_baseline = self.model.encode([query], convert_to_tensor=True)
        query_embedding_baseline = query_embedding_baseline.cpu().numpy()
        faiss.normalize_L2(query_embedding_baseline)
        
        # Efficient threshold-based search
        baseline_results = []
        search_k = 100
        max_search_k = min(1000, index.ntotal)
        
        while search_k <= max_search_k:
            scores, indices = index.search(query_embedding_baseline, search_k)
            
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1: continue
                if score < baseline_threshold:
                    continue
                    
                api_key = f"{documents[idx]['metadata']['tool_name']}_{documents[idx]['metadata']['api_name']}"
                if not any(f"{result['tool_name']}_{result['api_name']}" == api_key for result in baseline_results):
                    doc = documents[idx]
                    baseline_results.append({
                        'tool_name': doc['metadata']['tool_name'],
                        'api_name': doc['metadata']['api_name'],
                        'score': float(score)
                    })
            
            if len(scores[0]) > 0 and scores[0][-1] < baseline_threshold:
                break
                
            search_k = min(search_k * 2, max_search_k)
        
        # 2. Iterative greedy search using trained model
        iterative_results = []
        if threshold is not None and self.trained_model is not None:
            iterative_results = self.iterative_greedy_search(query, relevant_apis, threshold, db_dir, remove_duplicates=True)
        
        # 3. Beam search iterative search
        beam_results = []
        if threshold is not None and self.trained_model is not None:
            beam_results = self.beam_search_iterative(query, relevant_apis, threshold, db_dir, beam_size=3)
        
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
    
    def analyze_all_vector_dbs(self, test_data: List[Dict[str, Any]], 
                             baseline_threshold: float = 0.6) -> Dict[str, Any]:
        """Performance analysis for all Vector DBs"""
        # Calculate threshold only from vector_db_base and share across all DBs
        shared_threshold = None
        if "vector_db_base" in self.vector_dbs:
            print("Calculating threshold from vector_db_base only...")
            shared_threshold = self.calculate_threshold(test_data, "vector_db_base")
            print(f"Using shared threshold for all Vector DBs: {shared_threshold:.4f}")
        else:
            print("Warning: vector_db_base not found, using default threshold")
            shared_threshold = 0.6
        
        # Apply same threshold to all Vector DBs
        thresholds = {db_dir: shared_threshold for db_dir in self.vector_db_dirs if db_dir in self.vector_dbs}
        
        # Performance analysis by Vector DB
        db_results = {}
        
        for db_dir in self.vector_db_dirs:
            if db_dir not in self.vector_dbs:
                continue
                
            print(f"\nAnalyzing performance for {db_dir}...")
            
            all_results = []
            baseline_precisions = []
            iterative_precisions = []
            beam_precisions = []
            baseline_recalls = []
            iterative_recalls = []
            beam_recalls = []
            baseline_f1s = []
            iterative_f1s = []
            beam_f1s = []
            
            for i, test_item in enumerate(tqdm(test_data, desc=f"Processing {os.path.basename(db_dir)}")):
                result = self.evaluate_search_performance(
                    test_item['query'], 
                    test_item['relevant_apis'], 
                    db_dir,
                    thresholds.get(db_dir),
                    baseline_threshold
                )
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
            
            # Calculate statistics
            overall_stats = {
                'total_queries': len(all_results),
                'threshold': thresholds.get(db_dir),
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
            
            db_results[db_dir] = {
                'overall_stats': overall_stats,
                'all_individual_results': all_results,
                'model_info': self.vector_dbs[db_dir]['model_info']
            }
        
        return db_results
    
    def print_comparison_results(self, db_results: Dict[str, Any], baseline_threshold: float = 0.6):
        """Print Vector DB comparison results"""
        print("\n" + "=" * 120)
        print("MULTI-VECTOR DB PERFORMANCE COMPARISON (Hugging Face Dynamic Direction-focused Model)")
        print("=" * 120)
        
        # Summary table by Vector DB
        print(f"\n{'Vector DB':<25} {'Size':<10} {'Mode':<20} {'Method':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
        print("-" * 120)
        
        for db_dir, results in db_results.items():
            db_name = os.path.basename(db_dir)
            overall_stats = results['overall_stats']
            model_info = results['model_info']
            
            # Vector DB information
            total_docs = model_info.get('num_documents', 'N/A')
            mode = model_info.get('mode', 'N/A')
            
            # Performance output for three methods
            methods = ['Baseline', 'Iterative', 'Beam Search']
            precisions = [overall_stats['baseline_precision_mean'], overall_stats['iterative_precision_mean'], overall_stats['beam_precision_mean']]
            recalls = [overall_stats['baseline_recall_mean'], overall_stats['iterative_recall_mean'], overall_stats['beam_recall_mean']]
            f1s = [overall_stats['baseline_f1_mean'], overall_stats['iterative_f1_mean'], overall_stats['beam_f1_mean']]
            
            for i, method in enumerate(methods):
                if i == 0:
                    print(f"{db_name:<25} {total_docs:<10} {mode:<20} {method:<15} {precisions[i]:<15.4f} {recalls[i]:<15.4f} {f1s[i]:<15.4f}")
                else:
                    print(f"{'':<25} {'':<10} {'':<20} {method:<15} {precisions[i]:<15.4f} {recalls[i]:<15.4f} {f1s[i]:<15.4f}")
            
            # Find best performing method
            best_method_idx = np.argmax(f1s)
            best_method = methods[best_method_idx]
            best_f1 = f1s[best_method_idx]
            print(f"{'':<25} {'':<10} {'':<20} {'Best':<15} {'':<15} {'':<15} {best_f1:<15.4f} ({best_method})")
            print("-" * 120)
        
        # Detailed analysis
        print(f"\n{'='*80}")
        print("DETAILED ANALYSIS BY VECTOR DB")
        print(f"{'='*80}")
        
        for db_dir, results in db_results.items():
            db_name = os.path.basename(db_dir)
            overall_stats = results['overall_stats']
            model_info = results['model_info']
            
            print(f"\n{db_name} ({model_info.get('mode', 'N/A')}):")
            print(f"  Total documents: {model_info.get('num_documents', 'N/A')}")
            print(f"  Threshold: {overall_stats['threshold']:.4f}")
            print(f"  Total queries: {overall_stats['total_queries']}")
            
            print(f"\n  Performance:")
            print(f"    {'Method':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
            print(f"    {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
            print(f"    {'Baseline':<15} {overall_stats['baseline_precision_mean']:<15.4f} {overall_stats['baseline_recall_mean']:<15.4f} {overall_stats['baseline_f1_mean']:<15.4f}")
            print(f"    {'Iterative':<15} {overall_stats['iterative_precision_mean']:<15.4f} {overall_stats['iterative_recall_mean']:<15.4f} {overall_stats['iterative_f1_mean']:<15.4f}")
            print(f"    {'Beam Search':<15} {overall_stats['beam_precision_mean']:<15.4f} {overall_stats['beam_recall_mean']:<15.4f} {overall_stats['beam_f1_mean']:<15.4f}")
            
            # Calculate performance improvement
            baseline_f1 = overall_stats['baseline_f1_mean']
            iterative_f1 = overall_stats['iterative_f1_mean']
            beam_f1 = overall_stats['beam_f1_mean']
            
            if baseline_f1 > 0:
                iterative_improvement = ((iterative_f1 - baseline_f1) / baseline_f1) * 100
                beam_improvement = ((beam_f1 - baseline_f1) / baseline_f1) * 100
                print(f"\n  Improvement over Baseline:")
                print(f"    Iterative: {iterative_improvement:+.2f}%")
                print(f"    Beam Search: {beam_improvement:+.2f}%")
        
        # Find best performing Vector DB
        best_db = max(db_results.keys(), 
                     key=lambda x: max(db_results[x]['overall_stats']['baseline_f1_mean'],
                                     db_results[x]['overall_stats']['iterative_f1_mean'],
                                     db_results[x]['overall_stats']['beam_f1_mean']))
        
        best_stats = db_results[best_db]['overall_stats']
        best_f1s = [best_stats['baseline_f1_mean'], best_stats['iterative_f1_mean'], best_stats['beam_f1_mean']]
        best_method_idx = np.argmax(best_f1s)
        best_method = ['Baseline', 'Iterative', 'Beam Search'][best_method_idx]
        best_f1 = best_f1s[best_method_idx]
        best_db_name = os.path.basename(best_db)
        
        print(f"\n{'='*80}")
        print("BEST PERFORMING VECTOR DB")
        print(f"{'='*80}")
        print(f"Best Vector DB: {best_db_name}")
        print(f"Best Method: {best_method}")
        print(f"F1-Score: {best_f1:.4f}")
        print(f"Mode: {db_results[best_db]['model_info'].get('mode', 'N/A')}")
        print(f"Size: {db_results[best_db]['model_info'].get('num_documents', 'N/A')} documents")

def main():
    """Main function - Multi-Vector DB performance comparison"""
    # Configuration (test 5 Vector DBs)
    vector_db_dirs = [
        "vector_db_base", 
        "vector_db_5k",
        "vector_db_7k5",
        "vector_db_10k",
        "vector_db_all"
    ]
    
    data_dir = "../ToolBench/data/test_instruction"
    hf_model_name = "jhleepidl/librarian"  # Hugging Face model name
    BASELINE_THRESHOLD = 0.6
    
    # Initialize performance analyzer
    analyzer = MultiVectorDBPerformanceAnalyzer(vector_db_dirs, hf_model_name)
    
    # Load test data
    test_data = analyzer.load_test_data(data_dir)
    
    if not test_data:
        print("No test data found!")
        return
    
    # Run performance analysis for all Vector DBs
    print("Starting Multi-Vector DB Performance Comparison...")
    db_results = analyzer.analyze_all_vector_dbs(test_data, baseline_threshold=BASELINE_THRESHOLD)
    
    # Print results
    analyzer.print_comparison_results(db_results, baseline_threshold=BASELINE_THRESHOLD)
    
    # Save results
    output_file = "multi_vector_db_performance_comparison_hf.json"
    with open(output_file, 'w') as f:
        json.dump(db_results, f, indent=2, default=str)
    
    print(f"\nMulti-vector DB comparison results saved to: {output_file}")

if __name__ == "__main__":
    main() 