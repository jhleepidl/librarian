#!/usr/bin/env python3
"""
Train new retriever with the new dataset
"""

import sys
import os
import json
import torch
import numpy as np
from typing import List, Dict, Any
import logging
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from tqdm import tqdm
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our retriever classes
# from librarian.novel_retriever import NovelToolRetriever, NovelToolDataset, NovelLossFunctions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data/train.json")
EVAL_PATH = os.path.join(BASE_DIR, "data/eval.json")
TEST_PATH = os.path.join(BASE_DIR, "data/test.json")


class NewDatasetTrainer:
    """
    Trainer for new dataset retriever
    """
    
    def __init__(self, dataset_dir: str = "/home/jhlee/librarian/new_dataset"):
        self.dataset_dir = dataset_dir
        self.api_registry = {}
        self.api_to_index = {}
        self.index_to_api = {}
        self.tools = []
        self.tool_descriptions = []
        
    def load_dataset(self):
        """Load the new dataset"""
        print("📦 Loading new dataset...")
        
        # Load API registry
        apis_file = os.path.join(self.dataset_dir, "apis.json")
        if os.path.exists(apis_file):
            with open(apis_file, 'r', encoding='utf-8') as f:
                self.api_registry = json.load(f)
            print(f"✅ Loaded {len(self.api_registry)} APIs")
        
        # Load API mappings
        mappings_file = os.path.join(self.dataset_dir, "api_mappings.json")
        if os.path.exists(mappings_file):
            with open(mappings_file, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
                self.api_to_index = mappings.get('api_to_index', {})
                self.index_to_api = mappings.get('index_to_api', {})
            print(f"✅ Loaded API mappings")
        
        # Create tool descriptions
        self.tools = []
        self.tool_descriptions = []
        
        for api_key, api_info in self.api_registry.items():
            tool_description = self._format_api_description(api_info)
            self.tools.append({
                'api_key': api_key,
                'data': api_info,
                'description': tool_description
            })
            self.tool_descriptions.append(tool_description)
        
        print(f"✅ Created {len(self.tools)} tool descriptions")
        
        return self.tools
    
    def _format_api_description(self, api_info: Dict[str, Any]) -> str:
        """Format API info into description string"""
        category = api_info.get('category_name', '')
        tool_name = api_info.get('tool_name', '')
        api_name = api_info.get('api_name', '')
        api_description = api_info.get('api_description', '')
        required_params = api_info.get('required_parameters', [])
        optional_params = api_info.get('optional_parameters', [])
        method = api_info.get('method', '')
        
        doc_parts = [
            f"Category: {category}",
            f"Tool Name: {tool_name}",
            f"API Name: {api_name}",
            f"API Description: {api_description}",
            f"Required Parameters: {json.dumps(required_params)}",
            f"Optional Parameters: {json.dumps(optional_params)}",
            f"Method: {method}"
        ]
        
        return ', '.join(doc_parts)
    
    def load_training_data(self, split: str = "train") -> List[Dict[str, Any]]:
        """Load training data"""
        dataset_file = os.path.join(self.dataset_dir, f"{split}.json")
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data)} samples from {split} split")
        return data
    
    def load_toolbench_instruction_data(self, instruction_dir: str, splits: List[str] = ["G1_query.json", "G2_query.json", "G3_query.json"], max_samples_per_split: int = None, save_dir: str = "librarian/data") -> List[Dict[str, Any]]:
        """
        Load ToolBench instruction data from large json files (G1_query.json, G2_query.json, G3_query.json).
        Returns a list of dicts: {"query": str, "relevant_apis": List[dict], "irrelevant_apis": List[dict], "api_list": List[dict], ...}
        Also saves train/eval/test splits to librarian/data/.
        """
        import json
        import itertools
        from sklearn.model_selection import train_test_split
        os.makedirs(save_dir, exist_ok=True)

        all_samples = []
        for split in splits:
            file_path = os.path.join(instruction_dir, split)
            print(f"Loading {file_path} ...")
            with open(file_path, 'r', encoding='utf-8') as f:
                # Assume the file is a list of dicts (json array)
                data = json.load(f)
                if max_samples_per_split:
                    data = data[:max_samples_per_split]
                for sample in data:
                    query = sample.get("query", "")
                    api_list = sample.get("api_list", [])
                    relevant_api_keys = set()
                    for rel in sample.get("relevant APIs", []):
                        if len(rel) == 2:
                            relevant_api_keys.add((rel[0], rel[1]))  # (tool_name, api_name)
                    relevant_apis = [api for api in api_list if (api.get("tool_name"), api.get("api_name")) in relevant_api_keys]
                    irrelevant_apis = [api for api in api_list if (api.get("tool_name"), api.get("api_name")) not in relevant_api_keys]
                    all_samples.append({
                        "query": query,
                        "relevant_apis": relevant_apis,
                        "irrelevant_apis": irrelevant_apis,
                        "api_list": api_list,
                        "query_id": sample.get("query_id", None)
                    })
        print(f"Loaded {len(all_samples)} samples from ToolBench instruction data.")
        # Split into train/eval/test (8:1:1)
        train, temp = train_test_split(all_samples, test_size=0.2, random_state=42)
        eval, test = train_test_split(temp, test_size=0.5, random_state=42)
        # Save
        with open(os.path.join(save_dir, "train.json"), "w", encoding="utf-8") as f:
            json.dump(train, f, ensure_ascii=False, indent=2)
        with open(os.path.join(save_dir, "eval.json"), "w", encoding="utf-8") as f:
            json.dump(eval, f, ensure_ascii=False, indent=2)
        with open(os.path.join(save_dir, "test.json"), "w", encoding="utf-8") as f:
            json.dump(test, f, ensure_ascii=False, indent=2)
        print(f"Saved splits to {save_dir}/train.json, eval.json, test.json")
        return all_samples
    
    def train_retriever(self, 
                       model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                       loss_type: str = "cosine_similarity",
                       batch_size: int = 4,
                       epochs: int = 10,
                       learning_rate: float = 2e-5,
                       scale_factor: float = 1.0,
                       save_path: str = "/home/jhlee/librarian/trained_model"):
        """
        Train the retriever with new dataset
        """
        print(f"🚀 Starting retriever training...")
        print(f"📊 Model: {model_name}")
        print(f"📊 Loss: {loss_type}")
        print(f"📊 Batch size: {batch_size}")
        print(f"📊 Epochs: {epochs}")
        print(f"📊 Learning rate: {learning_rate}")
        
        try:
            # Load dataset
            tools = self.load_dataset()
            train_data = self.load_training_data("train")
            
            # Initialize retriever
            # retriever = NovelToolRetriever(
            #     model_name=model_name,
            #     max_tools=10
            # )
            
            print("✅ Retriever initialized")
            
            # Load model
            # retriever.load_model()
            print("✅ Model loaded")
            
            # Set tools
            # retriever.tools = self.tools
            # retriever.tool_descriptions = self.tool_descriptions
            print(f"✅ Set {len(tools)} tools")
            
            # Compute embeddings
            # retriever.compute_embeddings()
            print("✅ Tool embeddings computed")
            
            # Train the model
            # retriever.train(
            #     train_data=train_data,
            #     loss_type=loss_type,
            #     batch_size=batch_size,
            #     epochs=epochs,
            #     learning_rate=learning_rate,
            #     scale_factor=scale_factor
            # )
            
            # Save the model
            os.makedirs(save_path, exist_ok=True)
            # retriever.save_model(save_path)
            print(f"✅ Model saved to {save_path}")
            
            # Test recommendation
            test_query = "I need to check the health of the SQUAKE API and get project information"
            # recommendations = retriever.recommend_tools(test_query, top_k=5)
            
            print(f"\n�� Test query: '{test_query}'")
            print("🔍 Top recommendations:")
            # for i, rec in enumerate(recommendations):
            #     tool_name = rec['tool']['data'].get('tool_name', 'Unknown')
            #     api_name = rec['tool']['data'].get('api_name', 'Unknown')
            #     print(f"  {i+1}. {tool_name} - {api_name} (score: {rec['score']:.4f})")
            
            return tools # Return tools as a placeholder
            
        except Exception as e:
            print(f"❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_trained_model(self, model_path: str = "/home/jhlee/librarian/trained_model"):
        """Test the trained model"""
        print(f"\n🧪 Testing trained model...")
        
        try:
            # Load the trained model
            # retriever = NovelToolRetriever()
            # retriever.load_model(model_path)
            
            # Load tools
            tools = self.load_dataset()
            # retriever.tools = self.tools
            # retriever.tool_descriptions = self.tool_descriptions
            
            # Compute embeddings
            # retriever.compute_embeddings()
            
            # Test queries
            test_queries = [
                "I need to check the health of the SQUAKE API and get project information",
                "I want to track a package through Correo Argentino",
                "I'm looking for cocktail recipes and financial data"
            ]
            
            for i, query in enumerate(test_queries):
                print(f"\n📝 Test query {i+1}: '{query}'")
                # recommendations = retriever.recommend_tools(query, top_k=3)
                
                print("🔍 Top recommendations:")
                # for j, rec in enumerate(recommendations):
                #     tool_name = rec['tool']['data'].get('tool_name', 'Unknown')
                #     api_name = rec['tool']['data'].get('api_name', 'Unknown')
                #     print(f"  {j+1}. {tool_name} - {api_name} (score: {rec['score']:.4f})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing model: {e}")
            import traceback
            traceback.print_exc()
            return False


class ToolbenchRetrieverDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            'query': item['query'],
            'relevant_apis': item['relevant_apis'],
            'irrelevant_apis': item['irrelevant_apis']
        }

def l2_normalize(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def generate_negative_samples_dynamic(relevant_apis, all_apis_in_batch, num_negatives=1):
    """
    Generate a single negative sample per query with mixed strategy:
    - 70%: Different category APIs (completely different domain)
    - 20%: Same category, different tool APIs (similar but different tool)
    - 10%: Same tool, different API APIs (hard negatives)
    
    Args:
        relevant_apis: List of relevant APIs for current sample
        all_apis_in_batch: List of all APIs from all samples in the batch
        num_negatives: Number of negative samples to generate (should be 1)
    
    Returns:
        List of selected negative APIs (single API)
    """
    import random
    
    # Get relevant categories and tools for current sample
    relevant_categories = set(api.get('category_name', '') for api in relevant_apis)
    relevant_tools = set(api.get('tool_name', '') for api in relevant_apis)
    relevant_api_names = set(api.get('api_name', '') for api in relevant_apis)
    
    # Get all irrelevant APIs (not in relevant_apis)
    irrelevant_apis = [api for api in all_apis_in_batch if (api.get('tool_name'), api.get('api_name')) not in 
                      [(r.get('tool_name'), r.get('api_name')) for r in relevant_apis]]
    
    # Categorize irrelevant APIs
    different_category_apis = []
    same_category_diff_tool_apis = []
    same_tool_diff_api_apis = []
    
    for api in irrelevant_apis:
        category = api.get('category_name', '')
        tool_name = api.get('tool_name', '')
        api_name = api.get('api_name', '')
        
        if category not in relevant_categories:
            # Different category (70%)
            different_category_apis.append(api)
        elif tool_name not in relevant_tools:
            # Same category, different tool (20%)
            same_category_diff_tool_apis.append(api)
        elif api_name not in relevant_api_names:
            # Same tool, different API (10%)
            same_tool_diff_api_apis.append(api)
    
    # Select one negative sample based on probability distribution
    selected_negative = None
    
    # Generate random number to determine category
    rand_val = random.random()
    
    if rand_val < 0.7 and different_category_apis:
        # 70% chance: Different category
        selected_negative = random.choice(different_category_apis)
    elif rand_val < 0.9 and same_category_diff_tool_apis:
        # 20% chance: Same category, different tool
        selected_negative = random.choice(same_category_diff_tool_apis)
    elif same_tool_diff_api_apis:
        # 10% chance: Same tool, different API
        selected_negative = random.choice(same_tool_diff_api_apis)
    else:
        # Fallback: pick from any available irrelevant APIs
        if irrelevant_apis:
            selected_negative = random.choice(irrelevant_apis)
    
    return [selected_negative] if selected_negative else []

def improved_retriever_loss_v9(query_emb, pos_api_embs, neg_api_emb, alpha=0.5, beta=0.5):
    """
    Normalized positive distance loss function
    
    Args:
        query_emb: Query embedding (batch, dim)
        pos_api_embs: Positive API embeddings (batch, n_pos, dim)
        neg_api_emb: Single negative API embedding (batch, dim)
        alpha: Weight for positive distance (default: 0.5)
        beta: Weight for negative distance (default: 0.5)
    """
    # Normalize all embeddings
    query_emb_norm = F.normalize(query_emb, p=2, dim=-1)
    pos_api_embs_norm = F.normalize(pos_api_embs, p=2, dim=-1)
    neg_api_emb_norm = F.normalize(neg_api_emb, p=2, dim=-1)
    
    # Calculate positive API embeddings sum
    pos_sum = pos_api_embs_norm.sum(dim=1)  # (batch, dim)
    pos_sum_norm = F.normalize(pos_sum, p=2, dim=-1)
    
    # Calculate distances
    pos_dist = 1 - F.cosine_similarity(query_emb_norm, pos_sum_norm, dim=-1)  # (batch,)
    neg_dist = 1 - F.cosine_similarity(query_emb_norm, neg_api_emb_norm, dim=-1)  # (batch,)
    
    # Calculate the magnitude of positive API embeddings sum
    pos_sum_magnitude = torch.norm(pos_sum, p=2, dim=-1)  # (batch,)
    
    # Normalize positive distance by positive magnitude
    normalized_pos_dist = pos_dist / (pos_sum_magnitude + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Balanced loss with normalized positive distance
    loss = alpha * normalized_pos_dist + beta * neg_dist
    
    return loss.mean()


def custom_retriever_loss_single_negative(query_emb, pos_api_embs, neg_api_emb, alpha=0.7, beta=0.3):
    """
    Legacy loss function (kept for compatibility)
    """
    return improved_retriever_loss_v9(query_emb, pos_api_embs, neg_api_emb, alpha=0.5, beta=0.5)

def collate_fn(batch):
    # batch: list of dicts
    queries = [item['query'] for item in batch]
    
    # Process positive APIs
    pos_api_texts = [[f"{api['tool_name']} {api['api_name']} {api.get('api_description','')}" for api in item['relevant_apis']] for item in batch]
    
    # Collect all APIs from the batch for dynamic negative sampling
    all_apis_in_batch = []
    for item in batch:
        # Use relevant_apis + irrelevant_apis as all available APIs
        all_apis_in_batch.extend(item['relevant_apis'])
        all_apis_in_batch.extend(item['irrelevant_apis'])
    
    # Process negative APIs with single negative sample per query
    neg_api_texts = []
    for item in batch:
        # Generate single negative sample dynamically from all APIs in the batch
        selected_negatives = generate_negative_samples_dynamic(
            item['relevant_apis'], 
            all_apis_in_batch, 
            num_negatives=1
        )
        
        # Convert to text format (single negative)
        if selected_negatives:
            neg_texts = [f"{api['tool_name']} {api['api_name']} {api.get('api_description','')}" for api in selected_negatives]
        else:
            # Fallback: use first irrelevant API if no negative found
            neg_texts = [f"{item['irrelevant_apis'][0]['tool_name']} {item['irrelevant_apis'][0]['api_name']} {item['irrelevant_apis'][0].get('api_description','')}"] if item['irrelevant_apis'] else [""]
        neg_api_texts.append(neg_texts)
    
    return queries, pos_api_texts, neg_api_texts

def get_embeddings(model, texts, device):
    """
    SentenceTransformer 학습을 위한 임베딩 생성 함수
    """
    # SentenceTransformer 내부의 첫 번째 모듈(Transformer)에서 토크나이즈
    tokenizer = model.tokenizer
    max_length = model.get_max_seq_length()
    features = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    features = {k: v.to(device) for k, v in features.items()}
    # SentenceTransformer의 forward는 features dict를 받음
    with torch.set_grad_enabled(model.training):
        out = model({"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]})
        # out["sentence_embedding"] 또는 out["token_embeddings"] 등 모델 구조에 따라 다름
        return out["sentence_embedding"]


def process_api_embeddings_single_negative(model, api_texts, device):
    """
    Process API embeddings for single negative samples
    
    Args:
        model: SentenceTransformer model
        api_texts: List of API text lists (each list contains single API text)
        device: Device (cuda/cpu)
    
    Returns:
        torch.Tensor: Processed embedding tensor (batch, dim)
    """
    embeddings = []
    for apis in api_texts:
        if len(apis) == 0 or apis[0] == "":
            # Create zero embedding if no API
            emb = torch.zeros(768, device=device)
        else:
            # Single API text
            emb = get_embeddings(model, apis, device)
            # Ensure it's 1D (single embedding)
            if len(emb.shape) > 1:
                emb = emb.squeeze(0)  # Remove batch dimension if present
            # Ensure it has the right dimension
            if emb.shape[0] != 768:
                emb = emb.expand(768)
        
        embeddings.append(emb)
    
    return torch.stack(embeddings)  # (batch, dim)

def process_api_embeddings(model, api_texts, device, max_len=None):
    """
    공통 API 임베딩 처리 함수 (for positive APIs)
    
    Args:
        model: SentenceTransformer 모델
        api_texts: API 텍스트 리스트의 리스트
        device: 디바이스 (cuda/cpu)
        max_len: 최대 길이 (None이면 자동 계산)
    
    Returns:
        torch.Tensor: 처리된 임베딩 텐서
    """
    if max_len is None:
        max_len = max(len(apis) for apis in api_texts)
    
    embeddings = []
    for apis in api_texts:
        if len(apis) == 0:
            emb = torch.zeros(1, 768, device=device)
        else:
            emb = get_embeddings(model, apis, device)
        # Ensure all tensors have the same embedding dimension (768)
        if len(emb.shape) == 1:
            # If 1D, expand to 2D first
            emb = emb.unsqueeze(1)  # [seq_len] -> [seq_len, 1]
            # Then expand to 768 dimensions
            emb = emb.expand(-1, 768)  # [seq_len, 1] -> [seq_len, 768]
        elif emb.shape[1] != 768:
            # If not 768 dimensions, expand to 768
            emb = emb.expand(-1, 768)
        
        if len(emb) < max_len:
            # Pad with zeros at the end (batch dimension)
            padding_size = max_len - len(emb)
            zero_pad = torch.zeros(padding_size, 768, device=device, dtype=emb.dtype)
            emb = torch.cat([emb, zero_pad], dim=0)
        embeddings.append(emb)
    
    return torch.stack(embeddings)


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for queries, pos_api_texts, neg_api_texts in val_loader:
            with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.enable_grad():
                query_emb = get_embeddings(model, queries, device)
                pos_api_embs = process_api_embeddings(model, pos_api_texts, device)
                neg_api_embs = process_api_embeddings_single_negative(model, neg_api_texts, device)
            loss = custom_retriever_loss_single_negative(query_emb, pos_api_embs, neg_api_embs)
            total_loss += loss.item() * len(queries)
            count += len(queries)
    model.train()
    return total_loss / count if count > 0 else 0


def analyze_dataset_and_calculate_batch_size(train_dataset, val_dataset, device, target_vram_gb=20):
    """
    데이터셋을 분석하여 VRAM에 맞는 최적 배치 크기를 계산
    
    Args:
        train_dataset: 훈련 데이터셋
        val_dataset: 검증 데이터셋
        device: 디바이스
        target_vram_gb: 목표 VRAM 사용량 (GB)
    
    Returns:
        int: 최적 배치 크기
    """
    print("🔍 Analyzing dataset for optimal batch size...")
    
    # 샘플 데이터로 메모리 사용량 테스트
    sample_batch_size = 4
    test_loader = DataLoader(train_dataset, batch_size=sample_batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 첫 번째 배치로 메모리 사용량 측정
    model = SentenceTransformer('ToolBench/ToolBench_IR_bert_based_uncased')
    model = model.to(device)
    model.eval()
    
    try:
        with torch.no_grad():
            for queries, pos_api_texts, neg_api_texts in test_loader:
                # 메모리 사용량 측정 시작
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                query_emb = get_embeddings(model, queries, device)
                pos_api_embs = process_api_embeddings(model, pos_api_texts, device)
                neg_api_embs = process_api_embeddings_single_negative(model, neg_api_texts, device)
                
                # 메모리 사용량 확인
                max_memory_used = torch.cuda.max_memory_allocated() / (1024**3)  # GB
                current_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                
                print(f"📊 Memory usage for batch_size={sample_batch_size}:")
                print(f"   - Peak memory: {max_memory_used:.2f} GB")
                print(f"   - Current memory: {current_memory:.2f} GB")
                
                # 안전 마진을 위해 80%만 사용
                available_memory = target_vram_gb * 0.8
                optimal_batch_size = int(sample_batch_size * (available_memory / max_memory_used))
                
                # 최소/최대 배치 크기 제한
                optimal_batch_size = max(1, min(optimal_batch_size, 64))
                
                print(f"🎯 Calculated optimal batch size: {optimal_batch_size}")
                print(f"   - Available VRAM: {available_memory:.2f} GB")
                print(f"   - Estimated memory usage: {max_memory_used * (optimal_batch_size / sample_batch_size):.2f} GB")
                
                # 메모리 정리
                del query_emb, pos_api_embs, neg_api_embs
                torch.cuda.empty_cache()
                break
                
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("⚠️  Sample batch size too large, reducing...")
            optimal_batch_size = 2
        else:
            raise e
    
    return optimal_batch_size


def train(resume_from=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        # 메모리 최적화 설정
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    train_dataset = ToolbenchRetrieverDataset(TRAIN_PATH)
    val_dataset = ToolbenchRetrieverDataset(EVAL_PATH)
    
    # 최적 배치 크기 계산
    optimal_batch_size = analyze_dataset_and_calculate_batch_size(train_dataset, val_dataset, device, target_vram_gb=20)
    
    train_loader = DataLoader(train_dataset, batch_size=optimal_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=optimal_batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = SentenceTransformer('ToolBench/ToolBench_IR_bert_based_uncased')
    model = model.to(device)
    print(f"Model device: {next(model.parameters()).device}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # 체크포인트에서 재개
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        print(f"Resumed from epoch {start_epoch-1}, best_val_loss: {best_val_loss:.6f}")
    
    model.train()
    patience = 3
    min_delta = 1e-4
    num_epochs = 20
    
    # Gradient accumulation 설정 (원래 배치 크기 128을 유지하기 위해)
    target_effective_batch_size = 128
    accumulation_steps = max(1, target_effective_batch_size // optimal_batch_size)
    print(f"📈 Training with batch_size={optimal_batch_size}, accumulation_steps={accumulation_steps}")
    print(f"   - Effective batch size: {optimal_batch_size * accumulation_steps}")
    
    # 체크포인트 저장 디렉토리 생성
    checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()
        
        for batch_idx, (queries, pos_api_texts, neg_api_texts) in enumerate(pbar):
            loss = None  # loss 변수를 미리 초기화
            try:
                with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.enable_grad():
                    query_emb = get_embeddings(model, queries, device)
                    pos_api_embs = process_api_embeddings(model, pos_api_texts, device)
                    neg_api_embs = process_api_embeddings_single_negative(model, neg_api_texts, device)
                
                loss = custom_retriever_loss_single_negative(query_emb, pos_api_embs, neg_api_embs)
                # Gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # 메모리 정리
                del query_emb, pos_api_embs, neg_api_embs
                torch.cuda.empty_cache()
                
                elapsed = time.time() - start_time
                batches_done = batch_idx + 1
                batches_total = len(train_loader)
                eta = (elapsed / batches_done) * (batches_total - batches_done) if batches_done > 0 else 0
                pbar.set_postfix({'loss': loss.item() * accumulation_steps, 'elapsed': f"{elapsed/60:.1f}m", 'eta': f"{eta/60:.1f}m"})
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU OOM at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    # loss가 None이면 기본값 사용
                    if loss is None:
                        loss = torch.tensor(0.0, device=device)
                    pbar.set_postfix({'loss': 'OOM', 'elapsed': f"{(time.time() - start_time)/60:.1f}m", 'eta': 'N/A'})
                    continue
                else:
                    raise e
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {epoch_time/60:.2f} minutes")
        
        # 검증
        val_loss = evaluate(model, val_loader, device)
        print(f"Validation loss after epoch {epoch+1}: {val_loss:.6f}")
        
        # 체크포인트 저장 (매 에포크마다)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'val_loss': val_loss
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # 최고 성능 모델 저장
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(BASE_DIR, 'trained_toolbench_retriever_best')
            model.save(best_model_path)
            print(f"Best model saved at epoch {epoch+1}: {best_model_path}")
            
            # 최고 성능 체크포인트도 저장
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'val_loss': val_loss
            }, best_checkpoint_path)
            print(f"Best checkpoint saved: {best_checkpoint_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
    
    # 최종 모델 저장
    final_model_path = os.path.join(BASE_DIR, 'trained_toolbench_retriever')
    model.save(final_model_path)
    print(f"Final model saved: {final_model_path}")


def main():
    """Main training function"""
    
    print("🚀 New Retriever Training")
    print("=" * 50)
    
    # 체크포인트 재개 옵션 (명령행 인자로 받을 수 있음)
    import sys
    resume_from = None
    if len(sys.argv) > 1:
        resume_from = sys.argv[1]
        print(f"Will resume from: {resume_from}")
    
    # Initialize trainer
    trainer = NewDatasetTrainer()
    
    # Train the model using the new custom collate_fn
    train(resume_from=resume_from)


if __name__ == "__main__":
    main() 