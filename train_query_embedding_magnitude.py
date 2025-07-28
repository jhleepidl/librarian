#!/usr/bin/env python3
"""
Train only query embedding model while keeping API embedding model fixed
Improved version with simplified loss function (no negative sampling) and better utilities
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
from torch.optim.lr_scheduler import OneCycleLR

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data/train.json")
EVAL_PATH = os.path.join(BASE_DIR, "data/eval.json")
TEST_PATH = os.path.join(BASE_DIR, "data/test.json")


class QueryEmbeddingModel(torch.nn.Module):
    """
    Query embedding model that learns to map queries to API embedding space
    """
    
    def __init__(self, base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        # Use a pre-trained model as the base
        self.base_model = SentenceTransformer(base_model_name)
        
        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Add a projection layer to map to the same dimension as API embeddings
        self.projection = torch.nn.Linear(384, 768)  # all-MiniLM-L6-v2 -> ToolBench dimension
        
        # Add a small MLP for fine-tuning
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, 768)
        )
        
        # Add magnitude prediction layer
        self.magnitude_predictor = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1),
            torch.nn.Softplus()  # Ensure positive magnitude
        )
    
    def forward(self, input_ids, attention_mask):
        # Get base embeddings
        with torch.no_grad():
            base_output = self.base_model({"input_ids": input_ids, "attention_mask": attention_mask})
            base_embeddings = base_output["sentence_embedding"]
        
        # Project to target dimension
        projected = self.projection(base_embeddings)
        
        # Apply MLP to get raw embedding
        raw_embedding = self.mlp(projected)  # Raw embedding vector
        
        # Predict magnitude
        magnitude = self.magnitude_predictor(projected)  # (batch_size, 1)
        
        # Combine raw embedding and magnitude
        # First normalize the raw embedding, then scale by magnitude
        normalized_embedding = F.normalize(raw_embedding, p=2, dim=-1)
        output = normalized_embedding * magnitude
        
        return output, magnitude


class QueryEmbeddingDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            'query': item['query'],
            'relevant_apis': item['relevant_apis']
        }


class QueryEmbeddingTrainer:
    """
    Trainer for query embedding model
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.query_model = None
        self.api_model = None
        
    def load_models(self):
        """Load query and API models"""
        print("📦 Loading models...")
        
        # Load query embedding model
        self.query_model = QueryEmbeddingModel()
        self.query_model = self.query_model.to(self.device)
        print("✅ Query embedding model loaded")
        
        # Load API embedding model (ToolBench, frozen)
        self.api_model = SentenceTransformer("ToolBench/ToolBench_IR_bert_based_uncased")
        self.api_model = self.api_model.to(self.device)
        self.api_model.eval()
        
        # Freeze API model parameters
        for param in self.api_model.parameters():
            param.requires_grad = False
        print("✅ API embedding model loaded (frozen)")
        
        return True
    
    def get_query_embedding(self, queries: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Get query embeddings and magnitudes using the trainable model"""
        tokenizer = self.query_model.base_model.tokenizer
        max_length = self.query_model.base_model.get_max_seq_length()
        
        features = tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        features = {k: v.to(self.device) for k, v in features.items()}
        
        return self.query_model(features["input_ids"], features["attention_mask"])
    
    def get_api_embedding(self, api_texts: List[str]) -> torch.Tensor:
        """Get normalized API embeddings using the frozen ToolBench model"""
        with torch.no_grad():
            embeddings = self.api_model.encode(api_texts, convert_to_tensor=True, device=self.device)
            # Normalize embeddings
            normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
        return normalized_embeddings
    
    def format_api_text(self, api: Dict[str, Any]) -> str:
        """Format API info into text string"""
        tool_name = api.get('tool_name', '')
        api_name = api.get('api_name', '')
        api_description = api.get('api_description', '')
        category = api.get('category_name', '')
        method = api.get('method', '')
        
        return f"Tool: {tool_name}, API: {api_name}, Category: {category}, Method: {method}, Description: {api_description}"
    
    def adaptive_magnitude_loss_function(self, query_emb, query_magnitude, pos_api_embs):
        """
        Adaptive magnitude loss that considers query complexity and API diversity
        
        Args:
            query_emb: Query embedding (batch, dim)
            query_magnitude: Predicted magnitude (batch, 1)
            pos_api_embs: Positive API embeddings (batch, n_pos, dim)
        """
        # Calculate API diversity (how spread out the relevant APIs are)
        pos_centroid = pos_api_embs.mean(dim=1)  # (batch, dim)
        api_diversity = torch.std(torch.norm(pos_api_embs, p=2, dim=-1), dim=1, keepdim=True)  # (batch, 1)
        
        # Target magnitude based on API diversity
        # More diverse APIs → higher magnitude (more specific query)
        # Less diverse APIs → lower magnitude (more general query)
        base_magnitude = torch.norm(pos_centroid, p=2, dim=-1, keepdim=True)
        diversity_factor = 1.0 + 0.5 * api_diversity  # Scale factor based on diversity
        target_magnitude = base_magnitude * diversity_factor
        
        # Direction loss
        query_direction = F.normalize(query_emb, p=2, dim=-1)
        pos_direction = F.normalize(pos_centroid, p=2, dim=-1)
        direction_loss = 1 - F.cosine_similarity(query_direction, pos_direction, dim=-1)
        
        # Adaptive magnitude loss
        magnitude_loss = F.mse_loss(query_magnitude, target_magnitude, reduction='none').squeeze(-1)
        
        # Combined loss
        total_loss = 0.6 * direction_loss + 0.4 * magnitude_loss
        
        return total_loss.mean()
    
    def confidence_based_magnitude_loss_function(self, query_emb, query_magnitude, pos_api_embs):
        """
        Confidence-based magnitude loss where magnitude reflects prediction confidence
        
        Args:
            query_emb: Query embedding (batch, dim)
            query_magnitude: Predicted magnitude (batch, 1)
            pos_api_embs: Positive API embeddings (batch, n_pos, dim)
        """
        # Calculate confidence based on how well query aligns with API centroid
        pos_centroid = pos_api_embs.mean(dim=1)
        query_direction = F.normalize(query_emb, p=2, dim=-1)
        pos_direction = F.normalize(pos_centroid, p=2, dim=-1)
        
        # Confidence = cosine similarity (higher = more confident)
        confidence = F.cosine_similarity(query_direction, pos_direction, dim=-1, eps=1e-8)
        
        # Target magnitude should be proportional to confidence
        # High confidence → higher magnitude (stronger prediction)
        # Low confidence → lower magnitude (weaker prediction)
        base_magnitude = torch.norm(pos_centroid, p=2, dim=-1)
        target_magnitude = base_magnitude * confidence.unsqueeze(-1)
        
        # Direction loss
        direction_loss = 1 - confidence
        
        # Magnitude loss
        magnitude_loss = F.mse_loss(query_magnitude, target_magnitude, reduction='none').squeeze(-1)
        
        # Combined loss
        total_loss = 0.7 * direction_loss + 0.3 * magnitude_loss
        
        return total_loss.mean()
    
    def magnitude_aware_loss_function(self, query_emb, query_magnitude, pos_api_embs):
        """
        Loss function that considers both direction and magnitude of query embeddings
        
        Args:
            query_emb: Query embedding (batch, dim) - with learned magnitude
            query_magnitude: Predicted magnitude (batch, 1)
            pos_api_embs: Positive API embeddings (batch, n_pos, dim) - normalized
        """
        # Calculate target magnitude from positive API embeddings
        pos_centroid = pos_api_embs.mean(dim=1)  # (batch, dim)
        target_magnitude = torch.norm(pos_centroid, p=2, dim=-1, keepdim=True)  # (batch, 1)
        
        # Direction loss (cosine similarity)
        query_direction = F.normalize(query_emb, p=2, dim=-1)
        pos_direction = F.normalize(pos_centroid, p=2, dim=-1)
        direction_loss = 1 - F.cosine_similarity(query_direction, pos_direction, dim=-1)
        
        # Magnitude loss (L2 distance between predicted and target magnitude)
        magnitude_loss = F.mse_loss(query_magnitude, target_magnitude, reduction='none').squeeze(-1)
        
        # Combined loss with weights
        total_loss = 0.7 * direction_loss + 0.3 * magnitude_loss
        
        return total_loss.mean()
    
    def improved_loss_function(self, query_emb, pos_api_embs):
        """
        Improved loss function with normalization and cosine similarity
        Query embedding should be close to the centroid of relevant API embeddings
        
        Args:
            query_emb: Query embedding (batch, dim) - will be normalized
            pos_api_embs: Positive API embeddings (batch, n_pos, dim) - already normalized
        """
        # Normalize query embeddings
        query_emb_norm = F.normalize(query_emb, p=2, dim=-1)
        
        # Calculate centroid of positive API embeddings (mean instead of sum)
        pos_centroid = pos_api_embs.mean(dim=1)  # (batch, dim) - mean of normalized embeddings
        
        # Option 1: Cosine similarity loss (maximize similarity)
        cosine_sim = F.cosine_similarity(query_emb_norm, pos_centroid, dim=-1)
        loss_cosine = 1 - cosine_sim  # Convert to loss (0 = perfect similarity)
        
        # Option 2: L2 distance between normalized vectors
        loss_l2 = F.mse_loss(query_emb_norm, pos_centroid, reduction='none').mean(dim=-1)
        
        # Combine both losses for better training
        total_loss = 0.5 * loss_cosine + 0.5 * loss_l2
        
        return total_loss.mean()
    
    def simplified_loss_function(self, query_emb, pos_api_embs):
        """
        Simplified loss function focusing only on positive samples
        Query embedding should be close to the sum of normalized relevant API embeddings using L2 distance
        
        Args:
            query_emb: Query embedding (batch, dim) - NOT normalized
            pos_api_embs: Positive API embeddings (batch, n_pos, dim) - NORMALIZED
        """
        # Calculate sum of normalized positive API embeddings
        pos_sum = pos_api_embs.sum(dim=1)  # (batch, dim) - sum of normalized embeddings
        
        # Calculate L2 distance between query embedding and positive sum
        loss = F.mse_loss(query_emb, pos_sum, reduction='none')  # (batch, dim)
        loss = loss.mean(dim=-1)  # (batch,) - average over dimensions
        
        return loss.mean()  # scalar
    
    def collate_fn(self, batch):
        """Custom collate function - only positive APIs"""
        queries = [item['query'] for item in batch]
        
        # Process positive APIs only
        pos_api_texts = []
        for item in batch:
            api_texts = [self.format_api_text(api) for api in item['relevant_apis']]
            pos_api_texts.append(api_texts)
        
        return queries, pos_api_texts
    
    def analyze_dataset_and_calculate_batch_size(self, train_dataset, val_dataset, target_vram_gb=20):
        """
        Analyze dataset to calculate optimal batch size based on VRAM
        """
        print("🔍 Analyzing dataset for optimal batch size...")
        
        # Get total VRAM
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"📊 Total VRAM: {total_vram:.2f} GB")
        
        # Use 85% of VRAM for training (더 적극적으로)
        available_memory = total_vram * 0.85
        print(f"📊 Available VRAM for training: {available_memory:.2f} GB (85%)")
        
        # 더 큰 sample batch size로 시작 (더 정확한 측정을 위해)
        sample_batch_size = 32  # 4에서 32로 증가
        test_loader = DataLoader(train_dataset, batch_size=sample_batch_size, shuffle=False, collate_fn=self.collate_fn)
        
        # Test memory usage with first batch
        self.query_model.train()  # eval() 대신 train()으로 (실제 학습과 동일한 메모리 사용)
        self.api_model.eval()
        
        try:
            for queries, pos_api_texts in test_loader:
                # Measure memory usage
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # 실제 학습과 동일한 forward pass
                query_emb = self.get_query_embedding(queries)
                
                # Process positive API embeddings
                all_pos_apis = []
                for apis in pos_api_texts:
                    all_pos_apis.extend(apis)
                
                pos_api_emb = self.get_api_embedding(all_pos_apis)
                
                # Reshape positive embeddings
                pos_embeddings = []
                start_idx = 0
                for apis in pos_api_texts:
                    end_idx = start_idx + len(apis)
                    pos_embeddings.append(pos_api_emb[start_idx:end_idx])
                    start_idx = end_idx
                
                # Pad to same length
                max_len = max(len(emb) for emb in pos_embeddings)
                padded_pos_embeddings = []
                for emb in pos_embeddings:
                    if len(emb) < max_len:
                        padding = torch.zeros(max_len - len(emb), emb.shape[1], device=self.device)
                        emb = torch.cat([emb, padding], dim=0)
                    padded_pos_embeddings.append(emb)
                
                pos_api_embs = torch.stack(padded_pos_embeddings)
                
                # Calculate loss (실제 학습과 동일)
                loss = self.simplified_loss_function(query_emb, pos_api_embs)
                
                # Backward pass도 시뮬레이션 (메모리 사용량 증가)
                loss.backward()
                
                # Check memory usage
                max_memory_used = torch.cuda.max_memory_allocated() / (1024**3)  # GB
                current_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                
                print(f"📊 Memory usage for batch_size={sample_batch_size}:")
                print(f"   - Peak memory: {max_memory_used:.2f} GB")
                print(f"   - Current memory: {current_memory:.2f} GB")
                
                # Calculate optimal batch size using 85% of available VRAM
                optimal_batch_size = int(sample_batch_size * (available_memory / max_memory_used))
                
                # Limit batch size (더 큰 상한선으로 증가)
                optimal_batch_size = max(32, min(optimal_batch_size, 512))  # 최소 32, 최대 512
                
                print(f"🎯 Calculated optimal batch size: {optimal_batch_size}")
                print(f"   - Available VRAM: {available_memory:.2f} GB")
                print(f"   - Estimated memory usage: {max_memory_used * (optimal_batch_size / sample_batch_size):.2f} GB")
                print(f"   - VRAM utilization: {(max_memory_used * (optimal_batch_size / sample_batch_size) / total_vram) * 100:.1f}%")
                
                # Clean up
                del query_emb, pos_api_emb, pos_api_embs, loss
                torch.cuda.empty_cache()
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️  Sample batch size too large, reducing...")
                # 더 작은 batch size로 재시도
                sample_batch_size = 16
                test_loader = DataLoader(train_dataset, batch_size=sample_batch_size, shuffle=False, collate_fn=self.collate_fn)
                
                try:
                    for queries, pos_api_texts in test_loader:
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        
                        query_emb = self.get_query_embedding(queries)
                        
                        all_pos_apis = []
                        for apis in pos_api_texts:
                            all_pos_apis.extend(apis)
                        
                        pos_api_emb = self.get_api_embedding(all_pos_apis)
                        
                        pos_embeddings = []
                        start_idx = 0
                        for apis in pos_api_texts:
                            end_idx = start_idx + len(apis)
                            pos_embeddings.append(pos_api_emb[start_idx:end_idx])
                            start_idx = end_idx
                        
                        max_len = max(len(emb) for emb in pos_embeddings)
                        padded_pos_embeddings = []
                        for emb in pos_embeddings:
                            if len(emb) < max_len:
                                padding = torch.zeros(max_len - len(emb), emb.shape[1], device=self.device)
                                emb = torch.cat([emb, padding], dim=0)
                            padded_pos_embeddings.append(emb)
                        
                        pos_api_embs = torch.stack(padded_pos_embeddings)
                        loss = self.simplified_loss_function(query_emb, pos_api_embs)
                        loss.backward()
                        
                        max_memory_used = torch.cuda.max_memory_allocated() / (1024**3)
                        optimal_batch_size = int(sample_batch_size * (available_memory / max_memory_used))
                        optimal_batch_size = max(16, min(optimal_batch_size, 256))
                        
                        print(f"🎯 Recalculated optimal batch size: {optimal_batch_size}")
                        break
                        
                except RuntimeError as e2:
                    print("⚠️  Still too large, using minimum batch size")
                    optimal_batch_size = 16
            else:
                raise e
        
        return optimal_batch_size
    
    def train(self, 
              batch_size: int = 64,  # 기본값 64으로 증가
              epochs: int = 20,
              learning_rate: float = 3e-4,  # Increased for LR decay
              save_path: str = "/home/jhlee/librarian/trained_query_model",
              resume_from: str = None):
        """Train the query embedding model"""
        print(f"🚀 Starting query embedding training...")
        print(f"📊 Epochs: {epochs}")
        print(f"📊 Learning rate: {learning_rate}")
        
        # Load models
        if not self.load_models():
            return False
        
        # Load datasets
        train_dataset = QueryEmbeddingDataset(TRAIN_PATH)
        val_dataset = QueryEmbeddingDataset(EVAL_PATH)
        
        # Calculate optimal batch size if not provided
        if batch_size is None:
            batch_size = self.analyze_dataset_and_calculate_batch_size(train_dataset, val_dataset)
            batch_size = max(batch_size, 16)  # 최소값 16으로 제한
        
        print(f"📊 Batch size: {batch_size}")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        
        # Setup optimizer (only for query model)
        optimizer = torch.optim.AdamW(self.query_model.parameters(), lr=learning_rate)
        
        # Gradient accumulation for effective larger batch size (더 큰 target으로 증가)
        target_effective_batch_size = 1024  # 128에서 1024로 증가
        accumulation_steps = max(1, target_effective_batch_size // batch_size)
        print(f"📈 Training with batch_size={batch_size}, accumulation_steps={accumulation_steps}")
        print(f"   - Effective batch size: {batch_size * accumulation_steps}")
        print(f"   - Target effective batch size: {target_effective_batch_size}")
        
        # Setup OneCycleLR scheduler
        steps_per_epoch = len(train_loader)
        # Gradient accumulation을 고려한 실제 스케줄러 스텝 수 계산
        actual_scheduler_steps = epochs * (steps_per_epoch // accumulation_steps)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.01,  # 더 큰 max_lr (0.01)
            total_steps=actual_scheduler_steps,  # 실제 스케줄러 호출 횟수
            pct_start=0.1,  # 10%에서 max_lr 도달
            anneal_strategy='cos',
            div_factor=10,  # 초기 lr = max_lr / 10
            final_div_factor=1e4  # 마지막 lr = max_lr / 1e4
        )
        print(f"📊 Learning rate scheduler: OneCycleLR (max_lr: 0.01)")
        print(f"   - Total steps: {epochs * steps_per_epoch} (optimizer steps)")
        print(f"   - Scheduler steps: {actual_scheduler_steps} (actual scheduler calls)")
        print(f"   - Max LR will be reached at step: {int(actual_scheduler_steps * 0.1)}")
        
        # Training loop
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        start_epoch = 0
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            print(f"📂 Resuming from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.query_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # OneCycleLR은 step 수가 달라질 수 있으므로, 새로 생성
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            patience_counter = checkpoint['patience_counter']
            print(f"✅ Resumed from epoch {start_epoch-1}, best_val_loss: {best_val_loss:.6f}")
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(BASE_DIR, 'query_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            
            # Training
            self.query_model.train()
            train_loss = 0
            train_count = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            optimizer.zero_grad()
            
            for batch_idx, (queries, pos_api_texts) in enumerate(pbar):
                try:
                    # Get query embeddings
                    query_emb, query_magnitude = self.get_query_embedding(queries)
                    
                    # Get positive API embeddings
                    all_pos_apis = []
                    for apis in pos_api_texts:
                        all_pos_apis.extend(apis)
                    
                    pos_api_emb = self.get_api_embedding(all_pos_apis)
                    
                    # Reshape positive embeddings
                    pos_embeddings = []
                    start_idx = 0
                    for apis in pos_api_texts:
                        end_idx = start_idx + len(apis)
                        pos_embeddings.append(pos_api_emb[start_idx:end_idx])
                        start_idx = end_idx
                    
                    # Pad to same length
                    max_len = max(len(emb) for emb in pos_embeddings)
                    padded_pos_embeddings = []
                    for emb in pos_embeddings:
                        if len(emb) < max_len:
                            padding = torch.zeros(max_len - len(emb), emb.shape[1], device=self.device)
                            emb = torch.cat([emb, padding], dim=0)
                        padded_pos_embeddings.append(emb)
                    
                    pos_api_embs = torch.stack(padded_pos_embeddings)
                    
                    # Calculate loss
                    loss = self.magnitude_aware_loss_function(query_emb, query_magnitude, pos_api_embs)
                    
                    # Gradient accumulation
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()  # OneCycleLR step
                    
                    train_loss += loss.item() * len(queries) * accumulation_steps
                    train_count += len(queries)
                    
                    # Memory cleanup
                    del query_emb, query_magnitude, pos_api_emb, pos_api_embs
                    torch.cuda.empty_cache()
                    
                    # Progress bar
                    elapsed = time.time() - start_time
                    batches_done = batch_idx + 1
                    batches_total = len(train_loader)
                    eta = (elapsed / batches_done) * (batches_total - batches_done) if batches_done > 0 else 0
                    pbar.set_postfix({
                        'loss': loss.item() * accumulation_steps, 
                        'elapsed': f"{elapsed/60:.1f}m", 
                        'eta': f"{eta/60:.1f}m",
                        'lr': scheduler.get_last_lr()[0]
                    })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"GPU OOM at batch {batch_idx}, skipping...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1} finished in {epoch_time/60:.2f} minutes")
            
            # Validation
            val_loss = self.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}: Train Loss = {train_loss/train_count:.6f}, Val Loss = {val_loss:.6f}")
            
            # Save checkpoint every epoch
            checkpoint_path = os.path.join(checkpoint_dir, f'query_checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.query_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                os.makedirs(save_path, exist_ok=True)
                torch.save({
                                'model_state_dict': self.query_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss
                }, os.path.join(save_path, 'best_model.pt'))
                print(f"✅ Best model saved (val_loss: {val_loss:.6f})")
                
                # Save best checkpoint
                best_checkpoint_path = os.path.join(checkpoint_dir, 'best_query_checkpoint.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.query_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'val_loss': val_loss
                }, best_checkpoint_path)
                print(f"💾 Best checkpoint saved: {best_checkpoint_path}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
            
            # OneCycleLR은 step이 batch마다 적용됨 (에포크 끝에서 step 불필요)
            current_lr = scheduler.get_last_lr()[0]
            print(f"📊 Learning rate: {current_lr:.2e}")
        
        print("✅ Training completed!")
        return True
    
    def evaluate(self, val_loader):
        """Evaluate the model"""
        self.query_model.eval()
        total_loss = 0
        count = 0
        
        with torch.no_grad():
            for queries, pos_api_texts in val_loader:
                try:
                    # Get query embeddings
                    query_emb, query_magnitude = self.get_query_embedding(queries)
                    
                    # Get positive API embeddings
                    all_pos_apis = []
                    for apis in pos_api_texts:
                        all_pos_apis.extend(apis)
                    
                    pos_api_emb = self.get_api_embedding(all_pos_apis)
                    
                    # Reshape positive embeddings
                    pos_embeddings = []
                    start_idx = 0
                    for apis in pos_api_texts:
                        end_idx = start_idx + len(apis)
                        pos_embeddings.append(pos_api_emb[start_idx:end_idx])
                        start_idx = end_idx
                    
                    # Pad to same length
                    max_len = max(len(emb) for emb in pos_embeddings)
                    padded_pos_embeddings = []
                    for emb in pos_embeddings:
                        if len(emb) < max_len:
                            padding = torch.zeros(max_len - len(emb), emb.shape[1], device=self.device)
                            emb = torch.cat([emb, padding], dim=0)
                        padded_pos_embeddings.append(emb)
                    
                    pos_api_embs = torch.stack(padded_pos_embeddings)
                    
                    # Calculate loss
                    loss = self.magnitude_aware_loss_function(query_emb, query_magnitude, pos_api_embs)
                    
                    total_loss += loss.item() * len(queries)
                    count += len(queries)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        continue
                    else:
                        raise e
        
        return total_loss / count if count > 0 else 0


def main():
    """Main training function"""
    print("🚀 Query Embedding Model Training (Improved)")
    print("=" * 50)
    
    # Check for resume argument
    import sys
    resume_from = None
    if len(sys.argv) > 1:
        resume_from = sys.argv[1]
        print(f"Will resume from: {resume_from}")
    
    trainer = QueryEmbeddingTrainer()
    trainer.train(resume_from=resume_from)


if __name__ == "__main__":
    main() 