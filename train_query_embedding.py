#!/usr/bin/env python3
"""
Dynamic Direction-focused query embedding model training
Each query gets its own predicted scale factor for better magnitude handling
"""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any, Tuple
import random
import argparse
import time
import pickle

# Configuration
TOOLBENCH_MODEL_NAME = "ToolBench/ToolBench_IR_bert_based_uncased"
TRAIN_DATA_PATH = "training_data/train_dataset.json"
VAL_DATA_PATH = "training_data/val_dataset.json"
OUTPUT_DIR = "trained_query_model"
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = "best_model.pt"

class QueryEmbeddingDataset(Dataset):
    """Dataset for query embedding training"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loading dataset from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples")
        
        # Validate label vectors
        magnitudes = []
        for sample in self.data:
            label_vector = torch.tensor(sample['label_vector'], dtype=torch.float32)
            magnitude = torch.norm(label_vector, p=2).item()
            magnitudes.append(magnitude)
        
        print(f"Label vector statistics:")
        print(f"  - Mean magnitude: {np.mean(magnitudes):.4f}")
        print(f"  - Std magnitude: {np.std(magnitudes):.4f}")
        print(f"  - Min magnitude: {np.min(magnitudes):.4f}")
        print(f"  - Max magnitude: {np.max(magnitudes):.4f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        query_text = sample['query_text']
        label_vector = torch.tensor(sample['label_vector'], dtype=torch.float32)
        
        # Simple encoding without augmentation
        encoding = self.tokenizer(
            query_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Handle tensor dimensions
        for key in ['input_ids', 'attention_mask', 'token_type_ids']:
            if encoding[key].dim() == 2:
                encoding[key] = encoding[key][0]
            if encoding[key].dim() > 1:
                encoding[key] = encoding[key].squeeze()
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding['token_type_ids'],
            'label_vector': label_vector
        }

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

def direction_loss(pred, target):
    """Stable direction loss for better embedding alignment"""
    # Normalize both vectors to unit vectors
    pred_norm = F.normalize(pred, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)
    
    # Cosine similarity
    cosine_sim = torch.sum(pred_norm * target_norm, dim=1)
    cosine_sim = torch.clamp(cosine_sim, min=-1.0, max=1.0)
    
    # Stable loss components
    
    # 1. Cosine distance loss (primary)
    cosine_distance = 1.0 - cosine_sim
    
    # 2. Angular distance loss (more sensitive to small angle differences)
    angular_distance = torch.acos(torch.abs(cosine_sim))
    
    # Combine losses with stable weights
    total_direction_loss = 0.7 * cosine_distance + 0.3 * angular_distance
    
    return torch.mean(total_direction_loss)

def magnitude_loss(pred, target):
    """Stable magnitude loss with better stability"""
    pred_magnitude = torch.norm(pred, p=2, dim=1)
    target_magnitude = torch.norm(target, p=2, dim=1)
    
    # Use MSE loss for stability
    return F.mse_loss(pred_magnitude, target_magnitude)

def compute_losses(model, batch, device, direction_weight=2.0, magnitude_weight=1.0):
    """Compute losses with balanced weighting for stable learning"""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)
    label_vectors = batch['label_vector'].to(device)
    
    # Forward pass
    pred_embeddings = model(input_ids, attention_mask, token_type_ids)
    
    # Direction loss (balanced focus)
    dir_loss = direction_loss(pred_embeddings, label_vectors)
    
    # Magnitude loss (balanced weight)
    mag_loss = magnitude_loss(pred_embeddings, label_vectors)
    
    # Overall MSE loss
    mse_loss = F.mse_loss(pred_embeddings, label_vectors)
    
    # Total loss with balanced emphasis
    total_loss = mse_loss + direction_weight * dir_loss + magnitude_weight * mag_loss
    
    return {
        'total_loss': total_loss,
        'mse_loss': mse_loss,
        'direction_loss': dir_loss,
        'magnitude_loss': mag_loss
    }

def custom_collate_fn(batch):
    """Custom collate function"""
    input_ids = []
    attention_masks = []
    token_type_ids = []
    label_vectors = []
    
    for item in batch:
        input_ids.append(item['input_ids'])
        attention_masks.append(item['attention_mask'])
        token_type_ids.append(item['token_type_ids'])
        label_vectors.append(item['label_vector'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'token_type_ids': torch.stack(token_type_ids),
        'label_vector': torch.stack(label_vectors)
    }

def evaluate_model(model, val_loader, device):
    """Evaluate model with detailed direction analysis"""
    model.eval()
    total_mse = 0.0
    total_cosine = 0.0
    total_direction_loss = 0.0
    total_magnitude_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            label_vectors = batch['label_vector'].to(device)
            
            # Get model predictions
            pred_embeddings = model(input_ids, attention_mask, token_type_ids)
            
            # MSE loss
            mse_loss = F.mse_loss(pred_embeddings, label_vectors)
            total_mse += mse_loss.item()
            
            # Cosine similarity
            pred_norm = F.normalize(pred_embeddings, p=2, dim=1)
            label_norm = F.normalize(label_vectors, p=2, dim=1)
            cosine_sim = torch.mean(torch.sum(pred_norm * label_norm, dim=1)).item()
            total_cosine += cosine_sim
            
            # Direction loss
            direction_loss_val = direction_loss(pred_embeddings, label_vectors)
            total_direction_loss += direction_loss_val.item()
            
            # Magnitude loss
            magnitude_loss_val = magnitude_loss(pred_embeddings, label_vectors)
            total_magnitude_loss += magnitude_loss_val.item()
            
            num_samples += 1
    
    avg_mse = total_mse / num_samples if num_samples > 0 else 0.0
    avg_cosine = total_cosine / num_samples if num_samples > 0 else 0.0
    avg_direction_loss = total_direction_loss / num_samples if num_samples > 0 else 0.0
    avg_magnitude_loss = total_magnitude_loss / num_samples if num_samples > 0 else 0.0
    
    return avg_mse, avg_cosine, avg_direction_loss, avg_magnitude_loss

def train_model(args):
    """Main training function with dynamic scale prediction"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(TOOLBENCH_MODEL_NAME)
    model = DirectionFocusedQueryEmbeddingModel(TOOLBENCH_MODEL_NAME, dropout=args.dropout)
    model = model.to(device)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = QueryEmbeddingDataset(args.train_data_path, tokenizer, max_length=args.max_length)
    val_dataset = QueryEmbeddingDataset(args.val_data_path, tokenizer, max_length=args.max_length)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Optimizer with stable parameters
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    print(f"Training strategy: Dynamic direction-focused with AdamW optimizer")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Weight decay: {args.weight_decay}")
    print(f"  - Direction weight: {args.direction_weight}")
    print(f"  - Magnitude weight: {args.magnitude_weight}")
    print(f"  - Total epochs: {args.num_epochs}")
    
    # Stable learning rate scheduler with linear warmup and decay
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Load checkpoint if specified
    start_epoch = args.start_epoch
    best_score = float('inf')
    global_step = 0
    
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"Loading checkpoint from {args.resume_from_checkpoint}...")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['best_score']
        global_step = start_epoch * len(train_loader)
        
        print(f"Resumed from epoch {checkpoint['epoch']}, best_score: {best_score:.4f}")
    else:
        print("Starting training from scratch...")
    
    # Initialize logging
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    print(f"Logging to: {log_file}")
    
    # Training loop
    print("Starting dynamic training...")
    
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            # Debug: Check model outputs and label magnitudes
            if global_step == 0:
                with torch.no_grad():
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    label_vectors = batch['label_vector'].to(device)
                    
                    pred_embeddings = model(input_ids, attention_mask, token_type_ids)
                    
                    pred_magnitudes = torch.norm(pred_embeddings, p=2, dim=1)
                    label_magnitudes = torch.norm(label_vectors, p=2, dim=1)
                    
                    pred_norm = F.normalize(pred_embeddings, p=2, dim=1)
                    label_norm = F.normalize(label_vectors, p=2, dim=1)
                    cosine_sim = torch.sum(pred_norm * label_norm, dim=1)
                    
                    print(f"\n🔍 DEBUG - First batch:")
                    print(f"  Pred magnitudes - Mean: {pred_magnitudes.mean().item():.4f}, Std: {pred_magnitudes.std().item():.4f}")
                    print(f"  Label magnitudes - Mean: {label_magnitudes.mean().item():.4f}, Std: {label_magnitudes.std().item():.4f}")
                    print(f"  Cosine similarity - Mean: {cosine_sim.mean().item():.4f}, Range: [{cosine_sim.min().item():.4f}, {cosine_sim.max().item():.4f}]")
                    
                    # Debug scale factors
                    cls_token = model.encoder(input_ids, attention_mask, token_type_ids, output_hidden_states=True, return_dict=True).last_hidden_state[:, 0]
                    raw_scale = model.scale_predictor(cls_token)
                    dynamic_scale = F.softplus(raw_scale) + 1.0
                    print(f"  Raw scale outputs - Mean: {raw_scale.mean().item():.4f}, Range: [{raw_scale.min().item():.4f}, {raw_scale.max().item():.4f}]")
                    print(f"  Dynamic scales - Mean: {dynamic_scale.mean().item():.4f}, Range: [{dynamic_scale.min().item():.4f}, {dynamic_scale.max().item():.4f}]")
            
            # Compute losses
            losses = compute_losses(
                model, batch, device, 
                direction_weight=args.direction_weight,
                magnitude_weight=args.magnitude_weight
            )
            total_loss = losses['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            scheduler.step()
            
            # Log losses
            epoch_losses.append(total_loss.item())
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'mse': f"{losses['mse_loss'].item():.4f}",
                'direction': f"{losses['direction_loss'].item():.4f}",
                'magnitude': f"{losses['magnitude_loss'].item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to file
            if global_step % args.log_steps == 0:
                log_entry = f"Step {global_step}: total_loss={total_loss.item():.4f}, mse_loss={losses['mse_loss'].item():.4f}, direction_loss={losses['direction_loss'].item():.4f}, magnitude_loss={losses['magnitude_loss'].item():.4f}, lr={scheduler.get_last_lr()[0]:.2e}\n"
                with open(log_file, 'a') as f:
                    f.write(log_entry)
            
            global_step += 1
        
        # Evaluate on validation set
        if (epoch + 1) % args.eval_steps == 0:
            print(f"\nEvaluating epoch {epoch+1}...")
            val_mse, val_cosine, val_direction_loss, val_magnitude_loss = evaluate_model(model, val_loader, device)
            
            # Log validation scores
            val_log_entry = f"Epoch {epoch+1}: Validation MSE = {val_mse:.4f}, Cosine similarity = {val_cosine:.4f}, Direction loss = {val_direction_loss:.4f}, Magnitude loss = {val_magnitude_loss:.4f}\n"
            with open(log_file, 'a') as f:
                f.write(val_log_entry)
            
            print(f"Validation MSE: {val_mse:.4f}, Cosine similarity: {val_cosine:.4f}")
            print(f"Direction loss: {val_direction_loss:.4f}, Magnitude loss: {val_magnitude_loss:.4f}")
            
            # Save best model (using direction loss as primary metric)
            if val_direction_loss < best_score:
                best_score = val_direction_loss
                best_model_path = os.path.join(args.output_dir, BEST_MODEL_PATH)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_score': best_score,
                    'args': args
                }, best_model_path)
                print(f"New best model saved with direction loss: {best_score:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_score': best_score,
                'args': args
            }, checkpoint_path)
            
            model.train()
    
    print(f"Training completed! Best validation direction loss: {best_score:.4f}")
    
    # Save final training summary
    summary_log = f"\nTraining Summary:\nBest validation direction loss: {best_score:.4f}\nTotal epochs: {args.num_epochs}\n"
    with open(log_file, 'a') as f:
        f.write(summary_log)

def main():
    parser = argparse.ArgumentParser(description='Train dynamic direction-focused query embedding model')
    
    # Data arguments
    parser.add_argument('--train_data_path', type=str, default=TRAIN_DATA_PATH)
    parser.add_argument('--val_data_path', type=str, default=VAL_DATA_PATH)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR)
    
    # Model arguments
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=15)  # Stable training
    parser.add_argument('--learning_rate', type=float, default=2e-5)  # Higher LR for better convergence
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--direction_weight', type=float, default=2.0)  # Balanced weight on direction
    parser.add_argument('--magnitude_weight', type=float, default=1.0)  # Balanced weight on magnitude
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # Logging arguments
    parser.add_argument('--log_steps', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=1)
    
    # Checkpoint arguments
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    
    args = parser.parse_args()
    
    print("🚀 Starting dynamic direction-focused query embedding model training...")
    print(f"Arguments: {args}")
    
    train_model(args)
    
    print("✅ Dynamic training completed!")

if __name__ == "__main__":
    main() 