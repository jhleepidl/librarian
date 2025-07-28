#!/usr/bin/env python3
"""
Train a magnitude regression model using ToolBench base model
The model learns to output embeddings whose magnitude matches the sum of relevant API embeddings
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

# ToolBench model name
TOOLBENCH_MODEL_NAME = "ToolBench/ToolBench_IR_bert_based_uncased"


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


class MagnitudeDataset(Dataset):
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


class MagnitudeTrainer:
    """
    Trainer for magnitude regression model
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.query_model = None
        self.api_model = None
        
    def load_models(self):
        """Load query and API models"""
        print("📦 Loading models...")
        
        # Load query embedding model (for training)
        self.query_model = MagnitudeRegressionModel(TOOLBENCH_MODEL_NAME)
        self.query_model.to(self.device)
        
        # Load API embedding model (frozen, for target calculation)
        self.api_model = SentenceTransformer(TOOLBENCH_MODEL_NAME)
        self.api_model.to(self.device)
        
        # Freeze API model
        for param in self.api_model.parameters():
            param.requires_grad = False
        
        print(f"✅ Models loaded on {self.device}")
    
    def get_query_embedding(self, queries: List[str]) -> torch.Tensor:
        """Get query embeddings using the training model"""
        # Tokenize queries
        tokenized = self.query_model.base_model.tokenize(queries)
        
        # Move tokenized tensors to device
        for key in tokenized:
            if isinstance(tokenized[key], torch.Tensor):
                tokenized[key] = tokenized[key].to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.query_model.base_model(tokenized)
        
        return embeddings["sentence_embedding"]
    
    def get_api_embedding(self, api_texts: List[str]) -> torch.Tensor:
        """Get API embeddings using the frozen model"""
        # Tokenize API texts
        tokenized = self.api_model.tokenize(api_texts)
        
        # Move tokenized tensors to device
        for key in tokenized:
            if isinstance(tokenized[key], torch.Tensor):
                tokenized[key] = tokenized[key].to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.api_model(tokenized)
        
        return embeddings["sentence_embedding"]
    
    def format_api_text(self, api: Dict[str, Any]) -> str:
        """Format API information as text - only tool name and API name"""
        # Extract only tool name and API name
        tool_name = api.get('tool_name', '')
        api_name = api.get('api_name', '')
        
        # Combine into a single text
        api_text = f"Tool: {tool_name}, API: {api_name}"
        return api_text
    
    def calculate_target_magnitude(self, relevant_apis: List[List[Dict[str, Any]]]) -> torch.Tensor:
        """
        Calculate target magnitude for each query
        Target = magnitude of sum of normalized relevant API embeddings
        """
        target_magnitudes = []
        
        for apis in relevant_apis:
            # Format API texts
            api_texts = [self.format_api_text(api) for api in apis]
            
            # Get API embeddings
            api_embeddings = self.get_api_embedding(api_texts)
            
            # Normalize each API embedding
            api_embeddings_norm = F.normalize(api_embeddings, p=2, dim=-1)
            
            # Sum the normalized embeddings
            api_sum = api_embeddings_norm.sum(dim=0)  # Sum across APIs
            
            # Calculate magnitude (L2 norm)
            magnitude = torch.norm(api_sum, p=2)
            
            target_magnitudes.append(magnitude)
        
        return torch.stack(target_magnitudes)
    
    def magnitude_loss_function(self, predicted_magnitudes, target_magnitudes):
        """
        Loss function for magnitude regression
        """
        # MSE loss between predicted and target magnitudes
        loss = F.mse_loss(predicted_magnitudes, target_magnitudes)
        return loss
    
    def collate_fn(self, batch):
        """Custom collate function"""
        queries = [item['query'] for item in batch]
        relevant_apis = [item['relevant_apis'] for item in batch]
        
        return queries, relevant_apis
    
    def train(self, 
              batch_size: int = 32,
              epochs: int = 1,
              learning_rate: float = 1e-4,
              save_path: str = "/home/jhlee/librarian/trained_magnitude_model",
              resume_from: str = None):
        """
        Train the magnitude regression model
        """
        print("🚀 Starting magnitude regression training...")
        
        # Load models
        self.load_models()
        
        # Load datasets
        print("📊 Loading datasets...")
        train_dataset = MagnitudeDataset(TRAIN_PATH)
        val_dataset = MagnitudeDataset(EVAL_PATH)
        
        print(f"📈 Training samples: {len(train_dataset)}")
        print(f"📈 Validation samples: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(self.query_model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = OneCycleLR(optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=len(train_loader))
        
        # Resume training if specified
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            print(f"🔄 Resuming from {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.query_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch, epochs):
            print(f"\n📚 Epoch {epoch + 1}/{epochs}")
            
            # Training phase
            self.query_model.train()
            train_losses = []
            
            train_pbar = tqdm(train_loader, desc="Training")
            for batch_idx, (queries, relevant_apis) in enumerate(train_pbar):
                # Calculate target magnitudes
                target_magnitudes = self.calculate_target_magnitude(relevant_apis)
                target_magnitudes = target_magnitudes.to(self.device)
                
                # Tokenize queries
                tokenized = self.query_model.base_model.tokenize(queries)
                input_ids = tokenized['input_ids'].to(self.device)
                attention_mask = tokenized['attention_mask'].to(self.device)
                
                # Forward pass
                predicted_magnitudes = self.query_model(input_ids, attention_mask)
                
                # Calculate loss
                loss = self.magnitude_loss_function(predicted_magnitudes, target_magnitudes)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_losses.append(loss.item())
                train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                
                # Save checkpoint every 1000 batches
                if (batch_idx + 1) % 1000 == 0:
                    checkpoint_path = os.path.join(save_path, f'checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pt')
                    torch.save({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'model_state_dict': self.query_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item(),
                    }, checkpoint_path)
                    print(f"💾 Saved checkpoint: {checkpoint_path}")
            
            # Validation phase
            self.query_model.eval()
            val_losses = []
            
            val_pbar = tqdm(val_loader, desc="Validation")
            with torch.no_grad():
                for queries, relevant_apis in val_pbar:
                    # Calculate target magnitudes
                    target_magnitudes = self.calculate_target_magnitude(relevant_apis)
                    target_magnitudes = target_magnitudes.to(self.device)
                    
                    # Tokenize queries
                    tokenized = self.query_model.base_model.tokenize(queries)
                    input_ids = tokenized['input_ids'].to(self.device)
                    attention_mask = tokenized['attention_mask'].to(self.device)
                    
                    # Forward pass
                    predicted_magnitudes = self.query_model(input_ids, attention_mask)
                    
                    # Calculate loss
                    loss = self.magnitude_loss_function(predicted_magnitudes, target_magnitudes)
                    val_losses.append(loss.item())
                    
                    val_pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})
            
            # Calculate average losses
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            print(f"📊 Epoch {epoch + 1} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(save_path, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.query_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                }, best_model_path)
                print(f"🏆 New best model saved: {best_model_path}")
            
            # Save epoch checkpoint
            epoch_checkpoint_path = os.path.join(save_path, f'epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.query_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, epoch_checkpoint_path)
        
        print("✅ Training completed!")
        print(f"📁 Model saved to: {save_path}")


def main():
    """Main training function"""
    trainer = MagnitudeTrainer()
    
    # Training parameters
    trainer.train(
        batch_size=32,
        epochs=1,
        learning_rate=1e-4,
        save_path="/home/jhlee/librarian/trained_magnitude_model"
    )


if __name__ == "__main__":
    main() 