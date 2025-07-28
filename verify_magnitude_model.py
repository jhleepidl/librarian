#!/usr/bin/env python3
"""
Verify trained magnitude model by comparing actual API embedding magnitudes 
with the model's regression output
"""

import sys
import os
import json
import torch
import numpy as np
from typing import List, Dict, Any
import logging
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(BASE_DIR, "data/test.json")
TRAINED_MODEL_PATH = os.path.join(BASE_DIR, "trained_magnitude_model/best_model.pt")

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


class MagnitudeModelVerifier:
    """
    Verify the trained magnitude model by comparing actual vs predicted magnitudes
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.query_model = None
        self.api_model = None
        
    def load_models(self):
        """Load query, API, and trained magnitude models"""
        print("📦 Loading models...")
        
        # Load trained magnitude model (same as in training)
        self.query_model = MagnitudeRegressionModel(TOOLBENCH_MODEL_NAME)
        if os.path.exists(TRAINED_MODEL_PATH):
            try:
                checkpoint = torch.load(TRAINED_MODEL_PATH, map_location=self.device, weights_only=False)
                self.query_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded trained model from {TRAINED_MODEL_PATH}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                return False
        else:
            print(f"❌ Trained model not found at {TRAINED_MODEL_PATH}")
            return False
        
        self.query_model.to(self.device)
        self.query_model.eval()
        
        # Load API model (frozen, for target calculation)
        self.api_model = SentenceTransformer(TOOLBENCH_MODEL_NAME)
        self.api_model.to(self.device)
        
        # Freeze API model
        for param in self.api_model.parameters():
            param.requires_grad = False
        
        print("✅ All models loaded successfully")
        return True
    
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
    
    def calculate_actual_magnitude(self, relevant_apis: List[List[Dict[str, Any]]]) -> torch.Tensor:
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
            api_embeddings_norm = torch.nn.functional.normalize(api_embeddings, p=2, dim=-1)
            
            # Sum the normalized embeddings
            api_sum = api_embeddings_norm.sum(dim=0)  # Sum across APIs
            
            # Calculate magnitude (L2 norm)
            magnitude = torch.norm(api_sum, p=2)
            
            target_magnitudes.append(magnitude)
        
        return torch.stack(target_magnitudes)
    
    def predict_magnitude(self, queries: List[str]) -> torch.Tensor:
        """Predict magnitude using trained model"""
        # Tokenize queries
        tokenized = self.query_model.base_model.tokenize(queries)
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predicted_magnitudes = self.query_model(input_ids, attention_mask)
        
        return predicted_magnitudes
    
    def load_test_data(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Load test data samples"""
        print(f"📦 Loading test data from {TEST_PATH}")
        
        with open(TEST_PATH, 'r', encoding='utf-8') as f:
            all_samples = json.load(f)
        
        # Take only the first num_samples
        samples = all_samples[:num_samples]
        
        print(f"✅ Loaded {len(samples)} test samples")
        return samples
    
    def verify_model(self, test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify the trained model by comparing actual vs predicted magnitudes"""
        print("🔍 Verifying magnitude model...")
        
        results = {
            'actual_magnitudes': [],
            'predicted_magnitudes': [],
            'queries': [],
            'errors': [],
            'correlations': []
        }
        
        for i, sample in enumerate(tqdm(test_samples, desc="Processing samples")):
            try:
                query = sample['query']
                relevant_apis = sample['relevant_apis']
                
                # Calculate actual magnitude
                actual_magnitude = self.calculate_actual_magnitude([relevant_apis])
                
                # Predict magnitude
                predicted_magnitude = self.predict_magnitude([query])
                
                # Store results
                results['actual_magnitudes'].append(actual_magnitude.item())
                results['predicted_magnitudes'].append(predicted_magnitude.item())
                results['queries'].append(query)
                
                # Calculate error
                error = abs(actual_magnitude.item() - predicted_magnitude.item())
                results['errors'].append(error)
                
            except Exception as e:
                print(f"❌ Error processing sample {i}: {e}")
                continue
        
        # Calculate statistics
        actual_array = np.array(results['actual_magnitudes'])
        predicted_array = np.array(results['predicted_magnitudes'])
        errors_array = np.array(results['errors'])
        
        # Calculate correlation
        correlation = np.corrcoef(actual_array, predicted_array)[0, 1]
        results['correlations'].append(correlation)
        
        # Calculate metrics
        mae = np.mean(errors_array)
        mse = np.mean(errors_array ** 2)
        rmse = np.sqrt(mse)
        
        print(f"\n📊 Verification Results:")
        print(f"   Correlation: {correlation:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   MSE: {mse:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   Mean Actual Magnitude: {np.mean(actual_array):.4f}")
        print(f"   Mean Predicted Magnitude: {np.mean(predicted_array):.4f}")
        
        return results
    
    def plot_results(self, results: Dict[str, Any], save_path: str = "magnitude_verification_results.png"):
        """Plot verification results"""
        actual_array = np.array(results['actual_magnitudes'])
        predicted_array = np.array(results['predicted_magnitudes'])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scatter plot of actual vs predicted
        axes[0, 0].scatter(actual_array, predicted_array, alpha=0.6)
        axes[0, 0].plot([actual_array.min(), actual_array.max()], 
                        [actual_array.min(), actual_array.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Magnitude')
        axes[0, 0].set_ylabel('Predicted Magnitude')
        axes[0, 0].set_title('Actual vs Predicted Magnitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Error distribution
        errors = np.array(results['errors'])
        axes[0, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Absolute Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Magnitude distribution comparison
        axes[1, 0].hist(actual_array, bins=30, alpha=0.7, label='Actual', edgecolor='black')
        axes[1, 0].hist(predicted_array, bins=30, alpha=0.7, label='Predicted', edgecolor='black')
        axes[1, 0].set_xlabel('Magnitude')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Magnitude Distribution Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Error vs Actual magnitude
        axes[1, 1].scatter(actual_array, errors, alpha=0.6)
        axes[1, 1].set_xlabel('Actual Magnitude')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title('Error vs Actual Magnitude')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Results plot saved to {save_path}")
        plt.show()
    
    def run_verification(self, num_samples: int = 100):
        """Run the complete verification process"""
        print("🚀 Starting magnitude model verification...")
        
        # Load models
        if not self.load_models():
            return
        
        # Load test data
        test_samples = self.load_test_data(num_samples)
        
        if not test_samples:
            print("❌ No test samples loaded")
            return
        
        # Run verification
        results = self.verify_model(test_samples)
        
        # Plot results
        self.plot_results(results)
        
        # Save detailed results
        output_path = "magnitude_verification_detailed_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"📄 Detailed results saved to {output_path}")
        
        return results


def main():
    """Main function"""
    verifier = MagnitudeModelVerifier()
    verifier.run_verification(num_samples=100)


if __name__ == "__main__":
    main() 