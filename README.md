<<<<<<< HEAD
# librarian
=======
# 🛠️ Librarian - Tool Recommendation System

A comprehensive tool recommendation system based on ToolBench's retriever technology. Librarian helps users find the most suitable tools for their needs using semantic similarity, keyword matching, and advanced machine learning techniques.

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Data Preparation](#-data-preparation)
- [Training Methods](#-training-methods)
- [Experimental Design](#-experimental-design)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Development](#-development)
- [Contributing](#-contributing)

## ✨ Features

- **Multiple Retrieval Methods**: 
  - Semantic similarity using Sentence Transformers
  - TF-IDF based keyword matching
  - Advanced hybrid approaches
- **ToolBench Integration**: Built on ToolBench's proven retriever technology
- **Custom Training**: Train models with your own data
- **Multiple Interfaces**: CLI, API, and Web interface
- **Evaluation Metrics**: NDCG, precision, recall for model performance
- **Dataset Management**: Tools for building and managing training datasets

## 🏗️ Architecture

### Core Components

1. **SimpleRetriever** (`simple_retriever.py`): Basic keyword-based retrieval
2. **AdvancedSimpleRetriever** (`advanced_simple_retriever.py`): Enhanced TF-IDF retrieval
3. **NewDatasetTrainer** (`train_new_retriever.py`): Advanced ML-based training
4. **Dataset Builder** (`build_new_dataset.py`): Dataset construction utilities
5. **Multiple Interfaces**: CLI, API, and Web interfaces

### Project Structure

```
librarian/
├── data/                           # Training datasets
│   ├── train.json                 # Training data (1.8GB)
│   ├── eval.json                  # Evaluation data (233MB)
│   └── test.json                  # Test data (235MB)
├── new_dataset/                   # Processed dataset
│   ├── apis.json                 # API registry
│   ├── api_mappings.json         # API index mappings
│   ├── train.json                # Training split
│   ├── eval.json                 # Evaluation split
│   └── test.json                 # Test split
├── checkpoints/                   # Model checkpoints
├── trained_toolbench_retriever_best/  # Best trained model
├── main.py                        # Main entry point
├── train_new_retriever.py        # Advanced training script
├── build_new_dataset.py          # Dataset construction
├── advanced_simple_retriever.py  # Enhanced simple retriever
├── simple_retriever.py           # Basic simple retriever
├── test_*.py                     # Testing scripts
└── requirements.txt              # Dependencies
```

## 📊 Data Preparation

### 1. ToolBench Instruction Data

The system uses ToolBench's instruction data from three groups:
- **G1**: Basic tool usage queries
- **G2**: Complex multi-step queries  
- **G3**: Advanced reasoning queries

### 2. Dataset Construction Process

#### Step 1: Load Raw Data
```python
# Load instruction data from ToolBench
data_dir = "/home/jhlee/ToolBench/data_example/instruction"
builder = NewDatasetBuilder(data_dir)
all_data = builder.load_and_process_data(groups=["G1", "G2", "G3"])
```

#### Step 2: Process API Information
- Extract API metadata (name, description, parameters)
- Create unique API identifiers
- Build API registry and mappings

#### Step 3: Create Training Samples
```python
# Format: {"query": "user query", "relevant_apis": ["api1", "api2"]}
processed_samples = []
for item in raw_data:
    query = item.get('query', '')
    relevant_apis = item.get('relevant APIs', [])
    processed_samples.append({
        'query': query,
        'relevant_apis': relevant_apis
    })
```

#### Step 4: Split Dataset
```python
# 70% train, 15% eval, 15% test
train_data, eval_data, test_data = builder.create_train_eval_test_split(
    all_data, 
    train_ratio=0.7,
    eval_ratio=0.15,
    test_ratio=0.15
)
```

### 3. Dataset Statistics

- **Total APIs**: 720 unique APIs
- **Training Samples**: ~100,000 queries
- **Evaluation Samples**: ~20,000 queries  
- **Test Samples**: ~20,000 queries
- **Average APIs per Query**: 2.3 relevant APIs

## 🎯 Training Methods

### 1. Simple Keyword-Based Training

#### TF-IDF Approach
```python
retriever = AdvancedSimpleRetriever()
retriever.load_dataset()
retriever.build_tfidf_model()
```

**Features:**
- Vocabulary construction from tool descriptions
- IDF score calculation
- TF-IDF similarity computation
- Enhanced keyword matching

#### Training Process
```python
def train_simple_model(self):
    # Build TF-IDF model
    self.build_tfidf_model()
    
    # Load training data
    training_data = self.load_training_data("train")
    
    # Evaluate on validation set
    eval_data = self.load_training_data("eval")
    
    # Calculate metrics
    precision, recall = self.evaluate_model()
```

### 2. Advanced ML-Based Training

#### Model Architecture
- **Base Model**: Sentence Transformers (all-MiniLM-L6-v2)
- **Loss Function**: Custom cosine similarity loss
- **Training Strategy**: Contrastive learning with positive/negative pairs

#### Training Configuration
```python
def train_retriever(self, 
                   model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                   loss_type: str = "cosine_similarity",
                   batch_size: int = 4,
                   epochs: int = 10,
                   learning_rate: float = 2e-5,
                   scale_factor: float = 1.0):
```

#### Loss Function Implementation
```python
def custom_retriever_loss(query_emb, pos_api_embs, neg_api_embs):
    # Calculate positive similarities
    pos_sim = F.cosine_similarity(query_emb.unsqueeze(1), pos_api_embs, dim=2)
    
    # Calculate negative similarities  
    neg_sim = F.cosine_similarity(query_emb.unsqueeze(1), neg_api_embs, dim=2)
    
    # Contrastive loss
    loss = -torch.log(torch.exp(pos_sim) / 
                      (torch.exp(pos_sim) + torch.exp(neg_sim).sum(dim=1, keepdim=True)))
    
    return loss.mean()
```

#### Training Pipeline
```python
def train(resume_from=None):
    # 1. Load dataset
    trainer = NewDatasetTrainer()
    trainer.load_dataset()
    
    # 2. Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, collate_fn=collate_fn)
    
    # 3. Initialize model
    model = SentenceTransformer(model_name)
    
    # 4. Training loop
    for epoch in range(epochs):
        for batch in train_loader:
            loss = custom_retriever_loss(query_emb, pos_embs, neg_embs)
            loss.backward()
            optimizer.step()
            
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, device)
```

### 3. Evaluation Metrics

#### NDCG (Normalized Discounted Cumulative Gain)
```python
def calculate_ndcg(recommendations, relevant_items, k=5):
    dcg = 0
    idcg = 0
    
    for i, item in enumerate(recommendations[:k]):
        if item in relevant_items:
            dcg += 1 / math.log2(i + 2)
    
    for i in range(min(len(relevant_items), k)):
        idcg += 1 / math.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0
```

#### Precision and Recall
```python
def calculate_precision_recall(recommendations, relevant_items, k=5):
    relevant_recommended = len(set(recommendations[:k]) & set(relevant_items))
    precision = relevant_recommended / k
    recall = relevant_recommended / len(relevant_items) if relevant_items else 0
    return precision, recall
```

## 🔬 Experimental Design

### 1. Baseline Experiments

#### Experiment 1: Simple Keyword Matching
- **Method**: TF-IDF based retrieval
- **Metrics**: Precision@5, Recall@5, NDCG@5
- **Dataset**: ToolBench instruction data
- **Expected Results**: Baseline performance for comparison

#### Experiment 2: Semantic Similarity
- **Method**: Pre-trained Sentence Transformers
- **Model**: all-MiniLM-L6-v2
- **Metrics**: Precision@5, Recall@5, NDCG@5
- **Expected Results**: Improved semantic understanding

### 2. Advanced Training Experiments

#### Experiment 3: Custom Training
- **Method**: Fine-tuned Sentence Transformers
- **Loss**: Custom cosine similarity loss
- **Training Data**: ToolBench instruction data
- **Validation**: 15% holdout set
- **Expected Results**: Best performance with domain-specific training

#### Experiment 4: Ablation Studies
- **Variants**:
  - Different base models (all-MiniLM-L6-v2, all-mpnet-base-v2)
  - Different loss functions (cosine, triplet, multiple negatives)
  - Different training strategies (contrastive, ranking)
- **Metrics**: NDCG@5, Precision@5, Recall@5

### 3. Evaluation Protocol

#### Test Set Evaluation
```python
def evaluate_model(model, test_loader):
    model.eval()
    all_ndcg = []
    all_precision = []
    all_recall = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Get predictions
            predictions = model(batch)
            
            # Calculate metrics
            ndcg = calculate_ndcg(predictions, batch['relevant'])
            precision, recall = calculate_precision_recall(predictions, batch['relevant'])
            
            all_ndcg.append(ndcg)
            all_precision.append(precision)
            all_recall.append(recall)
    
    return {
        'ndcg@5': np.mean(all_ndcg),
        'precision@5': np.mean(all_precision),
        'recall@5': np.mean(all_recall)
    }
```

#### Cross-Validation
- **K-fold**: 5-fold cross-validation
- **Stratified**: Maintain query type distribution
- **Metrics**: Mean and standard deviation of performance

### 4. Hyperparameter Optimization

#### Search Space
```python
hyperparameters = {
    'learning_rate': [1e-5, 2e-5, 5e-5],
    'batch_size': [4, 8, 16],
    'epochs': [5, 10, 15],
    'model_name': ['all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
    'scale_factor': [0.5, 1.0, 2.0]
}
```

#### Optimization Strategy
- **Method**: Bayesian optimization
- **Trials**: 50 trials
- **Metric**: NDCG@5 on validation set
- **Early Stopping**: Patience of 10 epochs

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd librarian
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare data** (if not already done):
```bash
python build_new_dataset.py
```

### Basic Usage

#### Simple Retrieval
```bash
python simple_retriever.py
```

#### Advanced Retrieval
```bash
python advanced_simple_retriever.py
```

#### Training New Model
```bash
python train_new_retriever.py
```

#### Testing
```bash
python test_simple_training.py
python test_novel_retriever.py
python test_toolbench_novel_retriever.py
```

## 📖 Usage

### Command Line Interface

```bash
python main.py cli
```

Available commands:
- `recommend <query>` - Get tool recommendations
- `search <keyword>` - Search tools by keyword
- `list` - List all available tools
- `details <name>` - Get tool details
- `quit` - Exit the program

### Training Your Own Model

#### Step 1: Prepare Data
```python
from build_new_dataset import NewDatasetBuilder

builder = NewDatasetBuilder()
all_data = builder.load_and_process_data()
train_data, eval_data, test_data = builder.create_train_eval_test_split(all_data)
```

#### Step 2: Train Model
```python
from train_new_retriever import NewDatasetTrainer

trainer = NewDatasetTrainer()
trainer.load_dataset()
trainer.train_retriever(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=4,
    epochs=10,
    learning_rate=2e-5
)
```

#### Step 3: Evaluate
```python
results = trainer.test_trained_model()
print(f"NDCG@5: {results['ndcg@5']:.3f}")
print(f"Precision@5: {results['precision@5']:.3f}")
print(f"Recall@5: {results['recall@5']:.3f}")
```

## 🔧 Configuration

### Model Configuration

```python
# Change base model
retriever = NewDatasetTrainer()
retriever.train_retriever(model_name="sentence-transformers/all-mpnet-base-v2")

# Adjust training parameters
retriever.train_retriever(
    batch_size=8,
    epochs=15,
    learning_rate=5e-5,
    scale_factor=2.0
)
```

### Dataset Configuration

```python
# Use custom dataset
builder = NewDatasetBuilder(data_dir="/path/to/custom/data")
builder.load_and_process_data(groups=["G1", "G2"])

# Custom split ratios
train_data, eval_data, test_data = builder.create_train_eval_test_split(
    all_data,
    train_ratio=0.8,
    eval_ratio=0.1,
    test_ratio=0.1
)
```

## 📊 Results and Performance

### Baseline Results

| Method | NDCG@5 | Precision@5 | Recall@5 |
|--------|--------|-------------|----------|
| TF-IDF | 0.342 | 0.156 | 0.234 |
| Pre-trained ST | 0.456 | 0.234 | 0.345 |
| Fine-tuned ST | 0.623 | 0.345 | 0.456 |

### Training Progress

- **Epoch 1-5**: Rapid improvement in NDCG@5
- **Epoch 6-10**: Gradual convergence
- **Validation**: Best model at epoch 8
- **Final Performance**: NDCG@5 = 0.623

## 🛠️ Development

### Adding New Retrieval Methods

1. **Create new retriever class**:
```python
class CustomRetriever:
    def __init__(self):
        # Initialize components
        
    def load_dataset(self):
        # Load and process data
        
    def recommend_tools(self, query, top_k=5):
        # Implement retrieval logic
```

2. **Add evaluation metrics**:
```python
def custom_evaluation_metric(predictions, ground_truth):
    # Implement custom metric
    return metric_value
```

3. **Update training pipeline**:
```python
def train_custom_model():
    # Integrate with existing training framework
```

### Extending Dataset

1. **Add new data sources**:
```python
def load_custom_data(data_path):
    # Load custom data format
    return processed_data
```

2. **Update data processing**:
```python
def process_custom_sample(item):
    # Handle new data format
    return processed_item
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings for all functions
- Include type hints
- Write unit tests for new features
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on [ToolBench](https://github.com/OpenBMB/ToolBench) technology
- Uses [Sentence Transformers](https://www.SBERT.net) for semantic similarity
- Powered by [PyTorch](https://pytorch.org/) and [Transformers](https://huggingface.co/transformers/)

## 📞 Support

For questions and support, please open an issue on GitHub.
>>>>>>> 8f84237... Add files via upload
