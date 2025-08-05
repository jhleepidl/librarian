# 🛠️ Librarian of Tools - Compositional Tool Recommendation System

A compositional tool recommendation system based on ToolBench's retriever with advanced query embedding models. Librarian helps users find the most suitable tool combination for their NL query using semantic similarity, dynamic scale prediction, and beam search algorithm.

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Model Training](#-model-training)
- [Performance Analysis](#-performance-analysis)


## 🏗️ Architecture

### Components

1. **Dataset Preparation** (`build_training_dataset.py`): Training dataset preparation
2. **Query Embedding Training** (`train_query_embedding.py`): Dynamic direction-focused model training
3. **Vector Database Building** (`build_vector_db.py`, `build_multiple_vector_dbs.py`): FAISS-based vector database construction
4. **Performance Analysis** (`performance_analysis_local.py`, `performance_analysis_hf.py`): Comprehensive evaluation tools
5. **Beam Search Analysis** (`beam_size_analysis.py`): Search algorithm optimization
6. **Vector Database Analysis** (`compare_multiple_vector_dbs.py`): Compare search accuracy between different DB size

## 🔧 Prerequisites

### ToolBench Data Requirements

**Important**: This project requires ToolBench data to function properly. You need to download and set up ToolBench data in the following location:

You can download the ToolBench data from the official repository:

[ToolBench Data Release](https://github.com/OpenBMB/ToolBench/tree/master?tab=readme-ov-file#data-release)

Follow the instructions in the linked section to obtain and extract the required data files.

```bash
../ToolBench/
├── data/
│   ├── instruction/           # For training data
│   │   ├── G1_instruction.json
│   │   ├── G2_instruction.json
│   │   └── G3_instruction.json
│   └── test_instruction/      # For test data
│       ├── G1_test_instruction.json
│       ├── G2_test_instruction.json
│       └── G3_test_instruction.json
```

### Python Dependencies

- Python 3.11+
- PyTorch 1.12+
- Transformers 4.20+
- Sentence Transformers 2.2+
- FAISS-CPU or FAISS-GPU
- NumPy, Pandas, tqdm

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd librarian
```

2. **Install dependencies**:

**Option 1: Full installation (recommended for development)**
```bash
pip install -r requirements.txt
```

**Option 2: Minimal installation (for basic usage)**
```bash
pip install -r requirements-minimal.txt
```

3. **Set up ToolBench data** (see Prerequisites section above)

## 📊 Data Preparation

### Training Dataset Preparation

The system uses ToolBench's instruction data to build training datasets for query embedding models.

#### Step 1: Build Training Dataset
```bash
python build_training_dataset.py
```

This script:
- Loads ToolBench instruction data from `../ToolBench/data/instruction/`
- Processes G1, G2, G3 query files
- Extracts relevant and irrelevant APIs for each query
- Creates training dataset with query-label vector pairs
- Saves processed data to `training_data/` directory

#### Step 2: Dataset Format
```python
# Each sample format
{
    "query_text": "user query text",
    "label_vector": [0.1, 0.2, 0.3, ...],  # 768-dimensional vector
    "query_id": "unique identifier"
}
```

### Vector Database Construction

#### Basic Vector Database
```bash
python build_vector_db.py
```
This creates multiple vector databases:
- `vector_db_base/` - 2,024 APIs

#### Multiple Vector Databases (Different Sizes)
```bash
python build_multiple_vector_dbs.py
```

This creates multiple vector databases:
- `vector_db_5k/` - 5,000 APIs
- `vector_db_7k5/` - 7,500 APIs  
- `vector_db_10k/` - 10,000 APIs
- `vector_db_all/` - 13,865 APIs (All available APIs)

## 🎯 Model Training

### Dynamic Query Embedding Model

The system features an advanced query embedding model with dynamic scale prediction.

#### Training Configuration
```bash
python train_query_embedding.py \
    --batch_size 32 \
    --num_epochs 15 \
    --learning_rate 2e-5 \
    --direction_weight 2.0 \
    --magnitude_weight 1.0
```

#### Model Architecture
- **Base Model**: ToolBench/ToolBench_IR_bert_based_uncased
- **Dynamic Scale Predictor**: Multi-layer network for query-specific scaling
- **Loss Function**: Combined direction and magnitude loss
- **Training Strategy**: Dynamic direction-focused learning

### Use Pretrained Model

If you prefer not to train from scratch, you can use the pretrained dynamic query embedding model from HuggingFace:

- [jhleepidl/librarian](https://huggingface.co/jhleepidl/librarian)


## 📈 Performance Analysis

### Performance Analysis (Use locally trained model)
```bash
python performance_analysis_local.py
```

### Performance Analysis (Use pretrained model from HuggingFace)
```bash
python performance_analysis_hf.py
```

### Beam Search Analysis (Use pretrained model from HuggingFace)
```bash
python beam_size_analysis.py
```

### Database Size Comparison (Use pretrained model from HuggingFace)
```bash
python compare_multiple_vector_dbs.py
```

This script compares performance across different database sizes and provides detailed analysis.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [ToolBench](https://github.com/OpenBMB/ToolBench) dataset and retriever model
- Uses [Sentence Transformers](https://www.sbert.net) for semantic similarity
- Vector search powered by [FAISS](https://github.com/facebookresearch/faiss)

## 📞 Support

For questions and support:
- Open an issue on GitHub