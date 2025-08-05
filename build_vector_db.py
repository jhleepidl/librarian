import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import pickle
from collections import defaultdict

class DeduplicatedVectorDBBuilder:
    def __init__(self, model_path=None):
        """
        Class for building deduplicated Vector DB (ToolBench retriever approach)
        
        Args:
            model_path: Pre-trained model path (uses default model if None)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path and os.path.exists(model_path):
            print(f"Loading custom model from {model_path}")
            self.model = SentenceTransformer(model_path).to(self.device)
        else:
            print("Using ToolBench's trained retriever model")
            # Use ToolBench's trained retriever model
            self.model = SentenceTransformer('ToolBench/ToolBench_IR_bert_based_uncased').to(self.device)
        
        self.documents = []
        self.embeddings = None
        self.index = None
        
    def load_test_data(self, data_dir):
        """
        Load JSON files from test_instruction folder
        
        Args:
            data_dir: test_instruction folder path
        """
        print("Loading test data...")
        
        # Load JSON files (instruction files only)
        json_files = [
            'G1_instruction.json',
            'G2_instruction.json', 
            'G3_instruction.json'
        ]
        
        all_apis = []
        
        for file_name in json_files:
            file_path = os.path.join(data_dir, file_name)
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, skipping...")
                continue
                
            print(f"Loading {file_name}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract API information
            for item in data:
                if 'api_list' in item:
                    for api in item['api_list']:
                        # Construct API info
                        api_info = {
                            'tool_name': api.get('tool_name', ''),
                            'api_name': api.get('api_name', ''),
                            'api_description': api.get('api_description', ''),
                            'category_name': api.get('category_name', ''),
                            'method': api.get('method', ''),
                            'required_parameters': api.get('required_parameters', []),
                            'optional_parameters': api.get('optional_parameters', []),
                            'template_response': api.get('template_response', {})
                        }
                        all_apis.append(api_info)
        
        print(f"Loaded {len(all_apis)} APIs from test data")
        return all_apis
    
    def remove_duplicates(self, apis):
        """
        Remove duplicate APIs
        
        Args:
            apis: API information list
            
        Returns:
            Deduplicated API list
        """
        print("Removing duplicate APIs...")
        
        # Group APIs by unique identifier
        api_groups = defaultdict(list)
        
        for api in apis:
            # Unique identifier: tool_name + api_name
            api_id = f"{api['tool_name']}:{api['api_name']}"
            api_groups[api_id].append(api)
        
        print(f"Found {len(api_groups)} unique API groups")
        
        # Create deduplicated API list
        deduplicated_apis = []
        duplicate_stats = {
            'total_original': len(apis),
            'total_unique': len(api_groups),
            'duplicates_removed': len(apis) - len(api_groups),
            'duplicate_groups': []
        }
        
        for api_id, api_list in api_groups.items():
            if len(api_list) > 1:
                # If duplicate APIs exist, select the first one
                selected_api = api_list[0]
                duplicate_stats['duplicate_groups'].append({
                    'api_id': api_id,
                    'count': len(api_list),
                    'categories': [api['category_name'] for api in api_list],
                    'methods': [api['method'] for api in api_list]
                })
                print(f"  Duplicate found: {api_id} ({len(api_list)} instances) - keeping first one")
            else:
                selected_api = api_list[0]
            
            deduplicated_apis.append(selected_api)
        
        print(f"Removed {duplicate_stats['duplicates_removed']} duplicate APIs")
        print(f"Final unique API count: {len(deduplicated_apis)}")
        
        # Save duplicate statistics
        with open('duplicate_removal_stats.json', 'w') as f:
            json.dump(duplicate_stats, f, indent=2)
        print("Saved duplicate removal statistics to duplicate_removal_stats.json")
        
        return deduplicated_apis
    
    def prepare_documents(self, apis):
        """
        Convert API information to document format (ToolBench retriever approach)
        
        Args:
            apis: API information list
        """
        print("Preparing documents for embedding (deduplicated mode: all information)...")
        
        self.documents = []
        
        for i, api in enumerate(apis):
            # Construct document text using ToolBench's process_retrieval_document approach
            doc_text = f"{api.get('category_name', '') or ''}, {api.get('tool_name', '') or ''}, {api.get('api_name', '') or ''}, {api.get('api_description', '') or ''}"
            doc_text += f", required_params: {json.dumps(api.get('required_parameters', ''))}"
            doc_text += f", optional_params: {json.dumps(api.get('optional_parameters', ''))}"
            doc_text += f", return_schema: {json.dumps(api.get('template_response', ''))}"
            
            # Construct metadata
            metadata = {
                'doc_id': i,
                'tool_name': api['tool_name'],
                'api_name': api['api_name'],
                'category_name': api['category_name'],
                'method': api['method'],
                'original_api': api
            }
            
            self.documents.append({
                'text': doc_text,
                'metadata': metadata
            })
        
        print(f"Prepared {len(self.documents)} documents")
    
    def create_embeddings(self):
        """
        Convert documents to embeddings
        """
        print("Creating embeddings...")
        
        texts = [doc['text'] for doc in self.documents]
        
        # Create embeddings in batches
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_tensor=True)
            embeddings.append(batch_embeddings.cpu().numpy())
        
        self.embeddings = np.vstack(embeddings)
        print(f"Created embeddings with shape: {self.embeddings.shape}")
    
    def build_faiss_index(self, index_type='cosine'):
        """
        Build Faiss index
        
        Args:
            index_type: Index type ('cosine' or 'l2')
        """
        print(f"Building Faiss index with {index_type} similarity...")
        
        dimension = self.embeddings.shape[1]
        
        if index_type == 'cosine':
            # Normalize for cosine similarity
            faiss.normalize_L2(self.embeddings)
            # Cosine similarity equals dot product (for normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)
        else:
            # L2 distance
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(self.embeddings.astype('float32'))
        print(f"Built index with {self.index.ntotal} vectors")
    
    def save_index(self, output_dir):
        """
        Save index and metadata
        
        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Faiss index
        index_path = os.path.join(output_dir, 'faiss_index.bin')
        faiss.write_index(self.index, index_path)
        print(f"Saved Faiss index to {index_path}")
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.documents, f)
        print(f"Saved metadata to {metadata_path}")
        
        # Save model info
        model_info = {
            'model_name': self.model.get_sentence_embedding_dimension(),
            'embedding_dimension': self.embeddings.shape[1],
            'num_documents': len(self.documents),
            'mode': 'deduplicated_full'  # Indicates deduplicated full information mode
        }
        
        model_info_path = os.path.join(output_dir, 'model_info.json')
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"Saved model info to {model_info_path}")
    
    def build_vector_db(self, data_dir, output_dir, model_path=None):
        """
        Deduplicated Vector DB construction process
        
        Args:
            data_dir: test_instruction folder path
            output_dir: output directory
            model_path: model path (optional)
        """
        print("Starting Deduplicated Vector DB construction...")
        
        # 1. Load data
        apis = self.load_test_data(data_dir)
        
        # 2. Remove duplicates
        deduplicated_apis = self.remove_duplicates(apis)
        
        # 3. Prepare documents (use all information)
        self.prepare_documents(deduplicated_apis)
        
        # 4. Create embeddings
        self.create_embeddings()
        
        # 5. Build Faiss index (cosine similarity)
        self.build_faiss_index('cosine')
        
        # 6. Save
        self.save_index(output_dir)
        
        print("Deduplicated Vector DB construction completed!")

def main():
    # Configuration
    data_dir = "../ToolBench/data/test_instruction"
    output_dir = "vector_db_base"
    
    # Build Vector DB
    builder = DeduplicatedVectorDBBuilder()
    builder.build_vector_db(data_dir, output_dir)
    
    print(f"\nDeduplicated Vector DB has been built successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Number of documents: {len(builder.documents)}")
    print(f"Embedding dimension: {builder.embeddings.shape[1]}")
    print(f"Mode: deduplicated full information (category, tool, api, description, parameters, schema)")

if __name__ == "__main__":
    main() 