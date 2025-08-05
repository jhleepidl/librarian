import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import pickle
from collections import defaultdict
import random

class MultipleVectorDBBuilder:
    def __init__(self, model_path=None):
        """
        Multiple Vector DB builder for different sizes
        
        Args:
            model_path: Pre-trained model path (uses default model if None)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path and os.path.exists(model_path):
            print(f"Loading custom model from {model_path}")
            self.model = SentenceTransformer(model_path).to(self.device)
        else:
            print("Using ToolBench's trained retriever model")
            self.model = SentenceTransformer('ToolBench/ToolBench_IR_bert_based_uncased').to(self.device)
        
        self.documents = []
        self.embeddings = None
        self.index = None
        
    def load_test_instruction_data(self, data_dir):
        """
        Load JSON files from test_instruction folder
        
        Args:
            data_dir: test_instruction folder path
        """
        print("Loading test_instruction data...")
        
        # Load JSON files
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
                            'template_response': api.get('template_response', {}),
                            'source': 'test_instruction'  # Source indicator
                        }
                        all_apis.append(api_info)
        
        print(f"Loaded {len(all_apis)} APIs from test_instruction data")
        return all_apis
    
    def load_query_data(self, data_dir):
        """
        Load query JSON files from instruction folder
        
        Args:
            data_dir: instruction folder path
        """
        print("Loading query data...")
        
        # Load JSON files
        json_files = [
            'G1_query.json',
            'G2_query.json', 
            'G3_query.json'
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
                            'template_response': api.get('template_response', {}),
                            'source': 'query'  # Source indicator
                        }
                        all_apis.append(api_info)
        
        print(f"Loaded {len(all_apis)} APIs from query data")
        return all_apis
    
    def combine_and_sample_apis(self, test_apis, query_apis, target_size=None, use_query_only=False):
        """
        Combine test_instruction APIs with query APIs, or use query data only
        
        Args:
            test_apis: API list loaded from test_instruction
            query_apis: API list loaded from query
            target_size: Target number of unique APIs (None for all unique APIs)
            use_query_only: Whether to use query data only
            
        Returns:
            Combined and sampled API list
        """
        if use_query_only:
            print("Using query data only for vector_db_all...")
            
            # Extract unique APIs from query data
            query_api_dict = {}
            
            for api in query_apis:
                api_id = f"{api['tool_name']}:{api['api_name']}"
                query_api_dict[api_id] = api
            
            final_apis = list(query_api_dict.values())
            
            print(f"Found {len(final_apis)} unique APIs in query data")
            print(f"Final API count: {len(final_apis)}")
            
            # Save statistics
            stats = {
                'test_instruction_apis': 0,
                'query_only_apis': len(final_apis),
                'total_available': len(final_apis),
                'final_count': len(final_apis),
                'target_size': target_size,
                'mode': 'query_only'
            }
            
            with open('api_combination_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            print("Saved API combination statistics to api_combination_stats.json")
            
            return final_apis
        
        print("Combining and sampling APIs...")
        
        # 1. Combine all APIs into one pool and ensure uniqueness
        all_api_dict = {}
        
        # Add test_instruction APIs first (priority)
        for api in test_apis:
            api_id = f"{api['tool_name']}:{api['api_name']}"
            all_api_dict[api_id] = api
        
        print(f"Added {len(all_api_dict)} unique APIs from test_instruction")
        
        # Add query APIs (non-duplicates only)
        query_added_count = 0
        for api in query_apis:
            api_id = f"{api['tool_name']}:{api['api_name']}"
            if api_id not in all_api_dict:
                all_api_dict[api_id] = api
                query_added_count += 1
        
        print(f"Added {query_added_count} additional unique APIs from query data")
        print(f"Total unique APIs available: {len(all_api_dict)}")
        
        # 2. Sample according to target size
        all_apis_list = list(all_api_dict.values())
        
        if target_size:
            if len(all_apis_list) <= target_size:
                print(f"Available APIs ({len(all_apis_list)}) <= target size ({target_size}), using all APIs")
                final_apis = all_apis_list
            else:
                print(f"Sampling {target_size} APIs from {len(all_apis_list)} available APIs")
                # Ensure test_instruction APIs are prioritized
                test_api_ids = {f"{api['tool_name']}:{api['api_name']}" for api in test_apis}
                
                # Select test_instruction APIs first
                selected_test_apis = []
                selected_query_apis = []
                
                for api in all_apis_list:
                    api_id = f"{api['tool_name']}:{api['api_name']}"
                    if api_id in test_api_ids:
                        selected_test_apis.append(api)
                    else:
                        selected_query_apis.append(api)
                
                # If test_instruction API count exceeds target_size
                if len(selected_test_apis) > target_size:
                    print(f"Warning: test_instruction APIs ({len(selected_test_apis)}) exceed target size ({target_size})")
                    print("Randomly sampling from test_instruction APIs")
                    random.seed(42)
                    final_apis = random.sample(selected_test_apis, target_size)
                else:
                    # Include all test_instruction APIs, sample remaining from query
                    remaining_slots = target_size - len(selected_test_apis)
                    if remaining_slots > 0:
                        if len(selected_query_apis) > remaining_slots:
                            random.seed(42)
                            sampled_query_apis = random.sample(selected_query_apis, remaining_slots)
                            final_apis = selected_test_apis + sampled_query_apis
                        else:
                            final_apis = selected_test_apis + selected_query_apis
                    else:
                        final_apis = selected_test_apis
        else:
            # Use all APIs
            final_apis = all_apis_list
        
        print(f"Final API count: {len(final_apis)}")
        
        # Calculate statistics
        final_test_count = sum(1 for api in final_apis if api.get('source') == 'test_instruction')
        final_query_count = sum(1 for api in final_apis if api.get('source') == 'query')
        
        # Save statistics
        stats = {
            'test_instruction_apis_in_final': final_test_count,
            'query_apis_in_final': final_query_count,
            'total_available': len(all_api_dict),
            'final_count': len(final_apis),
            'target_size': target_size,
            'mode': 'combined'
        }
        
        with open('api_combination_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print("Saved API combination statistics to api_combination_stats.json")
        
        return final_apis
    
    def prepare_documents(self, apis):
        """
        Convert API information to document format
        
        Args:
            apis: API information list
        """
        print("Preparing documents for embedding...")
        
        self.documents = []
        
        for i, api in enumerate(apis):
            # Construct document text using ToolBench's process_retrieval_document method
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
                'source': api.get('source', 'unknown'),  # Include source information
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
    
    def save_index(self, output_dir, mode_name):
        """
        Save index and metadata
        
        Args:
            output_dir: Output directory
            mode_name: Mode name
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
            'mode': mode_name
        }
        
        model_info_path = os.path.join(output_dir, 'model_info.json')
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"Saved model info to {model_info_path}")
    
    def check_existing_vector_db(self, output_dir):
        """
        Check if vector DB already exists
        
        Args:
            output_dir: Directory to check
            
        Returns:
            bool: True if exists, False otherwise
        """
        required_files = ['faiss_index.bin', 'metadata.pkl', 'model_info.json']
        
        for file_name in required_files:
            file_path = os.path.join(output_dir, file_name)
            if not os.path.exists(file_path):
                return False
        
        return True
    
    def build_vector_db(self, test_data_dir, query_data_dir, output_dir, target_size=None, mode_name="combined_full", use_query_only=False):
        """
        Vector DB construction process
        
        Args:
            test_data_dir: test_instruction folder path
            query_data_dir: instruction folder path
            output_dir: output directory
            target_size: target number of unique APIs (None for all unique APIs)
            mode_name: mode name
            use_query_only: whether to use query data only
        """
        # Check if vector DB already exists
        if self.check_existing_vector_db(output_dir):
            print(f"Vector DB already exists at {output_dir}, skipping...")
            
            # Read existing information
            try:
                with open(os.path.join(output_dir, 'model_info.json'), 'r') as f:
                    model_info = json.load(f)
                num_docs = model_info.get('num_documents', 0)
                print(f"Existing vector DB has {num_docs} documents")
                return num_docs
            except:
                print("Could not read existing vector DB info")
                return 0
        
        print(f"Starting Vector DB construction for {mode_name}...")
        
        if use_query_only:
            # Use query data only
            query_apis = self.load_query_data(query_data_dir)
            combined_apis = self.combine_and_sample_apis([], query_apis, target_size, use_query_only=True)
        else:
            # 1. Load test_instruction data
            test_apis = self.load_test_instruction_data(test_data_dir)
            
            # 2. Load query data
            query_apis = self.load_query_data(query_data_dir)
            
            # 3. Combine and sample APIs
            combined_apis = self.combine_and_sample_apis(test_apis, query_apis, target_size)
        
        # 4. Prepare documents
        self.prepare_documents(combined_apis)
        
        # 5. Create embeddings
        self.create_embeddings()
        
        # 6. Build Faiss index (cosine similarity)
        self.build_faiss_index('cosine')
        
        # 7. Save
        self.save_index(output_dir, mode_name)
        
        print(f"{mode_name} Vector DB construction completed!")
        return len(self.documents)

def main():
    # Configuration
    test_data_dir = "../ToolBench/data/test_instruction"
    query_data_dir = "../ToolBench/data/instruction"
    
    # Create 4 different sized vector DBs
    configs = [
        {"target_size": 5000, "output_dir": "vector_db_5k", "mode_name": "combined_5k", "use_query_only": False},
        {"target_size": 7500, "output_dir": "vector_db_7k5", "mode_name": "combined_7k5", "use_query_only": False},
        {"target_size": 10000, "output_dir": "vector_db_10k", "mode_name": "combined_10k", "use_query_only": False},
        {"target_size": None, "output_dir": "vector_db_all", "mode_name": "query_all", "use_query_only": True}
    ]
    
    builder = MultipleVectorDBBuilder()
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Building {config['mode_name']} Vector DB")
        print(f"{'='*60}")
        
        target_size = config['target_size']
        output_dir = config['output_dir']
        mode_name = config['mode_name']
        use_query_only = config['use_query_only']
        
        if use_query_only:
            print(f"Target: All unique APIs from query data only")
        elif target_size:
            print(f"Target: {target_size} unique APIs (including all test_instruction APIs)")
        else:
            print(f"Target: All unique APIs (including all test_instruction APIs)")
        
        try:
            num_docs = builder.build_vector_db(test_data_dir, query_data_dir, output_dir, target_size, mode_name, use_query_only)
            print(f"\n{config['mode_name']} Vector DB has been built successfully!")
            print(f"Output directory: {output_dir}")
            print(f"Number of documents: {num_docs}")
            print(f"Embedding dimension: {builder.embeddings.shape[1]}")
        except Exception as e:
            print(f"Error building {config['mode_name']} Vector DB: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("All Vector DBs construction completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 