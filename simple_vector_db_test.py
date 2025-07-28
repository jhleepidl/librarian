#!/usr/bin/env python3
"""
Simple vector database test without sentence_transformers
"""

import sys
import os
import json
import numpy as np
import faiss
import pickle

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_TOOLBENCH_PATH = os.path.join(BASE_DIR, "vector_db_toolbench")
TOOLBENCH_DATABASE_PATH = os.path.join(BASE_DIR, "toolbench_database")
TEST_INSTRUCTION_DIR = "/home/jhlee/ToolBench/data/test_instruction"


class SimpleVectorDBTester:
    """Simple vector database test"""
    
    def __init__(self):
        self.vector_db_toolbench = None
        self.api_metadata_toolbench = []
        self.toolbench_database = None
        self.api_info_toolbench_db = []
        
    def load_dbs(self):
        """Load vector databases"""
        print("📦 Loading vector databases...")
        
        # Load vector_db_toolbench
        try:
            faiss_index_path = os.path.join(VECTOR_DB_TOOLBENCH_PATH, "api_embeddings_toolbench.index")
            self.vector_db_toolbench = faiss.read_index(faiss_index_path)
            print(f"✅ Loaded vector_db_toolbench with {self.vector_db_toolbench.ntotal} vectors")
            
            metadata_path = os.path.join(VECTOR_DB_TOOLBENCH_PATH, "api_metadata_toolbench.pkl")
            with open(metadata_path, 'rb') as f:
                self.api_metadata_toolbench = pickle.load(f)
            print(f"✅ Loaded metadata for {len(self.api_metadata_toolbench)} APIs")
        except Exception as e:
            print(f"❌ Error loading vector_db_toolbench: {e}")
            return False
        
        # Load toolbench_database
        try:
            faiss_index_path = os.path.join(TOOLBENCH_DATABASE_PATH, "faiss_index.bin")
            self.toolbench_database = faiss.read_index(faiss_index_path)
            print(f"✅ Loaded toolbench_database with {self.toolbench_database.ntotal} vectors")
            
            api_info_path = os.path.join(TOOLBENCH_DATABASE_PATH, "api_info.json")
            with open(api_info_path, 'r', encoding='utf-8') as f:
                self.api_info_toolbench_db = json.load(f)
            print(f"✅ Loaded api_info for {len(self.api_info_toolbench_db)} APIs")
        except Exception as e:
            print(f"❌ Error loading toolbench_database: {e}")
            return False
        
        return True
    
    def load_test_apis(self, num_samples: int = 3) -> list:
        """Load test APIs from instruction files"""
        print(f"📦 Loading {num_samples} test APIs from instruction files...")
        
        test_apis = []
        
        # Load from G1_instruction.json
        instruction_file = os.path.join(TEST_INSTRUCTION_DIR, "G1_instruction.json")
        try:
            with open(instruction_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract APIs from first few samples
            for i, sample in enumerate(data[:num_samples]):
                api_list = sample.get("api_list", [])
                for api in api_list[:1]:  # Take first API from each sample
                    test_apis.append({
                        'tool_name': api.get('tool_name', ''),
                        'api_name': api.get('api_name', ''),
                        'api_description': api.get('api_description', ''),
                        'category_name': api.get('category_name', ''),
                        'method': api.get('method', ''),
                        'sample_id': i
                    })
                    if len(test_apis) >= num_samples:
                        break
                if len(test_apis) >= num_samples:
                    break
                    
        except Exception as e:
            print(f"❌ Error loading test APIs: {e}")
            return []
        
        print(f"✅ Loaded {len(test_apis)} test APIs")
        return test_apis
    
    def find_api_in_metadata(self, tool_name: str, api_name: str, metadata_list: list) -> dict:
        """Find API in metadata list"""
        for api in metadata_list:
            if api.get('tool_name') == tool_name and api.get('api_name') == api_name:
                return api
        return None
    
    def test_api_presence(self, test_apis: list):
        """Test if APIs are present in vector databases"""
        print("\n🔍 Testing API presence in vector databases...")
        print("=" * 80)
        
        # Test vector_db_toolbench
        print(f"\n📊 Testing vector_db_toolbench ({len(self.api_metadata_toolbench)} APIs):")
        found_in_toolbench = 0
        
        for i, api in enumerate(test_apis):
            found = self.find_api_in_metadata(api['tool_name'], api['api_name'], self.api_metadata_toolbench)
            if found:
                found_in_toolbench += 1
                print(f"   ✅ {api['tool_name']} - {api['api_name']} (FOUND)")
            else:
                print(f"   ❌ {api['tool_name']} - {api['api_name']} (NOT FOUND)")
        
        # Test toolbench_database
        print(f"\n📊 Testing toolbench_database ({len(self.api_info_toolbench_db)} APIs):")
        found_in_toolbench_db = 0
        
        for i, api in enumerate(test_apis):
            found = self.find_api_in_metadata(api['tool_name'], api['api_name'], self.api_info_toolbench_db)
            if found:
                found_in_toolbench_db += 1
                print(f"   ✅ {api['tool_name']} - {api['api_name']} (FOUND)")
            else:
                print(f"   ❌ {api['tool_name']} - {api['api_name']} (NOT FOUND)")
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"   Total APIs tested: {len(test_apis)}")
        print(f"   Found in vector_db_toolbench: {found_in_toolbench}/{len(test_apis)}")
        print(f"   Found in toolbench_database: {found_in_toolbench_db}/{len(test_apis)}")
        
        if found_in_toolbench == len(test_apis):
            print("✅ vector_db_toolbench: ALL APIs PRESENT")
        else:
            print(f"⚠️  vector_db_toolbench: {len(test_apis) - found_in_toolbench} APIs MISSING")
            
        if found_in_toolbench_db == len(test_apis):
            print("✅ toolbench_database: ALL APIs PRESENT")
        else:
            print(f"⚠️  toolbench_database: {len(test_apis) - found_in_toolbench_db} APIs MISSING")
    
    def analyze_db_contents(self):
        """Analyze database contents"""
        print("\n📊 Database Contents Analysis:")
        print("=" * 50)
        
        # Analyze vector_db_toolbench
        print(f"\n🔍 vector_db_toolbench analysis:")
        print(f"   Total APIs: {len(self.api_metadata_toolbench)}")
        
        # Count unique tools
        unique_tools = set()
        for api in self.api_metadata_toolbench:
            unique_tools.add(api.get('tool_name', ''))
        print(f"   Unique tools: {len(unique_tools)}")
        
        # Show first few tools
        tool_list = list(unique_tools)[:10]
        print(f"   Sample tools: {tool_list}")
        
        # Analyze toolbench_database
        print(f"\n🔍 toolbench_database analysis:")
        print(f"   Total APIs: {len(self.api_info_toolbench_db)}")
        
        # Count unique tools
        unique_tools_db = set()
        for api in self.api_info_toolbench_db:
            unique_tools_db.add(api.get('tool_name', ''))
        print(f"   Unique tools: {len(unique_tools_db)}")
        
        # Show first few tools
        tool_list_db = list(unique_tools_db)[:10]
        print(f"   Sample tools: {tool_list_db}")
    
    def run_test(self):
        """Run the complete test"""
        print("🚀 Starting Simple Vector DB Test")
        print("=" * 50)
        
        # Load databases
        if not self.load_dbs():
            print("❌ Failed to load databases")
            return
        
        # Analyze database contents
        self.analyze_db_contents()
        
        # Load test APIs
        test_apis = self.load_test_apis(num_samples=3)
        
        if not test_apis:
            print("❌ No test APIs found")
            return
        
        # Test API presence
        self.test_api_presence(test_apis)
        
        print("\n✅ Simple vector DB test completed!")


def main():
    """Main test function"""
    tester = SimpleVectorDBTester()
    tester.run_test()


if __name__ == "__main__":
    main() 