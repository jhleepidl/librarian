import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

class DatasetAnalyzer:
    def __init__(self, model_path="ToolBench/ToolBench_IR_bert_based_uncased"):
        """데이터셋 분석을 위한 클래스 초기화"""
        self.model = SentenceTransformer(model_path)
        self.api_embeddings = {}
        self.api_to_text = {}
        
    def extract_api_text(self, api):
        """API 정보를 ToolBench inference 방식으로 텍스트로 변환"""
        # ToolBench inference에서는 tool_name과 api_name만 사용
        tool_name = api.get('tool_name', '') or ''
        api_name = api.get('api_name', '') or ''
        
        # 공백으로 연결 (ToolBench inference 방식과 동일)
        text = f"{tool_name} {api_name}"
        
        return text
    
    def get_api_embedding(self, api):
        """API의 임베딩을 가져오거나 생성"""
        api_text = self.extract_api_text(api)
        
        # API 식별자 생성 (tool_name + api_name)
        api_id = f"{api.get('tool_name', '')}_{api.get('api_name', '')}"
        
        if api_id not in self.api_embeddings:
            self.api_embeddings[api_id] = self.model.encode(api_text)
            self.api_to_text[api_id] = api_text
            
        return api_id, self.api_embeddings[api_id]
    
    def analyze_file(self, file_path, global_api_pool):
        """단일 파일 분석"""
        print(f"분석 중: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = {
            'total_queries': len(data),
            'queries_analyzed': 0,
            'avg_relevant_apis_per_query': 0,
            'similarity_stats': [],
            'clustering_results': [],
            'nearest_api_analysis': [],
            'api_embeddings': {}
        }
        
        all_relevant_apis = []
        all_apis_in_file = []  # 파일 내 모든 API 수집
        
        # 먼저 파일 내 모든 API의 임베딩을 수집
        for query_data in data:
            for api in query_data.get('api_list', []):
                api_id, embedding = self.get_api_embedding(api)
                all_apis_in_file.append((api_id, embedding))
        
        # 중복 제거
        unique_apis = {}
        for api_id, embedding in all_apis_in_file:
            if api_id not in unique_apis:
                unique_apis[api_id] = embedding
        
        for query_data in tqdm(data, desc="쿼리 분석 중"):
            if 'relevant APIs' not in query_data or not query_data['relevant APIs']:
                continue
                
            relevant_apis = query_data['relevant APIs']
            results['queries_analyzed'] += 1
            
            # 각 relevant API의 임베딩 생성
            api_embeddings = []
            api_ids = []
            
            for api_ref in relevant_apis:
                # api_ref는 [tool_name, api_name] 형태
                if len(api_ref) != 2:
                    continue
                    
                tool_name, api_name = api_ref
                
                # api_list에서 해당 API 찾기
                target_api = None
                for api in query_data.get('api_list', []):
                    if api.get('tool_name') == tool_name and api.get('api_name') == api_name:
                        target_api = api
                        break
                
                if target_api is None:
                    continue
                    
                api_id, embedding = self.get_api_embedding(target_api)
                api_embeddings.append(embedding)
                api_ids.append(api_id)
                all_relevant_apis.append(api_id)
            
            if len(api_embeddings) < 2:
                continue
                
            # 임베딩을 numpy 배열로 변환
            embeddings_array = np.array(api_embeddings)
            
            # 코사인 유사도 계산
            similarity_matrix = cosine_similarity(embeddings_array)
            
            # 대각선을 제외한 유사도만 계산 (자기 자신과의 유사도 제외)
            # 상삼각 행렬에서 대각선을 제외한 값들만 추출
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            # 평균 유사도 계산
            avg_similarity = np.mean(upper_triangle)
            min_similarity = np.min(upper_triangle)
            max_similarity = np.max(upper_triangle)
            
            results['similarity_stats'].append({
                'query_id': query_data.get('query_id', 'unknown'),
                'num_apis': len(api_embeddings),
                'avg_similarity': avg_similarity,
                'min_similarity': min_similarity,
                'max_similarity': max_similarity,
                'api_ids': api_ids
            })
            
            # Clustering 분석
            if len(api_embeddings) >= 3:
                # K-means 클러스터링 (클러스터 수는 API 개수의 절반으로 설정)
                n_clusters = min(len(api_embeddings) // 2, len(api_embeddings) - 1)
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(embeddings_array)
                    
                    # 클러스터 내 평균 거리 계산
                    cluster_distances = []
                    for i in range(n_clusters):
                        cluster_indices = np.where(cluster_labels == i)[0]
                        if len(cluster_indices) > 1:
                            cluster_embeddings = embeddings_array[cluster_indices]
                            cluster_similarity = cosine_similarity(cluster_embeddings)
                            np.fill_diagonal(cluster_similarity, 0)
                            cluster_distances.append(np.mean(cluster_similarity))
                    
                    results['clustering_results'].append({
                        'query_id': query_data.get('query_id', 'unknown'),
                        'num_clusters': n_clusters,
                        'cluster_labels': cluster_labels.tolist(),
                        'avg_cluster_similarity': np.mean(cluster_distances) if cluster_distances else 0,
                        'api_ids': api_ids
                    })
            
            # Relevant API들의 평균 임베딩과 가장 가까운 API 찾기
            if len(api_embeddings) >= 1:
                # Relevant API들의 평균 임베딩 계산
                relevant_avg_embedding = np.mean(embeddings_array, axis=0)
                
                # 모든 API와의 유사도 계산
                similarities_with_all = []
                for api_id, api_embedding in global_api_pool.items():
                    similarity = cosine_similarity([relevant_avg_embedding], [api_embedding])[0][0]
                    similarities_with_all.append((api_id, similarity))
                
                # 유사도 순으로 정렬
                similarities_with_all.sort(key=lambda x: x[1], reverse=True)
                
                # Top 5 API와 relevant API들의 유사도 분석
                top_5_apis = similarities_with_all[:5]
                relevant_api_similarities = []
                
                for relevant_api_id in api_ids:
                    for api_id, similarity in similarities_with_all:
                        if api_id == relevant_api_id:
                            relevant_api_similarities.append(similarity)
                            break
                
                # Relevant API들의 평균 유사도
                avg_relevant_similarity = np.mean(relevant_api_similarities) if relevant_api_similarities else 0
                
                # Top 1 API가 relevant API인지 확인
                top_1_is_relevant = similarities_with_all[0][0] in api_ids
                
                # Top 5 중 relevant API 개수
                top_5_relevant_count = sum(1 for api_id, _ in top_5_apis if api_id in api_ids)
                
                results['nearest_api_analysis'].append({
                    'query_id': query_data.get('query_id', 'unknown'),
                    'num_relevant_apis': len(api_embeddings),
                    'top_5_apis': top_5_apis,
                    'avg_relevant_similarity': avg_relevant_similarity,
                    'top_1_is_relevant': top_1_is_relevant,
                    'top_5_relevant_count': top_5_relevant_count,
                    'relevant_api_ids': api_ids
                })
        
        # 전체 통계 계산
        if results['similarity_stats']:
            results['avg_relevant_apis_per_query'] = np.mean([stat['num_apis'] for stat in results['similarity_stats']])
            results['avg_similarity_across_queries'] = np.mean([stat['avg_similarity'] for stat in results['similarity_stats']])
            results['min_similarity_across_queries'] = np.min([stat['min_similarity'] for stat in results['similarity_stats']])
            results['max_similarity_across_queries'] = np.max([stat['max_similarity'] for stat in results['similarity_stats']])
        
        return results
    
    def analyze_all_files(self, data_dir):
        """모든 파일 분석"""
        files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        all_results = {}
        
        # 먼저 모든 파일에서 API들을 수집하여 전역 API 풀 생성
        print("전역 API 풀을 생성하는 중...")
        global_api_pool = {}
        
        for file in tqdm(files, desc="전역 API 풀 생성"):
            file_path = os.path.join(data_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for query_data in data:
                for api in query_data.get('api_list', []):
                    api_id, embedding = self.get_api_embedding(api)
                    if api_id not in global_api_pool:
                        global_api_pool[api_id] = embedding
        
        print(f"전역 API 풀 크기: {len(global_api_pool)}")
        
        for file in files:
            file_path = os.path.join(data_dir, file)
            results = self.analyze_file(file_path, global_api_pool)
            all_results[file] = results
        
        # 전역 API 풀 크기를 결과에 추가
        all_results['_global_api_pool_size'] = len(global_api_pool)
            
        return all_results
    
    def analyze_partial_file(self, file_path, max_queries=100, global_api_pool=None):
        """대용량 파일의 일부만 분석"""
        print(f"분석 중: {file_path} (최대 {max_queries}개 쿼리)")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 일부만 사용
        if len(data) > max_queries:
            print(f"전체 {len(data)}개 중 {max_queries}개만 사용")
            data = data[:max_queries]
        
        results = {
            'total_queries': len(data),
            'queries_analyzed': 0,
            'avg_relevant_apis_per_query': 0,
            'similarity_stats': [],
            'clustering_results': [],
            'nearest_api_analysis': [],
            'api_embeddings': {}
        }
        
        all_relevant_apis = []
        all_apis_in_file = []  # 파일 내 모든 API 수집
        
        # 먼저 파일 내 모든 API의 임베딩을 수집
        for query_data in data:
            for api in query_data.get('relevant_apis', []):
                api_id, embedding = self.get_api_embedding(api)
                all_apis_in_file.append((api_id, embedding))
        
        # 중복 제거
        unique_apis = {}
        for api_id, embedding in all_apis_in_file:
            if api_id not in unique_apis:
                unique_apis[api_id] = embedding
        
        for query_data in tqdm(data, desc="쿼리 분석 중"):
            if 'relevant_apis' not in query_data or not query_data['relevant_apis']:
                continue
                
            relevant_apis = query_data['relevant_apis']
            results['queries_analyzed'] += 1
            
            # 각 relevant API의 임베딩 생성
            api_embeddings = []
            api_ids = []
            
            for api in relevant_apis:
                api_id, embedding = self.get_api_embedding(api)
                api_embeddings.append(embedding)
                api_ids.append(api_id)
                all_relevant_apis.append(api_id)
            
            if len(api_embeddings) < 2:
                continue
                
            # 임베딩을 numpy 배열로 변환
            embeddings_array = np.array(api_embeddings)
            
            # 코사인 유사도 계산
            similarity_matrix = cosine_similarity(embeddings_array)
            
            # 대각선을 제외한 유사도만 계산 (자기 자신과의 유사도 제외)
            # 상삼각 행렬에서 대각선을 제외한 값들만 추출
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            # 평균 유사도 계산
            avg_similarity = np.mean(upper_triangle)
            min_similarity = np.min(upper_triangle)
            max_similarity = np.max(upper_triangle)
            
            results['similarity_stats'].append({
                'query_id': query_data.get('query_id', 'unknown'),
                'num_apis': len(api_embeddings),
                'avg_similarity': avg_similarity,
                'min_similarity': min_similarity,
                'max_similarity': max_similarity,
                'api_ids': api_ids
            })
            
            # Clustering 분석
            if len(api_embeddings) >= 3:
                # K-means 클러스터링 (클러스터 수는 API 개수의 절반으로 설정)
                n_clusters = min(len(api_embeddings) // 2, len(api_embeddings) - 1)
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(embeddings_array)
                    
                    # 클러스터 내 평균 거리 계산
                    cluster_distances = []
                    for i in range(n_clusters):
                        cluster_indices = np.where(cluster_labels == i)[0]
                        if len(cluster_indices) > 1:
                            cluster_embeddings = embeddings_array[cluster_indices]
                            cluster_similarity = cosine_similarity(cluster_embeddings)
                            np.fill_diagonal(cluster_similarity, 0)
                            cluster_distances.append(np.mean(cluster_similarity))
                    
                    results['clustering_results'].append({
                        'query_id': query_data.get('query_id', 'unknown'),
                        'num_clusters': n_clusters,
                        'cluster_labels': cluster_labels.tolist(),
                        'avg_cluster_similarity': np.mean(cluster_distances) if cluster_distances else 0,
                        'api_ids': api_ids
                    })
            
            # Relevant API들의 평균 임베딩과 가장 가까운 API 찾기
            if len(api_embeddings) >= 1 and global_api_pool:
                # Relevant API들의 평균 임베딩 계산
                relevant_avg_embedding = np.mean(embeddings_array, axis=0)
                
                # 모든 API와의 유사도 계산
                similarities_with_all = []
                for api_id, api_embedding in global_api_pool.items():
                    similarity = cosine_similarity([relevant_avg_embedding], [api_embedding])[0][0]
                    similarities_with_all.append((api_id, similarity))
                
                # 유사도 순으로 정렬
                similarities_with_all.sort(key=lambda x: x[1], reverse=True)
                
                # Top 5 API와 relevant API들의 유사도 분석
                top_5_apis = similarities_with_all[:5]
                relevant_api_similarities = []
                
                for relevant_api_id in api_ids:
                    for api_id, similarity in similarities_with_all:
                        if api_id == relevant_api_id:
                            relevant_api_similarities.append(similarity)
                            break
                
                # Relevant API들의 평균 유사도
                avg_relevant_similarity = np.mean(relevant_api_similarities) if relevant_api_similarities else 0
                
                # Top 1 API가 relevant API인지 확인
                top_1_is_relevant = similarities_with_all[0][0] in api_ids
                
                # Top 5 중 relevant API 개수
                top_5_relevant_count = sum(1 for api_id, _ in top_5_apis if api_id in api_ids)
                
                results['nearest_api_analysis'].append({
                    'query_id': query_data.get('query_id', 'unknown'),
                    'num_relevant_apis': len(api_embeddings),
                    'top_5_apis': top_5_apis,
                    'avg_relevant_similarity': avg_relevant_similarity,
                    'top_1_is_relevant': top_1_is_relevant,
                    'top_5_relevant_count': top_5_relevant_count,
                    'relevant_api_ids': api_ids
                })
        
        # 전체 통계 계산
        if results['similarity_stats']:
            results['avg_relevant_apis_per_query'] = np.mean([stat['num_apis'] for stat in results['similarity_stats']])
            results['avg_similarity_across_queries'] = np.mean([stat['avg_similarity'] for stat in results['similarity_stats']])
            results['min_similarity_across_queries'] = np.min([stat['min_similarity'] for stat in results['similarity_stats']])
            results['max_similarity_across_queries'] = np.max([stat['max_similarity'] for stat in results['similarity_stats']])
        
        return results
    
    def analyze_partial_file_g2(self, file_path, max_queries=100, global_api_pool=None):
        """G2_query.json 형식의 대용량 파일 분석 (relevant APIs가 없는 경우)"""
        print(f"분석 중: {file_path} (최대 {max_queries}개 쿼리)")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 일부만 사용
        if len(data) > max_queries:
            print(f"전체 {len(data)}개 중 {max_queries}개만 사용")
            data = data[:max_queries]
        
        results = {
            'total_queries': len(data),
            'queries_analyzed': 0,
            'avg_relevant_apis_per_query': 0,
            'similarity_stats': [],
            'clustering_results': [],
            'nearest_api_analysis': [],
            'api_embeddings': {}
        }
        
        all_relevant_apis = []
        all_apis_in_file = []  # 파일 내 모든 API 수집
        
        # 먼저 파일 내 모든 API의 임베딩을 수집
        for query_data in data:
            for api in query_data.get('api_list', []):
                api_id, embedding = self.get_api_embedding(api)
                all_apis_in_file.append((api_id, embedding))
        
        # 중복 제거
        unique_apis = {}
        for api_id, embedding in all_apis_in_file:
            if api_id not in unique_apis:
                unique_apis[api_id] = embedding
        
        for query_data in tqdm(data, desc="쿼리 분석 중"):
            # G2_query.json은 relevant APIs가 없으므로 api_list의 모든 API를 relevant로 간주
            relevant_apis = query_data.get('api_list', [])
            if not relevant_apis:
                continue
                
            results['queries_analyzed'] += 1
            
            # 각 relevant API의 임베딩 생성
            api_embeddings = []
            api_ids = []
            
            for api in relevant_apis:
                api_id, embedding = self.get_api_embedding(api)
                api_embeddings.append(embedding)
                api_ids.append(api_id)
                all_relevant_apis.append(api_id)
            
            if len(api_embeddings) < 2:
                continue
                
            # 임베딩을 numpy 배열로 변환
            embeddings_array = np.array(api_embeddings)
            
            # 코사인 유사도 계산
            similarity_matrix = cosine_similarity(embeddings_array)
            
            # 대각선을 제외한 유사도만 계산 (자기 자신과의 유사도 제외)
            # 상삼각 행렬에서 대각선을 제외한 값들만 추출
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            # 평균 유사도 계산
            avg_similarity = np.mean(upper_triangle)
            min_similarity = np.min(upper_triangle)
            max_similarity = np.max(upper_triangle)
            
            results['similarity_stats'].append({
                'query_id': query_data.get('query_id', 'unknown'),
                'num_apis': len(api_embeddings),
                'avg_similarity': avg_similarity,
                'min_similarity': min_similarity,
                'max_similarity': max_similarity,
                'api_ids': api_ids
            })
            
            # Clustering 분석
            if len(api_embeddings) >= 3:
                # K-means 클러스터링 (클러스터 수는 API 개수의 절반으로 설정)
                n_clusters = min(len(api_embeddings) // 2, len(api_embeddings) - 1)
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(embeddings_array)
                    
                    # 클러스터 내 평균 거리 계산
                    cluster_distances = []
                    for i in range(n_clusters):
                        cluster_indices = np.where(cluster_labels == i)[0]
                        if len(cluster_indices) > 1:
                            cluster_embeddings = embeddings_array[cluster_indices]
                            cluster_similarity = cosine_similarity(cluster_embeddings)
                            np.fill_diagonal(cluster_similarity, 0)
                            cluster_distances.append(np.mean(cluster_similarity))
                    
                    results['clustering_results'].append({
                        'query_id': query_data.get('query_id', 'unknown'),
                        'num_clusters': n_clusters,
                        'cluster_labels': cluster_labels.tolist(),
                        'avg_cluster_similarity': np.mean(cluster_distances) if cluster_distances else 0,
                        'api_ids': api_ids
                    })
            
            # Relevant API들의 평균 임베딩과 가장 가까운 API 찾기
            if len(api_embeddings) >= 1 and global_api_pool:
                # Relevant API들의 평균 임베딩 계산
                relevant_avg_embedding = np.mean(embeddings_array, axis=0)
                
                # 모든 API와의 유사도 계산
                similarities_with_all = []
                for api_id, api_embedding in global_api_pool.items():
                    similarity = cosine_similarity([relevant_avg_embedding], [api_embedding])[0][0]
                    similarities_with_all.append((api_id, similarity))
                
                # 유사도 순으로 정렬
                similarities_with_all.sort(key=lambda x: x[1], reverse=True)
                
                # Top 5 API와 relevant API들의 유사도 분석
                top_5_apis = similarities_with_all[:5]
                relevant_api_similarities = []
                
                for relevant_api_id in api_ids:
                    for api_id, similarity in similarities_with_all:
                        if api_id == relevant_api_id:
                            relevant_api_similarities.append(similarity)
                            break
                
                # Relevant API들의 평균 유사도
                avg_relevant_similarity = np.mean(relevant_api_similarities) if relevant_api_similarities else 0
                
                # Top 1 API가 relevant API인지 확인
                top_1_is_relevant = similarities_with_all[0][0] in api_ids
                
                # Top 5 중 relevant API 개수
                top_5_relevant_count = sum(1 for api_id, _ in top_5_apis if api_id in api_ids)
                
                results['nearest_api_analysis'].append({
                    'query_id': query_data.get('query_id', 'unknown'),
                    'num_relevant_apis': len(api_embeddings),
                    'top_5_apis': top_5_apis,
                    'avg_relevant_similarity': avg_relevant_similarity,
                    'top_1_is_relevant': top_1_is_relevant,
                    'top_5_relevant_count': top_5_relevant_count,
                    'relevant_api_ids': api_ids
                })
        
        # 전체 통계 계산
        if results['similarity_stats']:
            results['avg_relevant_apis_per_query'] = np.mean([stat['num_apis'] for stat in results['similarity_stats']])
            results['avg_similarity_across_queries'] = np.mean([stat['avg_similarity'] for stat in results['similarity_stats']])
            results['min_similarity_across_queries'] = np.min([stat['min_similarity'] for stat in results['similarity_stats']])
            results['max_similarity_across_queries'] = np.max([stat['max_similarity'] for stat in results['similarity_stats']])
        
        return results
    
    def generate_report(self, results):
        """분석 결과 리포트 생성"""
        report = []
        report.append("=" * 60)
        report.append("데이터셋 특성 분석 리포트 (전역 API 풀 사용)")
        report.append("=" * 60)
        
        # 전역 API 풀 크기 가져오기
        global_api_count = 0
        for filename, result in results.items():
            if filename != '_global_api_pool_size' and '_global_api_pool_size' in result:
                global_api_count = result['_global_api_pool_size']
                break
        
        report.append(f"전역 API 풀 크기: {global_api_count}개")
        report.append("=" * 60)
        
        for filename, result in results.items():
            # 전역 API 풀 크기 정보는 건너뛰기
            if filename == '_global_api_pool_size':
                continue
                
            report.append(f"\n📁 파일: {filename}")
            report.append("-" * 40)
            report.append(f"총 쿼리 수: {result['total_queries']}")
            report.append(f"분석된 쿼리 수: {result['queries_analyzed']}")
            report.append(f"쿼리당 평균 relevant API 수: {result.get('avg_relevant_apis_per_query', 0):.2f}")
            
            if result['similarity_stats']:
                report.append(f"전체 평균 유사도: {result.get('avg_similarity_across_queries', 0):.4f}")
                report.append(f"최소 유사도: {result.get('min_similarity_across_queries', 0):.4f}")
                report.append(f"최대 유사도: {result.get('max_similarity_across_queries', 0):.4f}")
                
                # 유사도 분포
                similarities = [stat['avg_similarity'] for stat in result['similarity_stats']]
                report.append(f"유사도 표준편차: {np.std(similarities):.4f}")
                
                # 클러스터링 결과
                if result['clustering_results']:
                    cluster_similarities = [cr['avg_cluster_similarity'] for cr in result['clustering_results']]
                    report.append(f"클러스터링된 쿼리 수: {len(result['clustering_results'])}")
                    report.append(f"평균 클러스터 내 유사도: {np.mean(cluster_similarities):.4f}")
                
                # Nearest API 분석 결과 (전역 풀 사용)
                if result['nearest_api_analysis']:
                    top_1_relevant_count = sum(1 for na in result['nearest_api_analysis'] if na['top_1_is_relevant'])
                    top_5_relevant_counts = [na['top_5_relevant_count'] for na in result['nearest_api_analysis']]
                    avg_relevant_similarities = [na['avg_relevant_similarity'] for na in result['nearest_api_analysis']]
                    
                    report.append(f"Nearest API 분석 (전역 API 풀 사용):")
                    report.append(f"  - Top 1이 relevant인 쿼리 수: {top_1_relevant_count}/{len(result['nearest_api_analysis'])} ({top_1_relevant_count/len(result['nearest_api_analysis'])*100:.1f}%)")
                    report.append(f"  - Top 5 중 relevant 평균 개수: {np.mean(top_5_relevant_counts):.2f}")
                    report.append(f"  - Relevant API들의 평균 유사도: {np.mean(avg_relevant_similarities):.4f}")
                    report.append(f"  - 검색 난이도: 전역 {global_api_count}개 API 중에서 정확한 API 찾기")
        
        return "\n".join(report)
    
    def save_detailed_results(self, results, output_dir="analysis_results"):
        """상세 결과를 JSON 파일로 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        def convert_to_serializable(obj):
            """재귀적으로 객체를 JSON 직렬화 가능한 형태로 변환"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        for filename, result in results.items():
            output_file = os.path.join(output_dir, f"{filename.replace('.json', '_analysis.json')}")
            
            # 재귀적으로 모든 numpy 타입을 변환
            serializable_result = convert_to_serializable(result)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        print(f"상세 결과가 {output_dir} 디렉토리에 저장되었습니다.")

def main():
    """메인 실행 함수"""
    print("데이터셋 특성 분석을 시작합니다...")
    
    # 분석기 초기화
    analyzer = DatasetAnalyzer()
    
    # G3_query.json 파일 분석
    g3_query_file = "/home/jhlee/ToolBench/data/instruction/G3_query.json"
    
    if os.path.exists(g3_query_file):
        print(f"\n{g3_query_file} 파일을 분석합니다...")
        
        # 먼저 파일 크기 확인
        file_size = os.path.getsize(g3_query_file) / (1024 * 1024)  # MB
        print(f"파일 크기: {file_size:.1f} MB")
        
        # 전역 API 풀 생성 (전체 파일에서 API 수집)
        print("전역 API 풀을 생성하는 중...")
        global_api_pool = {}
        
        with open(g3_query_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 전체 데이터에서 API 수집 (최대 1000개 쿼리만 사용)
        max_queries_for_pool = min(1000, len(data))
        for i in range(max_queries_for_pool):
            query_data = data[i]
            for api in query_data.get('api_list', []):
                api_id, embedding = analyzer.get_api_embedding(api)
                if api_id not in global_api_pool:
                    global_api_pool[api_id] = embedding
        
        print(f"전역 API 풀 크기: {len(global_api_pool)}개")
        
        # 일부만 분석 (100개 쿼리)
        results = analyzer.analyze_partial_file_g2(g3_query_file, max_queries=100, global_api_pool=global_api_pool)
        
        # 전역 API 풀 크기를 결과에 추가
        results['_global_api_pool_size'] = len(global_api_pool)
        
        # 리포트 생성
        report = analyzer.generate_report({'g3_query_partial': results})
        print(report)
        
        # 상세 결과 저장
        analyzer.save_detailed_results({'g3_query_partial': results}, "analysis_results_g3")
        
        # 리포트를 파일로 저장
        with open("g3_query_analysis_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        print("\n분석이 완료되었습니다!")
        print("리포트가 g3_query_analysis_report.txt 파일에 저장되었습니다.")
    else:
        print(f"파일을 찾을 수 없습니다: {g3_query_file}")
        
        # train.json 파일 분석 (일부만)
        train_file = "librarian/data/train.json"
        
        if os.path.exists(train_file):
            print(f"\n{train_file} 파일을 분석합니다...")
            
            # 먼저 파일 크기 확인
            file_size = os.path.getsize(train_file) / (1024 * 1024)  # MB
            print(f"파일 크기: {file_size:.1f} MB")
            
            # 전역 API 풀 생성 (전체 파일에서 API 수집)
            print("전역 API 풀을 생성하는 중...")
            global_api_pool = {}
            
            with open(train_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 전체 데이터에서 API 수집 (최대 1000개 쿼리만 사용)
            max_queries_for_pool = min(1000, len(data))
            for i in range(max_queries_for_pool):
                query_data = data[i]
                for api in query_data.get('relevant_apis', []):
                    api_id, embedding = analyzer.get_api_embedding(api)
                    if api_id not in global_api_pool:
                        global_api_pool[api_id] = embedding
            
            print(f"전역 API 풀 크기: {len(global_api_pool)}개")
            
            # 일부만 분석 (100개 쿼리)
            results = analyzer.analyze_partial_file(train_file, max_queries=100, global_api_pool=global_api_pool)
            
            # 전역 API 풀 크기를 결과에 추가
            results['_global_api_pool_size'] = len(global_api_pool)
            
            # 리포트 생성
            report = analyzer.generate_report({'train_partial': results})
            print(report)
            
            # 상세 결과 저장
            analyzer.save_detailed_results({'train_partial': results}, "analysis_results_train")
            
            # 리포트를 파일로 저장
            with open("train_analysis_report.txt", "w", encoding="utf-8") as f:
                f.write(report)
            
            print("\n분석이 완료되었습니다!")
            print("리포트가 train_analysis_report.txt 파일에 저장되었습니다.")
        else:
            print(f"파일을 찾을 수 없습니다: {train_file}")
            
            # 기존 ToolBench 데이터 분석
            data_dir = "/home/jhlee/ToolBench/data/test_instruction"
            
            print(f"\n기존 ToolBench 데이터를 분석합니다: {data_dir}")
            
            # 모든 파일 분석
            results = analyzer.analyze_all_files(data_dir)
            
            # 리포트 생성
            report = analyzer.generate_report(results)
            print(report)
            
            # 상세 결과 저장
            analyzer.save_detailed_results(results)
            
            # 리포트를 파일로 저장
            with open("dataset_analysis_report.txt", "w", encoding="utf-8") as f:
                f.write(report)
            
            print("\n분석이 완료되었습니다!")
            print("리포트가 dataset_analysis_report.txt 파일에 저장되었습니다.")

if __name__ == "__main__":
    main() 