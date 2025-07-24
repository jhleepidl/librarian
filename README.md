# Task Description Based API Retriever

이 프로젝트는 Task Description을 기반으로 쿼리에 관련된 API들을 검색하는 시스템입니다. GPT를 활용하여 쿼리와 task description을 매칭하고, 해당 task에 연결된 API 리스트를 반환합니다.

## 📊 데이터 통계

### 1. 전체 데이터셋 통계
- **총 노드 수**: 555개
- **카테고리 수**: 32개 (Database, eCommerce, Movies, Visual_Recognition, Communication, Payments, Science, Tools, Gaming, Search, Medical, Text_Analysis, Data, Business_Software, Social, Artificial_Intelligence_Machine_Learning, Financial, Sports, Media, Entertainment, Food, News_Media, Finance, Education, Location, Business, Mapping, Other, Travel, Translation, Video_Images, Music)
- **Task 노드 수**: 20개 (merged_task_22 ~ merged_task_41)
- **API 노드 수**: 503개

### 2. 데이터 파일별 통계
```
📁 data/
├── G1_query_sampled_50apis_enhanced.json    # 15,502 lines (403KB)
├── G1_query_sampled_200apis_enhanced.json   # 1.8MB
├── G1_query_sampled_1000apis_enhanced.json  # 8.7MB
├── G2_query_sampled_50apis_enhanced.json    # 17,013 lines (552KB)
├── G2_query_sampled_200apis_enhanced.json   # 2.6MB
├── G2_query_sampled_1000apis_enhanced.json  # 11MB
├── G3_query_sampled_50apis_enhanced.json    # 21,100 lines (660KB)
├── G3_query_sampled_200apis_enhanced.json   # 3.1MB
└── G3_query_sampled_1000apis_enhanced.json  # 34MB
```

### 3. Task Description 통계
- **생성된 Task Description**: 17개
- **평균 Description 길이**: 30-50단어
- **Description 특성**: 키워드 중심, 간결함, 특징적인 기능 강조

### 4. 실험 데이터 통계
- **테스트 케이스**: 10개 (quantitative_evaluation.py)
- **평균 Precision**: 1.3%
- **평균 Recall**: 20.0%
- **평균 F1-Score**: 0.024

## 🔧 코드 생성 매커니즘

### 1. Task Description Generator (`task_description_generator.py`)

#### 생성 프로세스:
1. **데이터 로드**: `graph_info.json`에서 task-API 연결 정보 로드
2. **API 정보 수집**: 각 task에 연결된 API 목록과 설명 수집
3. **Query 정보 수집**: 각 task와 연결된 실제 쿼리들 수집
4. **GPT 프롬프트 구성**:
   ```
   Generate a concise task description for embedding vector search.
   
   Task: {task_name}
   APIs ({api_count}): {api_list}
   Sample queries: {sample_queries}
   
   Create a brief, keyword-rich description (30-50 words) that captures:
   - Core functionality
   - Key API types used
   - Main query patterns
   
   Focus on distinctive features for search matching.
   ```
5. **GPT-4o-mini 호출**: OpenAI API를 통해 description 생성
6. **결과 저장**: `task_descriptions.json`에 저장

#### 생성된 Description 예시:
```json
{
  "merged_task_24": "Task: Merged Task 24 integrates calendar invites and onboarding product queries via APIs. Key functionalities include sending native calendar invites, retrieving product categories, and checking order statuses. Common queries involve catalog inquiries, category counts, and order status checks.",
  "merged_task_25": "Task: Merged Task 25 integrates facial animation and domain backordering APIs to facilitate realistic facial expression creation and domain management. Key APIs include Face Animer for job creation and result retrieval, and Domain Backorder for managing backorders. Common queries involve blog launches and expressive facial animations."
}
```

### 2. Knowledge Graph Builder (`knowledge_graph_builder.py`)

#### 그래프 구축 프로세스:
1. **데이터 로드**: 3개 그룹(G1, G2, G3)의 enhanced JSON 파일들 로드
2. **카테고리 및 API 추출**: 
   - 카테고리별 API 분류
   - Query-API 매핑 생성
   - API별 카테고리 매핑 생성
3. **Task 노드 생성**:
   - 고유한 API set들을 기반으로 task 노드 생성
   - Jaccard 유사도를 사용한 task 병합 (최대 50개 task)
   - 유사도 임계값: 0.3
4. **그래프 구축**:
   - NetworkX DiGraph 사용
   - Category → Task → API → Query 연결 구조
   - 가중치 기반 엣지 생성

#### 클러스터링 알고리즘:
```python
def _merge_by_jaccard_similarity(self, api_sets: List[Set[str]], max_task_nodes: int):
    # Jaccard 유사도 계산
    similarity_matrix = np.zeros((len(api_sets), len(api_sets)))
    for i in range(len(api_sets)):
        for j in range(i+1, len(api_sets)):
            intersection = len(api_sets[i] & api_sets[j])
            union = len(api_sets[i] | api_sets[j])
            similarity = intersection / union if union > 0 else 0
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
    
    # Agglomerative Clustering 적용
    clustering = AgglomerativeClustering(
        n_clusters=max_task_nodes,
        affinity='precomputed',
        linkage='complete'
    )
    clusters = clustering.fit_predict(1 - similarity_matrix)
```

### 3. Hybrid Retriever (`hybrid_retriever.py`)

#### 하이브리드 검색 메커니즘:
1. **Vector Database 구축**:
   - SentenceTransformer('all-MiniLM-L6-v2') 사용
   - API 텍스트: `{tool_name}:{api_name} - {description}`
   - FAISS 인덱스 구축
2. **LLM 필터링**:
   - GPT-4o-mini를 사용한 카테고리 및 task 필터링
   - 쿼리 분석을 통한 관련 카테고리/태스크 식별
3. **하이브리드 검색**:
   - Vector similarity + LLM 필터링 조합
   - Top-k 결과 반환

#### 검색 프로세스:
```python
def search_apis(self, query: str, top_k: int = 10):
    # 1. LLM 필터링
    relevant_categories, relevant_tasks = self.llm_filter_query(query)
    
    # 2. Vector 검색
    query_embedding = self.embedding_model.encode([query])
    distances, indices = self.index.search(query_embedding, top_k * 2)
    
    # 3. 필터링 적용
    filtered_results = []
    for idx in indices[0]:
        metadata = self.api_metadata[idx]
        if self._matches_filters(metadata, relevant_categories, relevant_tasks):
            filtered_results.append(metadata)
    
    return filtered_results[:top_k]
```

### 4. Task Description Retriever (`task_description_retriever.py`)

#### 검색 프로세스:
1. **Task Description 로드**: `task_descriptions.json`에서 모든 task description 로드
2. **GPT 매칭**: 쿼리와 모든 task description을 GPT에 전달
3. **관련 Task 식별**: GPT가 관련 task들을 순서대로 반환
4. **API 수집**: 관련 task들에 연결된 API들을 수집
5. **중복 제거**: 중복된 API 제거 후 결과 반환

#### GPT 프롬프트:
```
Given a user query and a list of task descriptions, identify the most relevant tasks.

Query: {query}

Task Descriptions:
{task_descriptions}

Return only the task names in order of relevance (most relevant first), separated by commas.
```

## 🧪 실험 세팅

### 1. 정량적 평가 실험 (`quantitative_evaluation.py`)

#### 실험 설정:
- **테스트 케이스**: 10개 (수동 생성)
- **평가 메트릭**: Precision, Recall, F1-Score
- **Ground Truth**: 수동으로 정의된 관련 API 목록

#### 테스트 케이스 예시:
```python
test_cases = [
    {
        "query": "I want to listen to some music and get random facts. Can you help me find albums by a specific artist and also provide me with some interesting random facts?",
        "relevant_apis": [
            "Deezer:Album",
            "Deezer:Artist", 
            "Numbers:Get random fact",
            "Numbers:Get trivia fact"
        ]
    },
    {
        "query": "I'm looking for some humor to brighten my day. Can you fetch a random joke and also get me a funny quote?",
        "relevant_apis": [
            "Chuck Norris:/jokes/random",
            "Quotes:quote",
            "World of Jokes:Get Random Joke"
        ]
    }
]
```

#### 평가 함수:
```python
def evaluate_retriever(test_cases):
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    
    for test_case in test_cases:
        query = test_case["query"]
        ground_truth = set(test_case["relevant_apis"])
        
        # Retriever 실행
        retrieved_apis = retriever.retrieve_apis_for_query(query)
        retrieved_set = set(retrieved_apis)
        
        # 메트릭 계산
        precision = len(ground_truth & retrieved_set) / len(retrieved_set) if retrieved_set else 0
        recall = len(ground_truth & retrieved_set) / len(ground_truth) if ground_truth else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    
    return {
        'avg_precision': total_precision / len(test_cases),
        'avg_recall': total_recall / len(test_cases),
        'avg_f1': total_f1 / len(test_cases)
    }
```

### 2. 상세 분석 실험 (`detailed_retriever_test.py`)

#### 실험 목적:
- 단계별 retrieve 과정 시각화
- GPT 프롬프트와 응답 확인
- Task 매칭 및 API 수집 과정 분석

#### 분석 프로세스:
1. **쿼리 전처리**: 입력 쿼리 정규화
2. **GPT 호출**: 모든 task description과 함께 GPT 호출
3. **응답 파싱**: GPT 응답에서 관련 task 추출
4. **API 수집**: 관련 task들의 API 수집
5. **결과 분석**: 각 단계별 결과 출력

### 3. Ground Truth 비교 실험 (`compare_retriever_with_gt.py`)

#### 실험 설정:
- **데이터 소스**: 실제 쿼리 데이터에서 샘플링
- **Ground Truth**: 실제 쿼리에 연결된 API 목록
- **비교 메트릭**: 매칭/누락/추가 API 분석

#### 비교 프로세스:
```python
def compare_with_ground_truth(query_id, ground_truth_apis, retrieved_apis):
    ground_truth_set = set(ground_truth_apis)
    retrieved_set = set(retrieved_apis)
    
    matched = ground_truth_set & retrieved_set
    missed = ground_truth_set - retrieved_set
    extra = retrieved_set - ground_truth_set
    
    return {
        'matched': list(matched),
        'missed': list(missed),
        'extra': list(extra),
        'precision': len(matched) / len(retrieved_set) if retrieved_set else 0,
        'recall': len(matched) / len(ground_truth_set) if ground_truth_set else 0
    }
```

### 4. 하이브리드 검색 실험 (`test_hybrid_retriever.py`)

#### 실험 설정:
- **Vector Model**: all-MiniLM-L6-v2
- **LLM Model**: GPT-4o-mini
- **검색 방식**: Vector similarity + LLM 필터링
- **Top-k**: 10개 결과

#### 실험 프로세스:
1. **Vector Database 구축**: API 텍스트 임베딩 생성
2. **LLM 필터링**: 쿼리 분석을 통한 카테고리/태스크 필터링
3. **하이브리드 검색**: 필터링된 결과에서 vector similarity 검색
4. **성능 평가**: Precision, Recall, F1-Score 계산

## 📁 프로젝트 구조

```
librarian/
├── 📊 데이터 파일
│   ├── graph_info.json                    # Task-API 연결 정보
│   ├── task_descriptions.json             # 생성된 Task Descriptions
│   └── data/                             # 쿼리 데이터
│       ├── G1_query_sampled_50apis_enhanced.json
│       ├── G2_query_sampled_50apis_enhanced.json
│       └── G3_query_sampled_50apis_enhanced.json
│
├── 🔧 핵심 컴포넌트
│   ├── task_description_generator.py      # Task Description 생성기
│   ├── task_description_retriever.py     # Task Description 기반 Retriever
│   └── knowledge_graph_builder.py        # 지식 그래프 구축
│
├── 🧪 실험 및 평가
│   ├── test_task_retriever.py            # Retriever 테스트
│   ├── detailed_retriever_test.py        # 상세 Retrieve 과정 분석
│   ├── compare_retriever_with_gt.py      # Ground Truth와 비교
│   └── quantitative_evaluation.py        # 정량적 성능 평가
│
├── 📈 분석 도구
│   ├── graph_connection_analyzer.py      # 그래프 연결 분석
│   ├── graph_content_analyzer.py         # 그래프 내용 분석
│   ├── graph_visualizer.py               # 그래프 시각화
│   └── query_text_analyzer.py            # 쿼리 텍스트 분석
│
└── 📚 문서
    ├── README.md                         # 이 파일
    └── README_task_description.md        # Task Description 생성기 문서
```

## 🎯 주요 기능

### 1. Task Description 생성
- **파일**: `task_description_generator.py`
- **기능**: GPT-4o-mini를 사용하여 task의 description을 자동 생성
- **입력**: graph_info.json의 task-API 연결 정보
- **출력**: 간결하고 특징적인 task description (30-50단어)

### 2. Task Description 기반 Retriever
- **파일**: `task_description_retriever.py`
- **기능**: 쿼리를 받아 관련 task들을 찾고 연결된 API 리스트 반환
- **프로세스**:
  1. 쿼리와 모든 task description을 GPT에 전달
  2. 관련 task들을 순서대로 반환받음
  3. 해당 task들에 연결된 API들을 수집
  4. 중복 제거 후 결과 반환

### 3. 정량적 성능 평가
- **파일**: `quantitative_evaluation.py`
- **기능**: Precision, Recall, F1-Score를 통한 성능 측정
- **결과**: 현재 성능은 매우 낮음 (F1-Score: 0.024)

## 🚀 사용 방법

### 1. 환경 설정
```bash
# OpenAI API 키 설정
export OPENAI_API_KEY="your-api-key-here"

# 필요한 패키지 설치
pip install openai sentence-transformers faiss-cpu networkx scikit-learn numpy
```

### 2. Task Description 생성
```bash
# 모든 task의 description 생성
python task_description_generator.py

# 특정 task의 description 생성
python task_description_generator.py merged_task_24
```

### 3. Retriever 테스트
```bash
# 기본 테스트
python test_task_retriever.py

# 상세 과정 분석
python detailed_retriever_test.py

# Ground Truth와 비교
python compare_retriever_with_gt.py

# 정량적 성능 평가
python quantitative_evaluation.py
```

## 📊 실험 결과

### 1. Task Description 생성 결과
- **생성된 Task**: 17개
- **Description 특성**: 간결하고 키워드 중심 (30-50단어)
- **예시**:
  ```
  merged_task_24: "Integrates calendar invites and onboarding product queries via APIs. Key functionalities include sending native calendar invites, retrieving product categories, and checking order statuses."
  ```

### 2. Retriever 성능 평가
- **평균 Precision**: 1.3% (매우 낮음)
- **평균 Recall**: 20.0% (낮음)
- **평균 F1-Score**: 0.024 (매우 낮음)

### 3. 상세 실험 결과
- **테스트 케이스**: 10개
- **Ground Truth API 매칭**: 거의 없음
- **주요 문제점**:
  - Task description이 너무 일반적
  - GPT 프롬프트 최적화 필요
  - API 매칭 로직 개선 필요

## 🔍 실험 과정

### 1. 기본 Retriever 테스트
```bash
python test_task_retriever.py
```
- 실제 데이터에서 10개 쿼리 샘플링
- 각 쿼리에 대해 관련 task와 API 반환
- 결과 요약 및 분석

### 2. 상세 Retrieve 과정 분석
```bash
python detailed_retriever_test.py
```
- 단계별 retrieve 과정 시각화
- GPT 프롬프트와 응답 확인
- Task 매칭 및 API 수집 과정 분석

### 3. Ground Truth 비교
```bash
python compare_retriever_with_gt.py
```
- 실제 쿼리와 정답 API 비교
- 매칭/누락/추가 API 분석
- 정확도 측정

### 4. 정량적 성능 평가
```bash
python quantitative_evaluation.py
```
- 10개 테스트 케이스로 성능 측정
- Precision, Recall, F1-Score 계산
- 전체 성능 요약

## 🎯 주요 발견사항

### 1. Task Description의 중요성
- 현재 description이 너무 일반적
- API별 특성을 반영하지 못함
- 더 구체적이고 특징적인 description 필요

### 2. GPT 프롬프트 최적화 필요
- 현재 프롬프트로는 정확한 매칭 어려움
- API 이름과 기능을 더 명확히 인식하도록 개선 필요
- 의미적 유사도 기반 매칭 도입 필요

### 3. 성능 개선 방향
- Task description 재생성 (더 구체적)
- GPT 프롬프트 최적화
- API 매칭 알고리즘 개선
- 더 많은 테스트 케이스 확장

## 🛠️ 기술 스택

- **Python**: 주요 프로그래밍 언어
- **OpenAI GPT-4o-mini**: Task description 생성 및 쿼리 매칭
- **SentenceTransformer**: 텍스트 임베딩 생성
- **FAISS**: 벡터 검색 인덱스
- **NetworkX**: 그래프 분석 및 시각화
- **Scikit-learn**: 클러스터링 및 유사도 계산
- **JSON**: 데이터 저장 및 교환
- **Graph Analysis**: Task-API 연결 분석

## 📈 향후 개선 계획

1. **Task Description 개선**
   - API별 특성을 반영한 더 구체적인 description
   - 키워드 중심의 특징적인 description

2. **GPT 프롬프트 최적화**
   - 더 정확한 task 매칭을 위한 프롬프트 튜닝
   - API 이름과 기능 인식 개선

3. **API 매칭 알고리즘 개선**
   - 의미적 유사도 기반 매칭
   - 벡터 임베딩을 활용한 유사도 계산

4. **성능 평가 확장**
   - 더 많은 테스트 케이스
   - 다양한 도메인의 쿼리
   - 실시간 성능 모니터링

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여

버그 리포트, 기능 제안, 코드 기여를 환영합니다!

---

**참고**: 이 프로젝트는 Task Description 기반 API 검색 시스템의 프로토타입이며, 현재 성능은 매우 낮은 상태입니다. 지속적인 개선이 필요합니다.