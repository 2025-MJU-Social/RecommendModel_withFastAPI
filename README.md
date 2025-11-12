# OTT 개인화 추천 시스템 (Personalized OTT Recommendation System)
<img width="2300" height="1200" alt="스크린샷 2025-11-10 205638" src="https://github.com/user-attachments/assets/8f9fda47-e7fa-447b-b6d0-a13c2c2315f4" />

> **사용자의 성별, 연령대, 선호 콘텐츠, 주 시청 시간, 예산**을 입력받아  
> 어떤 **OTT 플랫폼**의 **요금제와 콘텐츠 조합이 가장 효율적인지** 추천하는 AI 기반 시스템입니다.

---

## 프로젝트 개요
<img width="600" height="600" alt="스크린샷 2025-11-10 195214" src="https://github.com/user-attachments/assets/c0b0f7f4-5e25-454b-ac0c-ced8508948e0" />

현대인은 다양한 OTT 서비스를 구독하지만,  
“**나에게 가장 효율적인 플랫폼과 요금제는 무엇일까?**”에 대한 판단은 어렵습니다.  
본 프로젝트는 데이터 기반 분석과 자연어 임베딩 모델을 활용하여  
개인에게 최적화된 OTT 조합을 제안하는 **AI 맞춤형 콘텐츠 추천 플랫폼**을 구현합니다.

---

## 주요 기능

| 구분 | 설명 |
|------|------|
| **개인화 콘텐츠 추천** | 사용자가 좋아하는 콘텐츠 제목을 기반으로 유사 장르 콘텐츠 자동 탐색 |
| **성별·연령대 기반 통계 추천** | 실제 OTT 이용 의향/경험 데이터를 반영한 가중치 계산 |
| **예산 최적화 요금제 추천** | 주 시청시간과 월 예산을 고려해 OTT별 효율적 요금제 조합 제안 |
| **자연어 임베딩 기반 유사도 분석** | SentenceTransformer(`MiniLM-L12-v2`)를 활용해 장르/세부 장르 의미적 유사도 계산 |
| **FastAPI 서버 연동** | `/recommend`와 `/ott_recommend` 두 가지 REST API 엔드포인트 제공 |

---

## 📁 프로젝트 구조
```
OTT_Recommendation_System/
├── main.py # FastAPI 서버 구동, 추천 API 엔드포인트 정의
├── recommendation.py # 연령/성별 기반 OTT 추천 및 콘텐츠 랭킹 로직
├── ott_recommendation.py # 예산·시청시간 기반 OTT 요금제 최적화 추천 모델
├── test_model.py # 로컬 테스트용 CLI 입력 기반 추천 실행
├── data/ # 데이터셋 폴더
│ ├── fixed_contents.csv # 콘텐츠 메타데이터
│ ├── ott_price.csv # OTT 서비스별 요금제 정보
│ ├── daily_MALE_250514.csv # 남성 이용자별 선호 콘텐츠
│ ├── daily_FEMALE_250514.csv # 여성 이용자별 선호 콘텐츠
│ ├── OTT_이용_경험_여부_서비스별_.csv
│ └── OTT_유료서비스_계속_이용_의향__서비스별_.csv
```

##  시스템 구성

### 1. 데이터 처리 및 전처리
- `fixed_contents.csv`에서 콘텐츠별 장르·세부장르·플랫폼 정보를 추출  
- `create_soup()` → 다중 컬럼을 하나의 텍스트 필드로 통합  
- `get_embeddings()` → SentenceTransformer를 이용해 의미 벡터 생성
- 
<img width="1493" height="688" alt="스크린샷 2025-11-10 204538" src="https://github.com/user-attachments/assets/6bba0b6b-dd42-4e56-818b-4e50fd4b0f17" />

### 2. 개인화 추천 알고리즘 (`recommendation.py`)
- **콘텐츠 기반 추천**: 사용자가 입력한 콘텐츠의 임베딩 벡터와 전체 콘텐츠 벡터 간 cosine similarity 계산  
- **사용자 기반 추천**: 성별·연령대별 선호 콘텐츠 통계 반영  
- **OTT 가중치 계산**: 실제 이용 경험/의향 데이터 기반으로 플랫폼별 점수 계산  
- **최종 출력**: 플랫폼별 가중 평균 점수를 기준으로 OTT 순위 생성  

###  3. 요금제 최적화 (`ott_recommendation.py`)
- 주 시청시간 × 4주 = 월 시청시간 계산  
- 콘텐츠 장르 유사도 + 평점 + 효율성(러닝타임 반비례)을 종합 점수로 산출  
- 예산을 초과하지 않도록 구독 조합 자동 선택  
- 출력:
  - 추천 콘텐츠 목록 (제목, 장르, 시청시간)
  - 추천 OTT 조합 및 월 구독료 총합
