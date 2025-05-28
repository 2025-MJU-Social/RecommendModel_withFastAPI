import os
import logging
import math
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

from recommendation import *
from ott_recommendation import *

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 상수 정의
DATA_DIR = Path("data")
REQUIRED_FILES = [
    "fixed_contents.csv",
    "train_data.csv", 
    "ott_price.csv",
    "daily_MALE_250514.csv",
    "daily_FEMALE_250514.csv"
]

class GlobalData:
    """전역 데이터를 관리하는 클래스"""
    def __init__(self):
        self.recommendation_contents = None
        self.embeddings = None
        self.preprocessing_contents = None
        self.intentions = None
        self.experience = None
        self.ott_contents = None  # ott_recommendation용 contents
        self.prices = None
        self.language_model = None
        self.is_loaded = False

global_data = GlobalData()

def validate_data_files():
    """필수 데이터 파일들이 존재하는지 확인"""
    missing_files = []
    for file_name in REQUIRED_FILES:
        if not (DATA_DIR / file_name).exists():
            missing_files.append(str(DATA_DIR / file_name))
    
    if missing_files:
        raise FileNotFoundError(f"필수 데이터 파일이 없습니다: {missing_files}")

async def load_data_once():
    """서버 시작 시 데이터 로드"""
    try:
        logger.info("서버 시작: 데이터 유효성 검사 중...")
        validate_data_files()
        
        logger.info("데이터 로드 중...")
        
        # Recommendation 시스템용 데이터 로드
        global_data.recommendation_contents = get_contents_data()
        global_data.preprocessing_contents = preprocessing_contents_data(
            global_data.recommendation_contents
        )
        global_data.embeddings = get_embeddings(global_data.preprocessing_contents)
        global_data.intentions = get_ott_intension_data()
        global_data.experience = get_ott_experience_data()
        
        # OTT Recommendation 시스템용 데이터/모델/임베딩 미리 준비
        global_data.ott_contents, global_data.prices, global_data.language_model = prepare_ott_recommendation_data()
        
        global_data.is_loaded = True
        logger.info("데이터 로드 완료")
        
    except Exception as e:
        logger.error(f"데이터 로드 실패: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시 실행
    logger.info("애플리케이션 시작 중...")
    await load_data_once()
    yield
    # 종료 시 실행 (필요한 경우)
    logger.info("애플리케이션 종료 중...")

# FastAPI 앱 생성 시 lifespan 매개변수 추가
app = FastAPI(
    title="OTT Recommendation API", 
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 공통 파라미터 변환 함수들
def age_group_to_age(age_group: str) -> int:
    """연령대를 나이로 변환 (기존 모델용)"""
    age_mapping = {
        "10대": 15,
        "20대": 25,
        "30대": 35,
        "40대": 45,
        "50대 이상": 55
    }
    return age_mapping.get(age_group, 25)  # 기본값: 25세

def gender_to_sex(gender: str) -> str:
    """성별 코드를 변환 (기존 모델용)"""
    return gender.lower()  # 'm' 또는 'f' 그대로 사용

# 통일된 요청 모델들
class RecommendRequest(BaseModel):
    age_group: str
    gender: str
    liked_titles: List[str]
    
    @field_validator('age_group')
    def validate_age_group(cls, v):
        valid_ages = ['10대', '20대', '30대', '40대', '50대', '50대 이상']
        if v not in valid_ages:
            raise ValueError(f'연령대는 {valid_ages} 중 하나여야 합니다')
        return v
    
    @field_validator('gender')
    def validate_gender(cls, v):
        if v.lower() not in ['m', 'f']:
            raise ValueError("성별은 'm' 또는 'f'여야 합니다")
        return v.lower()
    
    @field_validator('liked_titles')
    def validate_liked_titles(cls, v):
        if not v:
            raise ValueError('최소 하나의 좋아하는 제목을 입력해주세요')
        return v

class OttRecommendationRequest(BaseModel):
    base_genres: List[str]
    detail_genres: List[str]
    age_group: str
    gender: str
    weekly_hours: float
    budget: float
    
    @field_validator('age_group')
    def validate_age_group(cls, v):
        valid_ages = ['10대', '20대', '30대', '40대', '50대', '50대 이상']
        if v not in valid_ages:
            raise ValueError(f'연령대는 {valid_ages} 중 하나여야 합니다')
        return v
    
    @field_validator('gender')
    def validate_gender(cls, v):
        if v.lower() not in ['m', 'f']:
            raise ValueError("성별은 'm' 또는 'f'여야 합니다")
        return v.lower()
    
    @field_validator('weekly_hours')
    def validate_weekly_hours(cls, v):
        if v <= 0:
            raise ValueError('주간 시청 시간은 0보다 커야 합니다')
        return v
    
    @field_validator('budget')
    def validate_budget(cls, v):
        if v <= 0:
            raise ValueError('예산은 0보다 커야 합니다')
        return v

def check_data_loaded():
    """데이터 로드 상태 확인"""
    if not global_data.is_loaded:
        raise HTTPException(
            status_code=503, 
            detail="서버가 아직 초기화 중입니다. 잠시 후 다시 시도해주세요."
        )

@app.get("/")
def root():
    return {
        "message": "OTT 추천 시스템 API",
        "version": "1.0.0",
        "endpoints": ["/recommend", "/ott_recommend", "/health"],
        "parameter_format": {
            "age_group": "10대, 20대, 30대, 40대, 50대, 50대 이상",
            "gender": "m 또는 f"
        }
    }

@app.get("/health")
def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy" if global_data.is_loaded else "loading",
        "data_loaded": global_data.is_loaded
    }

@app.post("/recommend")
def recommend(request: RecommendRequest):
    """기존 추천 시스템 엔드포인트 (통일된 파라미터)"""
    try:
        check_data_loaded()
        
        # 통일된 파라미터를 기존 모델 형태로 변환
        age = age_group_to_age(request.age_group)
        sex = gender_to_sex(request.gender)
        
        logger.info(f"추천 요청 - 연령대: {request.age_group}({age}세), 성별: {request.gender}")
        
        result = run_recommendation(
            age=age,  # 변환된 나이
            sex=sex,  # 변환된 성별
            liked_titles=request.liked_titles,
            contents=global_data.recommendation_contents,
            preprocessing_contents=global_data.preprocessing_contents,
            embeddings=global_data.embeddings,
            intentions=global_data.intentions,
            experiences=global_data.experience
        )
        
        return {
            "status": "success",
            "user_info": {
                "age_group": request.age_group,
                "gender": request.gender,
                "liked_titles": request.liked_titles
            },
            "recommendations": result
        }
        
    except Exception as e:
        logger.error(f"추천 실행 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"추천 실행 중 오류가 발생했습니다: {str(e)}")

@app.post("/ott_recommend")
def ott_recommend(req: OttRecommendationRequest):
    """OTT 추천 시스템 엔드포인트 (통일된 파라미터)"""
    try:
        check_data_loaded()
        
        logger.info(f"OTT 추천 요청 - 연령대: {req.age_group}, 성별: {req.gender}")
        
        sel_df, plan, total_hours, total_cost = ott_recommendation_model(
            contents=global_data.ott_contents,
            prices=global_data.prices,
            base_genres=req.base_genres,
            detail_genres=req.detail_genres,
            age_group=req.age_group,  # 그대로 전달
            gender=req.gender,        # 그대로 전달
            weekly_hours=req.weekly_hours,
            budget=req.budget,
            model=global_data.language_model
        )
        
        # 데이터 정제
        recommendations = []
        for _, row in sel_df.iterrows():
            rec = {
                'title': row.get('title', ''),
                'platform': row.get('platform', '') if pd.notna(row.get('platform')) else '',
                'score': float(row.get('score', 0)) if pd.notna(row.get('score')) else 0.0,
                'watch_hours': float(row.get('watch_hours', 0)) if pd.notna(row.get('watch_hours')) else 0.0,
                'genre': row.get('genre', '') if pd.notna(row.get('genre')) else '',
                'genre_detail': row.get('genre_detail', '') if pd.notna(row.get('genre_detail')) else ''
            }
            recommendations.append(rec)

        # 구독 플랜 정제
        cleaned_plan = {}
        for key, v in plan.items():
            cleaned_plan[key] = {
                "plan_name": v["plan_name"],
                "price": int(v["price"]),
                "cover_count": int(v["cover_count"])
            }

        return {
            "status": "success",
            "user_info": {
                "age_group": req.age_group,
                "gender": req.gender,
                "preferences": {
                    "base_genres": req.base_genres,
                    "detail_genres": req.detail_genres,
                    "weekly_hours": req.weekly_hours,
                    "budget": req.budget
                }
            },
            "recommendations": recommendations,
            "total_estimated_watch_time": round(float(total_hours), 2),
            "total_subscription_cost": int(total_cost),
            "subscription_plan": cleaned_plan
        }

    except Exception as e:
        logger.error(f"OTT 추천 실행 중 오류: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"OTT 추천 실행 중 오류가 발생했습니다: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)