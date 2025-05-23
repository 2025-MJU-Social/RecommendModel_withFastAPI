from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# --- 초기 설정 ---
app = FastAPI()

# --- 요청 모델 정의 ---
class RecommendRequest(BaseModel):
    age: int
    sex: str  # 'm' 또는 'f'
    liked_titles: List[str]

# --- 데이터 로딩 (앱 시작 시 한 번만) ---
contents = get_contents_data()
preprocessing_contents = preprocessing_contents_data(contents)
embeddings = get_embeddings(preprocessing_contents)
intentions = get_ott_intension_data()
experiences = get_ott_experience_data()

# --- 엔드포인트 정의 ---
@app.post("/recommend")
def recommend(request: RecommendRequest):
    age = request.age
    sex = request.sex.lower()
    liked_titles = request.liked_titles

    if sex not in ['m', 'f']:
        raise HTTPException(status_code=400, detail="성별은 'm' 또는 'f'여야 합니다.")

    # 콘텐츠 기반 추천
    contents_based = genre_based_recommended_contents(preprocessing_contents, embeddings, liked_titles)
    if isinstance(contents_based, str):  # 오류 메시지 반환 시
        return {"error": contents_based}

    # 사용자 기반 추천
    user_based = user_based_recommended_contents(age, sex)

    # 추천 병합
    merged = merge_recommended_contents(preprocessing_contents, contents_based, user_based)

    # OTT 점수 계산
    ott_score = calculate_ott_score(age, sex, intentions, experiences)

    return {
        "recommendations": merged.to_dict(orient="records"),
        "ott_scores": ott_score
    }

# 헬스 체크
@app.get("/")
def root():
    return {"message": "FastAPI 추천 시스템 동작 중"}
