from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException
from recommendation import run_recommendation

# --- 초기 설정 ---
app = FastAPI()

class RecommendRequest(BaseModel):
    age: int
    sex: str
    liked_titles: List[str]

@app.post("/recommend")
def recommend(request: RecommendRequest):
    age = request.age
    sex = request.sex.lower()
    liked_titles = request.liked_titles
    print("여기는 post로 입력받는곳")
    print(age)
    print(sex)
    print(liked_titles)
    print("-----------------------")

    if sex not in ['m', 'f']:
        raise HTTPException(status_code=400, detail="성별은 'm' 또는 'f'여야 합니다.")

    result = run_recommendation(age, sex, liked_titles)
    print("result:",result)
    return result

# 헬스 체크
@app.get("/")
def root():
    return {"message": "FastAPI 추천 시스템 동작 중"}
