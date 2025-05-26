from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException
from recommendation import run_recommendation,get_contents_data,preprocessing_contents_data,get_embeddings,get_ott_intension_data,get_ott_experience_data
from ott_recommendation import load_data,add_genre_embeddings,estimate_runtime_hours,load_language_model,ott_recommendation_model
from fastapi.encoders import jsonable_encoder
#loging 설정
import logging
import math
logging.basicConfig(level=logging.INFO)  # INFO 레벨 이상만 출력
logger = logging.getLogger(__name__)

# --- 초기 설정 ---
app = FastAPI()

# --- 글로벌 변수로 데이터 캐싱용 ---
global_data = {}

# --- 시작 시 한 번만 실행되는 부분 ---
@app.on_event("startup")
def load_data_once():
    print("서버 시작: 데이터 로드 중...")
    contents = get_contents_data()
    preprocessing_contents = preprocessing_contents_data(contents)
    embeddings = get_embeddings(preprocessing_contents)
    intentions = get_ott_intension_data()
    experience = get_ott_experience_data()
    
# ott_recommendation_ 데이터들 시작 시 저장
    contents, prices = load_data()
    model = load_language_model()
    contents = add_genre_embeddings(contents, model)

    # global_data 딕셔너리에 저장
    global_data["contents"] = contents
    global_data["embeddings"] = embeddings
    global_data["preprocessing_contents"] = preprocessing_contents
    global_data["intentions"] = intentions
    global_data['experience']= experience
    global_data['contents']= contents
    global_data['prices']= prices
    global_data['model']= model
    

class RecommendRequest(BaseModel):
    age: int
    sex: str
    liked_titles: List[str]

@app.post("/recommend")
def recommend(request: RecommendRequest):
    age = request.age
    sex = request.sex.lower()
    liked_titles = request.liked_titles

    logger.info("여기는 post로 입력받는 곳")
    logger.info(f"age: {age}")
    logger.info(f"sex: {sex}")
    logger.info(f"liked_titles: {liked_titles}")
    logger.info("-----------------------")

     # 글로벌 캐시에 저장된 데이터 사용
    contents = global_data["contents"]
    embeddings = global_data["embeddings"]
    intentions = global_data["intentions"] 
    experience = global_data['experience']
    preprocessing_contents = global_data['preprocessing_contents']
    if sex not in ['m', 'f']:
        raise HTTPException(status_code=400, detail="성별은 'm' 또는 'f'여야 합니다.")

    result = run_recommendation(age, sex, liked_titles,contents,preprocessing_contents,
                                embeddings,intentions,experience)
    print("result:",result)
    return result

# 헬스 체크
@app.get("/")
def root():
    return {"message": "FastAPI 추천 시스템 동작 중"}

##--- ott reconmmendation



# ======= 요청 모델 정의 =======
class RecommendationRequest(BaseModel):
    base_genres: List[str]
    detail_genres: List[str]
    age_group: str
    gender: str
    weekly_hours: float
    budget: float

# ======= API 엔드포인트 =======
@app.post("/ott_recommend")
def ott_recommend(req: RecommendationRequest):
    try:
        sel_df, plan, total_hours, total_cost = ott_recommendation_model(
            contents=global_data['contents'],
            prices=global_data['prices'],
            base_genres=req.base_genres,
            detail_genres=req.detail_genres,
            age_group=req.age_group,
            gender=req.gender,
            weekly_hours=req.weekly_hours,
            budget=req.budget,
            model=global_data['model']
        )
        

        recommendations = sel_df[['title', 'platform', 'score', 'watch_hours', 'genre', 'genre_detail']].to_dict(orient='records')
        print("recommendations:",plan)

        def convert_plan(plan_dict):
            converted = {}
            for key, (plan_name, price) in plan_dict.items():
                converted[key] = (plan_name, int(price))  # np.int64 -> int 변환
            return converted

        cleaned_plan = convert_plan(plan)



        for rec in recommendations:
            if 'platform' in rec and (rec['platform'] is None or (isinstance(rec['platform'], float) and math.isnan(rec['platform']))):
                rec['platform'] = ""

# 사용 예:
        

        return {
            "recommendations": recommendations,
            "total_estimated_watch_time": float(round(float(total_hours), 2)),  # 리스트 안의 float 처리
            "total_subscription_cost": int(total_cost),
            "subscription_plan": cleaned_plan
}


    except Exception as e:
        return {"error": str(e)}