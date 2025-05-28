import pandas as pd
import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, List, Tuple
import pickle
from pathlib import Path

# 로깅 설정
logger = logging.getLogger(__name__)

# 전역 캐시 변수
GENRE_EMBEDDINGS_CACHE = {}
CONTENT_EMBEDDINGS_CACHE = {}

def load_data(content_path='./data/train_data.csv', price_path='./data/ott_price.csv'):
    """데이터 로드 함수"""
    contents = pd.read_csv(content_path)
    prices = pd.read_csv(price_path)
    return contents, prices

def get_user_input(contents):
    """사용자 입력 받기 함수 (CLI용)"""
    # 이용 가능한 base genre 목록 출력 - 콤마로 분리된 장르 처리
    all_genres = []
    for genre_str in contents['genre'].dropna():
        all_genres.extend([g.strip() for g in genre_str.split(',')])
    
    # 중복 제거 및 정렬
    base_options = sorted(set(all_genres))

    print('지원 가능한 장르(콤마로 다중 선택 가능):', ', '.join(base_options))
    base_genres = [g.strip() for g in input('장르 선택(영화, 드라마, 예능 등, 여러 개 가능): ').split(',')]

    # 세부 장르 옵션
    detail_options = sorted({
        x.strip() for sub in contents['genre_detail'].dropna() for x in sub.split(',')
    })
    print('세부 장르 옵션 예시(콤마로 선택 가능):', ', '.join(detail_options[:10]), '...')
    detail_genres = [g.strip() for g in input('선호 세부 장르(콤마로 구분): ').split(',')]

    age_group = input('연령대(ex: 20대, 30대): ').strip()
    gender = input('성별(male/female): ').strip()
    weekly_hours = float(input('주간 OTT 시청 시간(시간): '))
    budget = float(input('한 달 예산(원): '))

    return base_genres, detail_genres, age_group, gender, weekly_hours, budget

def estimate_runtime_hours(row):
    """러닝타임을 시간 단위로 변환하는 함수"""
    if pd.notna(row.get('runtime', None)):
        try:
            mins = int(str(row.runtime).replace('분','').strip())
            return mins / 60
        except:
            pass
    if pd.notna(row.get('episodes', None)):
        try:
            eps = int(str(row.episodes).replace('부작','').strip())
            return eps * 1.0
        except:
            pass
    return 1.0

def load_language_model():
    """언어 모델을 로드하는 함수"""
    logger.info("언어 모델 로드 중...")
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)
    logger.info("언어 모델 로드 완료")
    return model

def precompute_genre_embeddings(model, contents):
    """
    모든 고유 장르에 대한 임베딩을 미리 계산하여 캐시에 저장
    """
    logger.info("장르 임베딩 사전 계산 중...")
    
    # 모든 고유 장르 수집
    all_genres = set()
    for genre_str in contents['genre_detail'].dropna():
        for genre in genre_str.split(','):
            genre = genre.strip()
            if genre:
                all_genres.add(genre)
    
    all_genres = list(all_genres)
    logger.info(f"총 {len(all_genres)}개 장르 임베딩 계산 중...")
    
    # 배치로 한번에 임베딩 계산 (효율성 증대)
    if all_genres:
        embeddings = model.encode(all_genres, show_progress_bar=False, batch_size=32)
        
        # 캐시에 저장
        global GENRE_EMBEDDINGS_CACHE
        GENRE_EMBEDDINGS_CACHE = {
            genre: embedding for genre, embedding in zip(all_genres, embeddings)
        }
    
    logger.info(f"장르 임베딩 사전 계산 완료: {len(GENRE_EMBEDDINGS_CACHE)}개")
    return GENRE_EMBEDDINGS_CACHE

def precompute_content_embeddings(contents):
    """
    모든 콘텐츠의 장르 조합에 대한 임베딩을 미리 계산
    """
    logger.info("콘텐츠 장르 조합 임베딩 사전 계산 중...")
    
    global CONTENT_EMBEDDINGS_CACHE
    
    for idx, row in contents.iterrows():
        genres = []
        if pd.notna(row.get('genre_detail')):
            genres = [g.strip() for g in row['genre_detail'].split(',') if g.strip()]
        
        if genres:
            # 장르 조합을 키로 사용
            genre_key = tuple(sorted(genres))
            
            # 이미 계산된 조합이 아닌 경우만 계산
            if genre_key not in CONTENT_EMBEDDINGS_CACHE:
                # 개별 장르 임베딩들의 평균 계산
                genre_embeddings = []
                for genre in genres:
                    if genre in GENRE_EMBEDDINGS_CACHE:
                        genre_embeddings.append(GENRE_EMBEDDINGS_CACHE[genre])
                
                if genre_embeddings:
                    avg_embedding = np.mean(genre_embeddings, axis=0)
                    CONTENT_EMBEDDINGS_CACHE[genre_key] = avg_embedding
    
    logger.info(f"콘텐츠 임베딩 사전 계산 완료: {len(CONTENT_EMBEDDINGS_CACHE)}개 조합")

def calculate_genre_similarity_optimized(user_genres: List[str], content_genres: List[str]) -> float:
    """
    최적화된 장르 유사도 계산 (사전 계산된 임베딩 사용)
    """
    if not user_genres or not content_genres:
        return 0.0
    
    # 사용자 장르 임베딩 가져오기
    user_embeddings = []
    for genre in user_genres:
        if genre in GENRE_EMBEDDINGS_CACHE:
            user_embeddings.append(GENRE_EMBEDDINGS_CACHE[genre])
    
    # 콘텐츠 장르 임베딩 가져오기
    content_embeddings = []
    for genre in content_genres:
        if genre in GENRE_EMBEDDINGS_CACHE:
            content_embeddings.append(GENRE_EMBEDDINGS_CACHE[genre])
    
    if not user_embeddings or not content_embeddings:
        return 0.0
    
    # 유사도 계산
    user_embeddings = np.array(user_embeddings)
    content_embeddings = np.array(content_embeddings)
    
    similarity_matrix = cosine_similarity(user_embeddings, content_embeddings)
    max_similarities = np.max(similarity_matrix, axis=1)
    
    return float(np.mean(max_similarities))

def add_genre_embeddings(contents, model):
    """
    콘텐츠 데이터프레임에 장르 정보를 추가하고 임베딩 사전 계산
    """
    logger.info("콘텐츠 데이터 전처리 중...")
    
    # 장르 텍스트 정리
    contents['base_genre_clean'] = contents['genre'].fillna('')
    
    # 장르 상세 정보를 리스트로 변환
    contents['genre_detail_list'] = contents['genre_detail'].fillna('').apply(
        lambda x: [genre.strip() for genre in x.split(',')] if x else []
    )
    
    # 임베딩 사전 계산
    precompute_genre_embeddings(model, contents)
    precompute_content_embeddings(contents)
    
    logger.info("콘텐츠 데이터 전처리 완료")
    return contents

def ott_recommendation_model(
        contents, 
        prices, 
        base_genres, 
        detail_genres, 
        age_group, gender, 
        weekly_hours, 
        budget, model):
    """
    최적화된 추천 시스템 함수 (사전 계산된 임베딩 사용)
    """
    max_hours = weekly_hours * 4    # 월간 시청 시간
    desired_min, desired_max = 3, 8 # 추천 콘텐츠 개수
    logger.info(f"사용자의 월간 시청 시간: {max_hours:.1f}시간, 추천 콘텐츠 개수: {desired_min}~{desired_max}개")

    logger.info("추천 분석 시작...")
    
    # 기본 필터링 (base 장르, 연령대, 성별)
    genre_mask = contents['genre'].apply(
        lambda x: any(genre in str(x).split(',') for genre in base_genres) if pd.notna(x) else False
    )
    age_gender_mask = (contents['age_group'] == age_group) & (contents['gender'] == gender)
    
    # 후보 데이터셋 생성
    candidates = contents[genre_mask & age_gender_mask].copy()
    
    # 필터 완화 로직
    original_count = len(candidates)
    if original_count < desired_min:
        logger.info('콘텐츠 부족: 필터 완화')
        if original_count == 0:
            candidates = contents[genre_mask].copy()
        else:
            additional_candidates = contents[genre_mask & ~contents.index.isin(candidates.index)].copy()
            candidates = pd.concat([candidates, additional_candidates])
    
    if len(candidates) < desired_min:
        logger.info('모든 필터 완화')
        if len(candidates) == 0:
            candidates = contents.copy()
        else:
            additional_candidates = contents[~contents.index.isin(candidates.index)].copy()
            candidates = pd.concat([candidates, additional_candidates])
    
    if candidates.empty:
        logger.warning('추천할 콘텐츠가 없습니다.')
        return pd.DataFrame(), {}, 0, 0
    
    # 🚀 최적화된 장르 유사도 계산 (사전 계산된 임베딩 사용)
    logger.info("장르 유사도 계산 중...")
    genre_scores = []
    
    for _, row in candidates.iterrows():
        content_genres = row['genre_detail_list']
        similarity_score = calculate_genre_similarity_optimized(detail_genres, content_genres)
        genre_scores.append(similarity_score)
    
    candidates['genre_similarity'] = genre_scores
    
    # 러닝타임 계산
    candidates['watch_hours'] = candidates.apply(estimate_runtime_hours, axis=1)
    
    # 종합 점수 계산
    candidates['combined_score'] = (
        0.5 * candidates['genre_similarity'] +
        0.3 * (candidates['score'] / 100) +
        0.2 * (1 / (1 + candidates['watch_hours']))
    )
    
    # 정렬 및 중복 제거
    candidates = candidates.sort_values('combined_score', ascending=False)
    candidates = candidates.drop_duplicates(subset=['title'])
    
    # 그리디 선택
    selected = []
    total_hours = 0
    
    for _, row in candidates.iterrows():
        if total_hours + row.watch_hours > max_hours:
            continue
        selected.append(row)
        total_hours += row.watch_hours
        if len(selected) >= desired_max:
            break
    
    # 최소 개수 보장
    if len(selected) < desired_min:
        top = candidates.head(desired_min)
        selected = [row for _, row in top.iterrows()]
        total_hours = sum(row.watch_hours for row in selected)
        logger.info(f'종합 점수 기준 상위 {desired_min}개 추천')
    
    sel_df = pd.DataFrame(selected)
    
    # 플랫폼 및 요금제 계산
    plats = set()
    for entry in sel_df.platform.fillna('').tolist():
        for p in str(entry).split(','):
            name = p.strip()
            if name:
                plats.add(name)
    
    total_cost = 0
    plan = {}
    for p in plats:
        opts = prices[prices['서비스명'] == p]
        if opts.empty:
            continue
        cheapest = opts.loc[opts['월 구독료(원)'].idxmin()]
        plan[p] = (cheapest['요금제'], cheapest['월 구독료(원)'])
        total_cost += int(cheapest['월 구독료(원)'])
    
    # === 예산 초과 OTT만 포함된 콘텐츠 제외 및 대체 추천 ===
    over_budget_ott = set()
    if total_cost > budget:
        running_cost = 0
        for p, (plan_name, price) in plan.items():
            running_cost += int(price)
            if running_cost > budget:
                over_budget_ott.add(p)

    def is_only_on_over_budget_ott(platforms, over_budget_ott, all_ott):
        platform_set = set([pp.strip() for pp in str(platforms).split(',') if pp.strip()])
        # 예산 내 OTT가 하나라도 있으면 False
        if platform_set - over_budget_ott:
            return False
        # 예산 초과 OTT만 있으면 True
        return bool(platform_set & over_budget_ott)

    if over_budget_ott:
        filtered = []
        for _, row in sel_df.iterrows():
            if not is_only_on_over_budget_ott(row['platform'], over_budget_ott, set(plan.keys())):
                filtered.append(row)
        # 대체 콘텐츠 추가 (예산 내 OTT에 포함된 것 중에서)
        if len(filtered) < desired_min:
            for _, row in candidates.iterrows():
                if not is_only_on_over_budget_ott(row['platform'], over_budget_ott, set(plan.keys())):
                    if row['title'] not in [r['title'] for r in filtered]:
                        filtered.append(row)
                    if len(filtered) >= desired_min:
                        break
        sel_df = pd.DataFrame(filtered)

    # =========================

    logger.info("추천 분석 완료")
    return sel_df, plan, float(total_hours), int(total_cost)

def prepare_ott_recommendation_data():
    """
    OTT 추천에 필요한 모든 데이터와 임베딩을 사전 준비
    """
    logger.info("OTT 추천 시스템 초기화 시작...")
    
    # 모델 로드
    model = load_language_model()
    
    # 데이터 로드
    contents, prices = load_data()
    
    # 임베딩 사전 계산 (여기서 모든 계산 완료)
    contents = add_genre_embeddings(contents, model)
    
    logger.info("OTT 추천 시스템 초기화 완료")
    return contents, prices, model

def clear_cache():
    """캐시 정리 함수"""
    global GENRE_EMBEDDINGS_CACHE, CONTENT_EMBEDDINGS_CACHE
    GENRE_EMBEDDINGS_CACHE.clear()
    CONTENT_EMBEDDINGS_CACHE.clear()
    logger.info("임베딩 캐시 정리 완료")

# CLI 실행용
if __name__ == '__main__':
    print("=== 추천 시스템 시작 ===")
    
    # 모든 데이터 및 임베딩 사전 준비
    contents, prices, model = prepare_ott_recommendation_data()
    
    print("2. 사용자 입력 받기")
    base_genres, detail_genres, age_group, gender, weekly_hours, budget = get_user_input(contents)
    
    print("3. 추천 실행")
    sel_df, plan, hours, cost = ott_recommendation_model(
        contents, prices, base_genres, detail_genres,
        age_group, gender, weekly_hours, budget, model
    )
    
    if not sel_df.empty:
        print("\n=== 추천 구독 플랜 ===")
        for p, (pkg, c) in plan.items():
            print(f"- {p}: {pkg} / {c}원")
        print(f"총 구독비: {cost}원, 예상 시청시간: {hours:.1f}시간\n")
        
        print("=== 추천 콘텐츠 ===")
        for _, row in sel_df.iterrows():
            similarity_str = f"장르 유사도: {row.get('genre_similarity', 0):.2f}" if 'genre_similarity' in row else ""
            print(f"- {row.get('title', '')} ({similarity_str})")
    else:
        print("\n조건에 맞는 추천 콘텐츠가 없습니다.")