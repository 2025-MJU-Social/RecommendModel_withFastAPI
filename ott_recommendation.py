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

def calculate_age_similarity(user_age_group: str, content_age_group: str) -> float:
    """연령대 유사도 계산"""
    if user_age_group == content_age_group:
        return 1.0
    
    age_order = ["10대", "20대", "30대", "40대", "50대", "50대 이상"]
    
    try:
        user_idx = age_order.index(user_age_group)
        content_idx = age_order.index(content_age_group)
        
        # 거리 기반 유사도 (인접할수록 높은 점수)
        distance = abs(user_idx - content_idx)
        similarity = max(0, 1 - (distance * 0.2))  # 한 단계당 0.2씩 감소
        return similarity
    except ValueError:
        return 0.0

def calculate_gender_similarity(user_gender: str, content_gender: str) -> float:
    """성별 유사도 계산"""
    if user_gender.lower() == content_gender.lower():
        return 1.0
    else:
        return 0.3  # 다른 성별이어도 어느 정도 점수 부여

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

def normalize_platform_name(name):
    # 한글/영문, 대소문자, 특수문자, 공백 등 최대한 유연하게 변환
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("+", "plus")
        .replace("-", "")
        .replace("_", "")
        .replace("tv", "")
        .replace("(주)", "")
        .replace("쿠팡플레이", "coupangplay")
        .replace("왓챠", "watcha")
        .replace("웨이브", "wavve")
        .replace("티빙", "tving")
        .replace("디즈니플러스", "disneyplus")
        .replace("애플tv", "appletv")
        .replace("넷플릭스", "netflix")
        .replace("유플러스", "uplusmobile")
        .replace("u+모바일tv", "uplusmobile")
        .replace("uplus", "uplusmobile")
        .replace(".", "")
    )

def optimize_ott_subscription(sel_df, prices):
    """
    Set Cover Problem을 해결하여 최소 비용으로 모든 콘텐츠를 커버하는 플랫폼 조합 찾기
    """
    logger.info("OTT 구독 최적화 시작...")
    
    # 1. 각 콘텐츠가 어떤 플랫폼에서 상영되는지 매핑
    content_platforms = {}
    all_platforms = set()
    
    for idx, row in sel_df.iterrows():
        title = row['title']
        platforms = []
        if pd.notna(row.get('platform')):
            platforms = [p.strip() for p in str(row['platform']).split(',') if p.strip()]
        
        content_platforms[title] = platforms
        all_platforms.update(platforms)
    
    # 2. 각 플랫폼의 최저 요금제 가격 구하기
    platform_costs = {}
    platform_plans = {}
    
    for platform in all_platforms:
        opts = prices[prices['서비스명'] == platform]
        if not opts.empty:
            cheapest = opts.loc[opts['월 구독료(원)'].idxmin()]
            platform_costs[platform] = int(cheapest['월 구독료(원)'])
            platform_plans[platform] = (cheapest['요금제'], cheapest['월 구독료(원)'])
    
    # 콘텐츠가 없거나 플랫폼 정보가 없는 경우 처리
    if not content_platforms or not platform_costs:
        logger.warning("콘텐츠 또는 플랫폼 정보가 없습니다.")
        return {}, 0
    
    # 3. Greedy Set Cover 알고리즘으로 최적 조합 찾기
    uncovered_contents = set(content_platforms.keys())
    selected_platforms = {}
    total_cost = 0
    
    while uncovered_contents:
        best_platform = None
        best_ratio = float('inf')  # cost per new content covered
        best_new_contents = set()
        
        for platform in platform_costs:
            # 이 플랫폼으로 새로 커버할 수 있는 콘텐츠들
            new_contents = set()
            for content, content_plats in content_platforms.items():
                if content in uncovered_contents and platform in content_plats:
                    new_contents.add(content)
            
            if new_contents:  # 새로 커버할 콘텐츠가 있다면
                cost_per_content = platform_costs[platform] / len(new_contents)
                if cost_per_content < best_ratio:
                    best_ratio = cost_per_content
                    best_platform = platform
                    best_new_contents = new_contents
        
        if best_platform:
            selected_platforms[best_platform] = platform_plans[best_platform]
            total_cost += platform_costs[best_platform]
            uncovered_contents -= best_new_contents
            logger.info(f"선택: {best_platform} (비용: {platform_costs[best_platform]}원, 커버: {len(best_new_contents)}개 콘텐츠)")
        else:
            # 더 이상 커버할 수 있는 플랫폼이 없는 경우
            logger.warning(f"커버되지 않은 콘텐츠: {list(uncovered_contents)}")
            break
    
    logger.info(f"OTT 구독 최적화 완료 - 총 {len(selected_platforms)}개 플랫폼, {total_cost}원")

    return selected_platforms, total_cost

def filter_candidates(contents, base_genres, desired_min):
    # 기본 장르 필터링
    genre_mask = contents['genre'].apply(
        lambda x: any(genre in str(x).split(',') for genre in base_genres) if pd.notna(x) else False
    )
    candidates = contents[genre_mask].copy()
    if len(candidates) < desired_min:
        candidates = contents.copy()
    return candidates

def compute_scores(candidates, detail_genres, age_group, gender):
    # 장르 유사도
    genre_scores = [
        calculate_genre_similarity_optimized(detail_genres, row['genre_detail_list'])
        for _, row in candidates.iterrows()
    ]
    candidates['genre_similarity'] = genre_scores
    # 연령/성별 유사도
    age_scores = [
        calculate_age_similarity(age_group, row.get('age_group', ''))
        for _, row in candidates.iterrows()
    ]
    gender_scores = [
        calculate_gender_similarity(gender, row.get('gender', ''))
        for _, row in candidates.iterrows()
    ]
    candidates['age_similarity'] = age_scores
    candidates['gender_similarity'] = gender_scores
    # 러닝타임
    candidates['watch_hours'] = candidates.apply(estimate_runtime_hours, axis=1)

    # 랭킹 점수 추가 (순위가 높을수록 점수 높게)
    if 'rank' in candidates.columns:
        # rank가 1에 가까울수록 높은 점수 (역순 정규화)
        max_rank = candidates['rank'].max()
        candidates['rank_score'] = (max_rank - candidates['rank'] + 1) / max_rank
    else:
        candidates['rank_score'] = 0.5  # rank 정보가 없으면 중간값
    
    # 종합 점수 (가중치 재조정)
    candidates['combined_score'] = (
        0.35 * candidates['genre_similarity'] +    # 장르 유사도 
        0.15 * candidates['gender_similarity'] +    # 성별 유사도
        0.15 * (candidates['score'] / 100) +       # 콘텐츠 점수
        0.15 * candidates['rank_score'] +          # 랭킹 점수 추가
        0.1 * candidates['age_similarity'] +      # 연령 유사도 
        0.1 * (1 / (1 + candidates['watch_hours']))  # 시간 효율성
    )
    return candidates

def select_contents(candidates, max_hours, desired_min, desired_max):
    """플랫폼별 시청시간을 고려한 콘텐츠 선택"""
    candidates = candidates.sort_values('combined_score', ascending=False)
    candidates = candidates.drop_duplicates(subset=['title'])
    
    # 플랫폼별 시청시간 추적
    platform_hours = {}
    selected_contents = set()  # 이미 선택된 콘텐츠 추적
    selected = []
    
    for _, row in candidates.iterrows():
        title = row['title']
        platforms = []
        if pd.notna(row.get('platform')):
            platforms = [p.strip() for p in str(row['platform']).split(',') if p.strip()]
        
        watch_hours = row.get('watch_hours', 0)
        
        # 이미 선택된 콘텐츠면 스킵 (다른 플랫폼에서 이미 추가됨)
        if title in selected_contents:
            continue
            
        # 각 플랫폼에서 이 콘텐츠를 볼 수 있는지 확인
        can_add = False
        best_platform = None
        
        for platform in platforms:
            if platform not in platform_hours:
                platform_hours[platform] = 0
                
            # 이 플랫폼에서 시청시간 여유가 있는지 확인
            if platform_hours[platform] + watch_hours <= max_hours:
                can_add = True
                best_platform = platform
                break
        
        if can_add:
            selected.append(row)
            selected_contents.add(title)
            platform_hours[best_platform] += watch_hours
            
            if len(selected) >= desired_max:
                break
    
    # 최소 개수 보장 로직
    if len(selected) < desired_min:
        top = candidates.head(desired_min)
        selected = [row for _, row in top.iterrows()]
        # 이 경우는 플랫폼별 시간 제약을 무시하고 최소 개수 보장
    
    sel_df = pd.DataFrame(selected)
    total_hours = sum(row.get('watch_hours', 0) for row in selected)
    
    logger.info(f"플랫폼별 시청시간 분배: {platform_hours}")
    logger.info(f"중복 제거된 콘텐츠 수: {len(selected_contents)}")

    return sel_df, total_hours

def get_ott_plan_candidates(sel_df, prices, budget):
    """
    1. 추천 콘텐츠를 최대한 커버하는 ott+요금제 조합을 greedy하게 선정(중복 콘텐츠는 한 번만 커버)
    2. 선정된 ott 리스트에 한해서, 각 ott별로 예산 내 모든 요금제를 리스트로 추가(cover_count는 0 이상)
    즉, 최종적으로 선정된 ott만, 그 ott의 예산 내 모든 요금제를 보여준다.
    """
    # 1. greedy로 ott+요금제 조합 선정 (실제로 새롭게 커버할 수 있는 콘텐츠가 1개 이상인 경우만)
    platforms = set()
    for _, row in sel_df.iterrows():
        if pd.notna(row.get('platform')):
            for p in str(row['platform']).split(','):
                platforms.add(p.strip())
    # 후보 생성 (cover_set 포함)
    raw_candidates = []
    for platform in platforms:
        norm_platform = normalize_platform_name(platform)
        for _, plan_row in prices.iterrows():
            price_name = str(plan_row['서비스명'])
            if normalize_platform_name(price_name) == norm_platform:
                price = int(plan_row['월 구독료(원)'])
                if price <= budget:
                    plan_name = plan_row['요금제']
                    # 이 요금제로 커버 가능한 콘텐츠 set
                    cover_set = set()
                    for _, row in sel_df.iterrows():
                        if pd.notna(row.get('platform')):
                            content_platforms = [p.strip() for p in str(row['platform']).split(',')]
                            if platform in content_platforms:
                                cover_set.add(row['title'])
                    raw_candidates.append({
                        'platform': platform,
                        'plan_name': plan_name,
                        'price': price,
                        'cover_set': cover_set
                    })
    # greedy로 ott 선정
    used_contents = set()
    selected_ott = set()
    for cand in sorted(raw_candidates, key=lambda x: len(x['cover_set']), reverse=True):
        new_covers = cand['cover_set'] - used_contents
        if len(new_covers) > 0:
            selected_ott.add(cand['platform'])
            used_contents.update(new_covers)
    # 2. 선정된 ott에 한해서, 예산 내 모든 요금제 리스트업(cover_count=실제로 커버 가능한 콘텐츠 수)
    ott_plans = []
    for platform in selected_ott:
        norm_platform = normalize_platform_name(platform)
        for _, plan_row in prices.iterrows():
            price_name = str(plan_row['서비스명'])
            if normalize_platform_name(price_name) == norm_platform:
                price = int(plan_row['월 구독료(원)'])
                if price <= budget:
                    plan_name = plan_row['요금제']
                    # 이 요금제로 커버 가능한 콘텐츠 set
                    cover_set = set()
                    for _, row in sel_df.iterrows():
                        if pd.notna(row.get('platform')):
                            content_platforms = [p.strip() for p in str(row['platform']).split(',')]
                            if platform in content_platforms:
                                cover_set.add(row['title'])
                    ott_plans.append({
                        'platform': platform,
                        'plan_name': plan_name,
                        'price': price,
                        'cover_count': len(cover_set)
                    })
    # cover_count 내림차순 정렬
    ott_plans = sorted(ott_plans, key=lambda x: x['cover_count'], reverse=True)
    return ott_plans

def ott_recommendation_model(
        contents, 
        prices, 
        base_genres, 
        detail_genres, 
        age_group, gender, 
        weekly_hours, 
        budget, model):
    """
    최적화된 추천 시스템 함수 (사전 계산된 임베딩 사용 + 비용 최적화)
    """
    max_hours = weekly_hours * 4    # 월간 시청 시간
    desired_min, desired_max = 3, 8 # 추천 콘텐츠 개수
    logger.info(f"사용자의 월간 시청 시간: {max_hours:.1f}시간, 추천 콘텐츠 개수: {desired_min}~{desired_max}개")
    logger.info("추천 분석 시작...")

    # 1. 후보군 생성
    candidates = filter_candidates(contents, base_genres, desired_min)
    if candidates.empty:
        logger.warning('추천할 콘텐츠가 없습니다.')
        return pd.DataFrame(), {}, 0, 0

    # 2. 점수 계산
    candidates = compute_scores(candidates, detail_genres, age_group, gender)

    # 3. 콘텐츠 선택
    sel_df, total_hours = select_contents(candidates, max_hours, desired_min, desired_max)

    # 4. OTT+요금제 후보 생성 및 정렬
    ott_plan_candidates = get_ott_plan_candidates(sel_df, prices, budget)
    # 반환값 포맷 맞추기
    final_plan_with_cover = {}
    for idx, plan in enumerate(ott_plan_candidates):
        key = f"{plan['platform']}|{plan['plan_name']}"
        final_plan_with_cover[key] = {
            'platform': plan['platform'],
            'plan_name': plan['plan_name'],
            'price': plan['price'],
            'cover_count': plan['cover_count']
        }

    # 5. 반환값 정리
    # 추천 콘텐츠 정보 전체 로그 출력 (플랫폼, 장르, 시청시간, 점수, 순위만)
    logger.info("최종 추천 콘텐츠 리스트:")
    # 점수 내림차순 정렬
    if not sel_df.empty:
        if 'score' in sel_df.columns:
            sel_df = sel_df.sort_values('score', ascending=False)
        elif 'combined_score' in sel_df.columns:
            sel_df = sel_df.sort_values('combined_score', ascending=False)
        sel_df = sel_df.reset_index(drop=True)

    for _, row in sel_df.iterrows():
        platform = row.get('platform', '')
        genre = row.get('genre', '')
        genre_detail = row.get('genre_detail', '')
        watch_hours = row.get('watch_hours', 0)
        score = row.get('score', row.get('combined_score', 0))
        rank = row.get('rank', None)
        if rank is not None:
            logger.info(f"[{int(rank)}위] 플랫폼: {platform} | 장르: {genre} / {genre_detail} | 예상 시청 시간: {watch_hours}h | 점수: {score:.1f}")
        else:
            logger.info(f"플랫폼: {platform} | 장르: {genre} / {genre_detail} | 예상 시청 시간: {watch_hours}h | 점수: {score:.1f}")

    # 최종 out 결과 전체 로그
    logger.info("=== 모델 최종 반환값 ===")
    logger.info(f"추천 콘텐츠 개수: {len(sel_df)}")
    logger.info(f"최종 구독 플랜: {final_plan_with_cover}")
    logger.info(f"총 예상 시청시간: {total_hours}h, 총 구독비: 0원")

    return sel_df, final_plan_with_cover, total_hours, 0

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
        print("\n=== 최적화된 구독 플랜 ===")
        for p, v in plan.items():
            print(f"- {p}: {v['plan_name']} / {v['price']}원, 커버: {v['cover_count']}개 콘텐츠")
        print(f"총 구독비: {cost}원, 예상 시청시간: {hours:.1f}시간\n")
        
        print("=== 추천 콘텐츠 ===")
        for _, row in sel_df.iterrows():
            similarity_str = f"장르 유사도: {row.get('genre_similarity', 0):.2f}" if 'genre_similarity' in row else ""
            print(f"- {row.get('title', '')} ({similarity_str})")
    else:
        print("\n조건에 맞는 추천 콘텐츠가 없습니다.")