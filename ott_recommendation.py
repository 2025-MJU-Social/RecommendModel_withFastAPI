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
    # 종합 점수
    candidates['combined_score'] = (
        0.4 * candidates['genre_similarity'] +
        0.2 * candidates['age_similarity'] +
        0.1 * candidates['gender_similarity'] +
        0.2 * (candidates['score'] / 100) +
        0.1 * (1 / (1 + candidates['watch_hours']))
    )
    return candidates

def select_contents(candidates, max_hours, desired_min, desired_max):
    candidates = candidates.sort_values('combined_score', ascending=False)
    candidates = candidates.drop_duplicates(subset=['title'])
    selected = []
    total_hours = 0
    for _, row in candidates.iterrows():
        if total_hours + row.watch_hours > max_hours:
            continue
        selected.append(row)
        total_hours += row.watch_hours
        if len(selected) >= desired_max:
            break
    if len(selected) < desired_min:
        top = candidates.head(desired_min)
        selected = [row for _, row in top.iterrows()]
        total_hours = sum(row.watch_hours for row in selected)
    sel_df = pd.DataFrame(selected)
    return sel_df, total_hours

def format_final_result(sel_df, final_plan, total_hours, total_cost):
    # 각 플랫폼별 커버 콘텐츠 개수 계산
    platform_cover_count = {}
    for platform in final_plan:
        count = 0
        for _, row in sel_df.iterrows():
            if pd.notna(row.get('platform')):
                content_platforms = [p.strip() for p in str(row['platform']).split(',')]
                if platform in content_platforms:
                    count += 1
        platform_cover_count[platform] = count
    # 최종 구독 플랜에 cover_count 추가
    final_plan_with_cover = {}
    for k, v in final_plan.items():
        final_plan_with_cover[k] = {
            'plan_name': v[0],
            'price': int(v[1]),
            'cover_count': platform_cover_count.get(k, 0)
        }
    return sel_df, final_plan_with_cover, float(total_hours), int(total_cost)

def handle_budget_excess(sel_df, candidates, optimized_plan, budget, desired_min):
    logger.info(f"최적화된 비용이 예산({budget}원)을 초과합니다. 예산 내 콘텐츠로 재추천...")
    budget_platforms = set()
    running_cost = 0
    platform_efficiency = []
    for platform, (plan_name, price) in optimized_plan.items():
        covered_count = 0
        for _, row in sel_df.iterrows():
            if pd.notna(row.get('platform')):
                content_platforms = [p.strip() for p in str(row['platform']).split(',')]
                if platform in content_platforms:
                    covered_count += 1
        if covered_count > 0:
            efficiency = covered_count / int(price)
            platform_efficiency.append((platform, int(price), efficiency, covered_count))
    platform_efficiency.sort(key=lambda x: x[2], reverse=True)
    for platform, price, efficiency, count in platform_efficiency:
        if running_cost + price <= budget:
            budget_platforms.add(platform)
            running_cost += price
    if budget_platforms:
        filtered_content = []
        for _, row in sel_df.iterrows():
            if pd.notna(row.get('platform')):
                content_platforms = set([p.strip() for p in str(row['platform']).split(',')])
                if content_platforms & budget_platforms:
                    filtered_content.append(row)
        if len(filtered_content) < desired_min:
            for _, row in candidates.iterrows():
                if pd.notna(row.get('platform')):
                    content_platforms = set([p.strip() for p in str(row['platform']).split(',')])
                    if content_platforms & budget_platforms:
                        if row['title'] not in [r['title'] for r in filtered_content]:
                            filtered_content.append(row)
                        if len(filtered_content) >= desired_min:
                            break
        if filtered_content:
            sel_df = pd.DataFrame(filtered_content)
            final_plan = {p: optimized_plan[p] for p in budget_platforms}
            final_cost = sum(int(plan[1]) for plan in final_plan.values())
            total_hours = sum(row.get('watch_hours', 1.0) for _, row in sel_df.iterrows())
            return sel_df, final_plan, total_hours
        else:
            logger.warning("예산 내에서 추천할 콘텐츠가 없습니다.")
            return None, None, None
    else:
        logger.warning("예산이 너무 적어 어떤 플랫폼도 구독할 수 없습니다.")
        return None, None, None

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

    # 4. 플랫폼 최적화
    optimized_plan, optimized_cost = optimize_ott_subscription(sel_df, prices)

    # 5. 예산 초과 처리 
    if optimized_cost > budget:
        result = handle_budget_excess(sel_df, candidates, optimized_plan, budget, desired_min)
        if result[0] is None:
            return pd.DataFrame(), {}, 0, 0
        sel_df, final_plan, total_hours = result
        final_cost = sum(int(plan[1]) for plan in final_plan.values())
    else:
        final_plan = optimized_plan
        final_cost = optimized_cost

    print(final_cost, budget)
    if final_cost > budget:
        warning_msg = "예산을 초과했습니다! 한 달에 하나씩 결제하는 것도 고려하세요."
        logger.warning(f"⚠️ {warning_msg}")
        global ment
        ment.append(warning_msg)

    # 6. 반환값 정리
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
    logger.info(f"최종 구독 플랜: {final_plan}")
    logger.info(f"총 예상 시청시간: {total_hours}h, 총 구독비: {final_cost}원")

    return format_final_result(sel_df, final_plan, total_hours, final_cost)

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