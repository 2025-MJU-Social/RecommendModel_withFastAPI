import pandas as pd
import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(content_path='./data/train_data.csv', price_path='./data/ott_price.csv'):
    """_summary_
    데이터 로드 함수
    콘텐츠 데이터와 OTT 가격 데이터를 로드하여 반환
    Args:
        content_path (str, optional): _description_. Defaults to './data/train_data.csv'.
        price_path (str, optional): _description_. Defaults to './data/ott_price.csv'.

    Returns:
        _type_: _description_
    """
    contents = pd.read_csv(content_path)
    prices = pd.read_csv(price_path)
    return contents, prices

def get_user_input(contents):
    """_summary_
    사용자에게 장르, 세부 장르, 연령대, 성별, 주간 시청 시간, 예산을 입력받는 함수
    장르와 세부 장르는 콤마로 구분된 문자열로 입력받고, 연령대는 문자열로 입력받음
    Args:
        contents (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    """_summary_
    러닝타임을 시간 단위로 변환하는 함수
    분 -> 시간, 회차는 1회차당 1시간으로 가정
    러닝타임이 없는 경우는 1시간으로 가정했고, 이 부분은 추후 수정이 필요함
    Args:
        row (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    """_summary_
    언어 모델을 로드하는 함수
    다국어 지원 모델을 사용하여 한국어 장르명에도 작동하도록 설정
    Returns:
        _type_: _description_
    """
    print("언어 모델 로드 중...")
    # 다국어 지원 모델 사용 (한국어 장르명에도 작동)
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)
    print("언어 모델 로드 완료")
    return model

def calculate_genre_similarity(model, user_genres, content_genres):
    """
    사용자 선호 장르와 콘텐츠 장르 간의 의미적 유사도 계산
    
    Args:
        model: 사전 훈련된 언어 모델
        user_genres: 사용자 선호 장르 리스트
        content_genres: 콘텐츠 장르 리스트
    
    Returns:
        float: 유사도 점수 (0~1)
    """
    if not user_genres or not content_genres:
        return 0.0
    
    # 텍스트를 임베딩 벡터로 변환
    user_embeddings = model.encode(user_genres)
    content_embeddings = model.encode(content_genres)
    
    # 각 사용자 장르와 각 콘텐츠 장르 간의 유사도 계산
    similarity_matrix = cosine_similarity(user_embeddings, content_embeddings)
    
    # 최대 유사도 접근법: 각 사용자 장르별 최대 유사도 계산 후 평균
    max_similarities = np.max(similarity_matrix, axis=1)
    return float(np.mean(max_similarities))

def add_genre_embeddings(contents, model):
    """
    콘텐츠 데이터프레임에 장르 임베딩을 추가.
    콘텐츠의 장르를 정리하고, 세부 장르를 리스트로 변환하여 임베딩을 계산
    Args:
        contents: 콘텐츠 데이터프레임
        model: 언어 모델
    
    Returns:
        DataFrame: 장르 임베딩이 추가된 데이터프레임
    """
    print("콘텐츠 장르 임베딩 계산 중...")
    
    # 장르 텍스트 정리
    contents['base_genre_clean'] = contents['genre'].fillna('')
    
    # 장르 상세 정보를 리스트로 변환
    contents['genre_detail_list'] = contents['genre_detail'].fillna('').apply(
        lambda x: [genre.strip() for genre in x.split(',')] if x else []
    )
    
    print("장르 임베딩 계산 완료")
    return contents

def ott_recommendation_model(
        contents, 
        prices, 
        base_genres, 
        detail_genres, 
        age_group, gender, 
        weekly_hours, 
        budget, model):
    """_summary_
    추천 시스템의 핵심 로직을 수행하는 함수
    사용자가 입력한 장르, 세부 장르, 연령대, 성별, 주간 시청 시간, 예산을 기반으로 콘텐츠를 추천
    현재는 장르 유사도 50%, 평점 30%, 시청 효율성 20%로 가중치를 설정하였고,
    예산은 OTT 서비스별로 구독비를 계산하여 총합을 반환하기만 하였음.

    따라서 가중치 수정과 예산을 고려한 콘텐츠 추천 로직을 추가할 필요가 있음.
    """
    max_hours = weekly_hours * 4    # 월간 시청 시간
    desired_min, desired_max = 3, 8 # 추천 콘텐츠 개수
    print(f"사용자의 월간 시청 시간: {max_hours:.1f}시간, \n추천 콘텐츠 개수: {desired_min}~{desired_max}개")

    print("추천 분석 중...")
    
    # 기본 필터링 (base 장르, 연령대, 성별)
    genre_mask = contents['genre'].apply(
        lambda x: any(genre in str(x).split(',') for genre in base_genres) if pd.notna(x) else False
    )
    age_gender_mask = (contents['age_group'] == age_group) & (contents['gender'] == gender)
    
    # 후보 데이터셋 생성 - 가장 제한적인 필터링
    candidates = contents[genre_mask & age_gender_mask].copy()
    
    # 필터 완화 단계 - 기존 후보군 유지하면서 추가
    original_count = len(candidates)
    if original_count < desired_min:
        print('⚠️ 콘텐츠 부족: 연령대·성별 필터 완화하여 추가')
        # 성별/연령대 필터 완화하고 장르 필터만 유지
        if original_count == 0:
            # candidates가 비어있을 경우 직접 장르 필터만 적용
            candidates = contents[genre_mask].copy()
        else:
            # candidates에 이미 항목이 있다면 추가 항목만 가져오기
            additional_candidates = contents[genre_mask & ~contents.index.isin(candidates.index)].copy()
            candidates = pd.concat([candidates, additional_candidates])
        print(f'  - {original_count}개 → {len(candidates)}개 후보 확보')
    
    original_count = len(candidates)
    if len(candidates) < desired_min:
        print('⚠️ 콘텐츠 부족: 모든 필터 완화하여 추가')
        # 모든 필터 완화
        if original_count == 0:
            # candidates가 비어있을 경우 모든 콘텐츠 사용
            candidates = contents.copy()
        else:
            # 기존 후보군에 없는 콘텐츠만 추가
            additional_candidates = contents[~contents.index.isin(candidates.index)].copy()
            candidates = pd.concat([candidates, additional_candidates])
        print(f'  - {original_count}개 → {len(candidates)}개 후보 확보')
    
    # 유효한 후보가 없는 경우 (여전히 빈 경우는 거의 없겠지만 안전장치)
    if candidates.empty:
        print('⚠️ 추천할 콘텐츠가 없습니다.')
        return pd.DataFrame(), {}, 0, 0
    
    # 나머지 코드는 동일...
    # 각 콘텐츠에 대해 장르 유사도 점수 계산
    print("세부 장르 유사도 계산 중...")
    genre_scores = []
    
    for _, row in candidates.iterrows():
        content_genres = row['genre_detail_list']
        # 장르 유사도 계산
        similarity_score = calculate_genre_similarity(model, detail_genres, content_genres)
        genre_scores.append(similarity_score)
    
    candidates['genre_similarity'] = genre_scores
    
    # 러닝타임 계산
    candidates['watch_hours'] = candidates.apply(estimate_runtime_hours, axis=1)
    
    # 종합 점수 계산 (장르 유사도 50%, 평점 30%, 시청 효율성 20%)
    candidates['combined_score'] = (
        0.5 * candidates['genre_similarity'] +  # 장르 유사도
        0.3 * (candidates['score'] / 100) +     # 평점 (정규화)
        0.2 * (1 / (1 + candidates['watch_hours']))  # 시청 효율성 (짧은 콘텐츠 선호)
    )
    
    # 종합 점수로 정렬 및 중복 제거
    candidates = candidates.sort_values('combined_score', ascending=False)
    candidates = candidates.drop_duplicates(subset=['title'])
    
    # 그리디 선택: 시청 시간 제약 내에서 점수가 높은 콘텐츠 선택
    selected = []
    total_hours = 0
    
    for _, row in candidates.iterrows():
        if total_hours + row.watch_hours > max_hours:
            continue
        selected.append(row)
        total_hours += row.watch_hours
        if len(selected) >= desired_max:
            break
    
    # 최소 갯수 부족 시 상위 효율 추천
    if len(selected) < desired_min:
        top = candidates.head(desired_min)  # desired_max에서 desired_min으로 변경
        selected = [row for _, row in top.iterrows()]  # itertuples() 대신 iterrows() 사용
        total_hours = sum(row.watch_hours for row in selected)
        print(f'⚠️ 최소 {desired_min}개 부족: 종합 점수 기준 상위 {desired_min}개 추천')
    
    sel_df = pd.DataFrame(selected)
    
    # 플랫폼 파싱
    plats = set()
    for entry in sel_df.platform.fillna('').tolist():
        for p in str(entry).split(','):
            name = p.strip()
            if name:
                plats.add(name)
    
    # 요금제 계산
    total_cost = 0
    plan = {}
    for p in plats:
        opts = prices[prices['서비스명'] == p]
        if opts.empty:
            continue
        cheapest = opts.loc[opts['월 구독료(원)'].idxmin()]
        plan[p] = (cheapest['요금제'], cheapest['월 구독료(원)'])
        total_cost += int(cheapest['월 구독료(원)'])
    
    if total_cost > budget:
        print(f"⚠️ 예산({int(budget)}원) 초과: 구독비 {total_cost}원")
    
    return sel_df, plan, float(total_hours), int(total_cost)


# 5. 실행
if __name__ == '__main__':
    print("=== 추천 시스템 시작 ===")
    
    # 언어 모델 로드
    model = load_language_model()
    
    print("1. 데이터 로드")
    contents, prices = load_data()
    
    # 콘텐츠에 장르 임베딩 추가
    contents = add_genre_embeddings(contents, model)
    
    print("2. 사용자 입력 받기")
    base_genres, detail_genres, age_group, gender, weekly_hours, budget = get_user_input(contents)
    
    print("3. 추천 실행")
    sel_df, plan, hours, cost = recommend(
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
            # 장르 유사도 점수 포함하여 출력
            similarity_str = f"장르 유사도: {row.get('genre_similarity', 0):.2f}" if 'genre_similarity' in row else ""
            print(f"- {row.title} | {row.genre} | {row.genre_detail} | 예상 시청: {row.watch_hours:.1f}시간 | 평점: {row.score} | {similarity_str}")
    else:
        print("\n조건에 맞는 추천 콘텐츠가 없습니다.")