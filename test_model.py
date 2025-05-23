import pandas as pd

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# 사용자 입력 테스팅
def input_data():
    age = int(input('나이?: '))
    sex = input('성별? (M/F): ')
    sex = sex.lower()
    titles_to_recommend = []
    while True:
        title = input('좋아하는 컨텐츠 (공백 입력 시 종료): ')
        if(not title): break
        titles_to_recommend.append(title)
    return age, sex, titles_to_recommend

# 모든 텍스트 데이터를 하나의 필드로 결합
def create_soup(row):
    soup = ' '.join(row['genre_detail'])
    return soup

# 임베딩 벡터 계산 함수
def get_embeddings(preprocessing_contents):
    texts = preprocessing_contents['soup'].tolist()

    # 모델과 토크나이저 불러오기
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # 입력 텍스트를 토큰화
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        # 모델 출력 얻기
        outputs = model(**inputs)
        # 토큰 임베딩과 어텐션 마스크
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        # Mean Pooling 계산
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        embeddings = sum_embeddings / sum_mask
        
    return embeddings

# 콘텐츠 데이터를 가져옴
def get_contents_data():
    contents = pd.read_csv('./data/fixed_contents.csv')
    contents = contents.fillna('')

    return contents
    
# 콘텐츠 데이터를 전처리
def preprocessing_contents_data(contents):
    # 다중값 컬럼을 리스트로 변환
    preprocessing_contents = contents.copy()
    multi_cols = ['genre_detail', 'director', 'platform', 'production', 'cast', 'country']
    for col in multi_cols:
        preprocessing_contents[col] = contents[col].apply(
            lambda x: sorted(x.split(', '))
        )
    preprocessing_contents['soup'] = preprocessing_contents.apply(create_soup, axis=1)

    return preprocessing_contents

# 장르 기반으로 추천 콘텐츠 리스트 결정
def genre_based_recommended_contents(contents, embeddings, titles):
    # 인덱스를 title로 설정
    temp = contents.reset_index()
    title_index = pd.Series(temp.index, index=temp['title']).drop_duplicates()

    # 유효한 제목들만 추출
    valid_indices = [title_index[title] for title in titles if title in title_index]

    if not valid_indices:
        return "입력된 제목 중 데이터셋에 존재하는 제목이 없습니다."

    # 선택한 제목들의 임베딩 벡터 추출
    selected_embeddings = embeddings[valid_indices]

    # 입력 제목들의 평균 벡터 계산
    mean_embedding = selected_embeddings.mean(dim=0)

    # 모든 콘텐츠와의 유사도 계산
    sim_scores = F.cosine_similarity(mean_embedding.unsqueeze(0), embeddings, dim=1)

    # 유사도 점수와 인덱스를 튜플로 묶고, 내림차순 정렬
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 입력 제목 자체를 제외하고 상위 5개 추출
    top_scores = [score for score in sim_scores if score[0] not in valid_indices][:5]

    # 추천 영화의 인덱스와 유사도 점수 추출
    movie_indices = [i[0] for i in top_scores]
    similarity_scores = [i[1].item() for i in top_scores]

    # 추천 영화 제목과 유사도 점수 반환
    recommendations = contents['title'].iloc[movie_indices]

    # 결과를 데이터프레임으로 묶어서 반환
    contents_based_recommendations = pd.DataFrame({
        'title': recommendations,
        'similarity score': similarity_scores
    })

    # 유사도에 따른 weight 추가
    contents_based_recommendations['weight'] = range(5,0,-1)

    return contents_based_recommendations

# 사용자 기반으로 추천 콘텐츠 리스트 결정
def user_based_recommended_contents(age, sex):
    user_data = None

    # 성별에 따른 데이터 로드
    if(sex=='m'):
        user_data = pd.read_csv('./data/daily_MALE_250514.csv')
    else:
        user_data = pd.read_csv('./data/daily_FEMALE_250514.csv')
        user_data = user_data.fillna('')

    # 연령에 따른 데이터 필터링
    if(age<20):
        user_data = user_data[user_data['age_group']=='10대'][['rank','title']]
    elif(age<30):
        user_data = user_data[user_data['age_group']=='20대'][['rank','title']]
    elif(age<40):
        user_data = user_data[user_data['age_group']=='30대'][['rank','title']]
    elif(age<50):
        user_data = user_data[user_data['age_group']=='40대'][['rank','title']]
    else:
        user_data = user_data[user_data['age_group']=='50대'][['rank','title']]

    # 유저에 따른 콘텐츠 5개 추천
    user_based_recommendations = user_data[:5]['title'].reset_index(drop=True).to_frame()
    user_based_recommendations['weight'] = range(5,0,-1)
    
    return user_based_recommendations

# 추천 콘텐츠 리스트 결합
def merge_recommended_contents(preprocessing_contents, contents_based_recommendations, user_based_recommendations):
    # 후보군을 위아래로 concat
    recommendations = pd.concat([user_based_recommendations, contents_based_recommendations], ignore_index=True, sort=False)

    # title이 겹치면 weight가 큰 컨텐츠를 남김
    idx = recommendations.groupby('title')['weight'].idxmax()
    recommendations = recommendations.loc[idx].reset_index(drop=True)

    # title, weight, platform 정보를 남김
    recommendations = recommendations[['title', 'weight']]
    recommendations = recommendations.merge(preprocessing_contents[["title", "platform"]], on="title", how="left")

    return recommendations

# OTT 서비스별 이용 의향 데이터를 가져옴
def get_ott_intension_data():
    # 필요한 정보만 필터링
    intentions = pd.read_csv('./data/OTT_유료서비스_계속_이용_의향__서비스별_20250413203427.csv', encoding='euc-kr')
    intentions.columns = intentions.iloc[0]
    intentions=intentions.loc[19:]

    # 컬럼 이름 통일
    intentions = intentions.rename(columns={"U+모바일 TV (%)": "U+모바일TV (%)"})

    # 50대 이상은 하나로 분류
    numeric_columns = intentions.columns[3:]
    intentions[numeric_columns] = intentions[numeric_columns].astype(float)
    sum_row = intentions.iloc[6:9, 3:].sum()
    intentions.loc[intentions['구분별(2)'] == '50대', intentions.columns[3:]] = sum_row.values
    intentions = intentions.iloc[:7]

    return intentions

# OTT 서비스별 이용 경험 데이터를 가져옴
def get_ott_experience_data():
    # 필요한 정보만 필터링
    experiences = pd.read_csv('./data/OTT_이용_경험_여부_서비스별_20250413203230.csv', encoding='euc-kr')
    experiences.columns = experiences.iloc[0]
    experiences=experiences.loc[19:]

    # 50대 이상은 하나로 분류
    numeric_columns = experiences.columns[3:]
    experiences[numeric_columns] = experiences[numeric_columns].astype(float)
    sum_row = experiences.iloc[6:9, 3:].sum()
    experiences.loc[experiences['구분별(2)'] == '50대', experiences.columns[3:]] = sum_row.values
    experiences = experiences.iloc[:7]

    return experiences

# 사용자 데이터에 따른 OTT별 점수 계산
def calculate_ott_score(age, sex, intentions, experiences):
    # 사용자 데이터에 맞는 row 추출
    intentions_age_row = None
    intentions_gender_row = None
    experiences_age_row = None
    experiences_gender_row = None

    # 성별에 따른 데이터 필터링
    if(sex=='m'):
        intentions_gender_row = intentions[intentions["구분별(2)"] == "남자"]
        experiences_gender_row = experiences[intentions["구분별(2)"] == "남자"]
    else:
        intentions_gender_row = intentions[intentions["구분별(2)"] == "여자"]
        experiences_gender_row = experiences[intentions["구분별(2)"] == "여자"]

    # 연령에 따른 데이터 필터링
    if(age<20):
        intentions_age_row = intentions[intentions["구분별(2)"] == "13~19세"]
        experiences_age_row = experiences[intentions["구분별(2)"] == "13~19세"]
    elif(age<30):
        intentions_age_row = intentions[intentions["구분별(2)"] == "20대"]
        experiences_age_row = experiences[intentions["구분별(2)"] == "20대"]
    elif(age<40):
        intentions_age_row = intentions[intentions["구분별(2)"] == "30대"]
        experiences_age_row = experiences[intentions["구분별(2)"] == "30대"]
    elif(age<50):
        intentions_age_row = intentions[intentions["구분별(2)"] == "40대"]
        experiences_age_row = experiences[intentions["구분별(2)"] == "40대"]
    else:
        intentions_age_row = intentions[intentions["구분별(2)"] == "50대"]
        experiences_age_row = experiences[intentions["구분별(2)"] == "50대"]

    # 점수를 저장할 딕셔너리
    score_dict = {}

    # OTT 서비스 리스트
    ott_services = ["넷플릭스", "웨이브", "티빙", "왓챠", "U+모바일TV", "디즈니플러스", "쿠팡플레이", "애플TV+"]

    # 가중치 설정
    weight_age = 0.5
    weight_gender = 0.5
    weight_experience = 0.6
    weight_intension = 0.4

    # scaling을 위한 변수
    max_score, min_score = 0, 1e9

    # 종합 점수 계산
    for ott in ott_services:
        # 의향 및 경험 데이터 추출
        intention_age = float(intentions_age_row[ott + " (%)"].values[0])
        experience_age = float(experiences_age_row[ott + " (%)"].values[0])
        intention_gender = float(intentions_gender_row[ott + " (%)"].values[0])
        experience_gender = float(experiences_gender_row[ott + " (%)"].values[0])

        # 각각의 종합 점수 계산
        score_age = weight_experience * experience_age + weight_intension * intention_age
        score_gender = weight_experience * experience_gender + weight_intension * intention_gender

        # 최종 종합 점수 계산
        final_score = (weight_age * score_age) + (weight_gender * score_gender)

        if(ott=='넷플릭스'):
            score_dict['Netflix']=final_score
        elif(ott=='웨이브'):
            score_dict['Wavve']=final_score
        elif(ott=='티빙'):
            score_dict['TVING']=final_score
        elif(ott=='왓챠'):
            score_dict['WATCHA']=final_score
        elif(ott=='U+모바일TV'):
            score_dict['U+모바일tv']=final_score
        elif(ott=='디즈니플러스'):
            score_dict['Disney+']=final_score
        elif(ott=='쿠팡플레이'):
            score_dict['coupang play']=final_score
        elif(ott=='애플TV+'):
            score_dict['Apple TV+']=final_score

        max_score = max(max_score, final_score)
        min_score = min(min_score, final_score)

    # scaling (score가 1보다 작으면 weight에 곱해졌을 때 값이 작아지므로 최소값 1을 더함)
    for k in score_dict:
        score_dict[k] = round((score_dict[k] - min_score) / (max_score - min_score) + 1, 2)

    return score_dict

# 콘텐츠를 제공하는 OTT 플랫폼을 기반으로 최종 OTT 점수를 계산하여 OTT 추천 순위 부여
def get_ott_recommendation_ranking(recommendations, score_dict):
    # sum of ((추천된 컨텐츠를 제공하는 ott별 종합 점수) * (추천된 컨텐츠의 weight))
    ott_score=[0]*8
    for i in range(recommendations.shape[0]):
        for ott in recommendations.iloc[i].iloc[2]:
            if(ott=='Netflix'): ott_score[0] += score_dict['Netflix']*float(recommendations.iloc[i].iloc[1])
            elif(ott=='Wavve'): ott_score[1] += score_dict['Wavve']*float(recommendations.iloc[i].iloc[1])
            elif(ott=='TVING'): ott_score[2] += score_dict['TVING']*float(recommendations.iloc[i].iloc[1])
            elif(ott=='WATCHA'): ott_score[3] += score_dict['WATCHA']*float(recommendations.iloc[i].iloc[1])
            elif(ott=='U+모바일tv'): ott_score[4] += score_dict['U+모바일tv']*float(recommendations.iloc[i].iloc[1])
            elif(ott=='Disney+'): ott_score[5] += score_dict['Disney+']*float(recommendations.iloc[i].iloc[1])
            elif(ott=='coupang play'): ott_score[6] += score_dict['coupang play']*float(recommendations.iloc[i].iloc[1])
            elif(ott=='Apple TV+'): ott_score[7] += score_dict['Apple TV+']*float(recommendations.iloc[i].iloc[1])

    # score 내림차순으로 정렬해서 ott 추천 순위를 보기 쉽게 만듬
    ott_score_df = pd.DataFrame({'OTT': [ott for ott in score_dict],
                                'score': ott_score})
    result = ott_score_df.sort_values(by="score", ascending=False).reset_index(drop=True)
    
    return result

def main():
    age, sex, titles_to_recommend = input_data()

    # 1차 후보군 생성: 좋아하는 콘텐츠 기반 (장르 기반 추천)
    contents = get_contents_data()
    preprocessing_contents = preprocessing_contents_data(contents)
    embeddings = get_embeddings(preprocessing_contents)
    contents_based_recommendations = genre_based_recommended_contents(contents, embeddings, titles_to_recommend)

    # 2차 후보군 생성: 사용자 통계 기반 (사용자의 연령/성별 기반 추천)
    user_based_recommendations = user_based_recommended_contents(age, sex)
    
    # 후보군 통합
    recommendations = merge_recommended_contents(preprocessing_contents, contents_based_recommendations, user_based_recommendations)

    # 사용자 정보에 따라 ott 별로 가중치 부여
    intentions = get_ott_intension_data()
    experiences = get_ott_experience_data()
    score_dict = calculate_ott_score(age, sex, intentions, experiences)

    # 추천된 컨텐츠를 기반한 최종적인 ott 점수 계산
    result = get_ott_recommendation_ranking(recommendations, score_dict)

    print(result)
    
if __name__ == '__main__':
    main()