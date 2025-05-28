import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os

# 한글 폰트 설정 (윈도우/맥/리눅스 호환)
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rc('font', family='Malgun Gothic')  # 윈도우
except:
    try:
        plt.rc('font', family='AppleGothic')  # 맥
    except:
        plt.rc('font', family='NanumGothic')  # 리눅스

# 데이터 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OTT_PRICE_PATH = os.path.join(BASE_DIR, 'data', 'ott_price.csv')
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data', 'train_data.csv')

# 1. OTT별 요금제별 가격 분포
ott_price = pd.read_csv(OTT_PRICE_PATH)
plt.figure(figsize=(12, 6))
sns.barplot(data=ott_price, x='서비스명', y='월 구독료(원)', hue='요금제')
plt.title('OTT별 요금제별 가격 분포')
plt.ylabel('월 구독료(원)')
plt.xlabel('OTT 서비스명')
plt.legend(title='요금제')
plt.tight_layout()
plt.show()

# 2. 장르별 콘텐츠 수(상위 10개)
train = pd.read_csv(TRAIN_DATA_PATH)
genre_counts = train['genre'].str.split(',').explode().str.strip().value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='viridis')
plt.title('장르별 콘텐츠 수(상위 10개)')
plt.ylabel('콘텐츠 수')
plt.xlabel('장르')
plt.tight_layout()
plt.show()

# 3. 러닝타임 분포(히스토그램)
if 'runtime' in train.columns:
    plt.figure(figsize=(8, 5))
    runtime = train['runtime'].dropna().astype(str).str.replace('분', '').astype(float)
    sns.histplot(runtime, bins=30, kde=True, color='skyblue')
    plt.title('러닝타임 분포')
    plt.xlabel('러닝타임(분)')
    plt.ylabel('콘텐츠 수')
    plt.tight_layout()
    plt.show()

# 4. 평점(score) 분포(히스토그램)
score_col = None
for col in ['score', '평점', 'rating']:
    if col in train.columns:
        score_col = col
        break
if score_col:
    plt.figure(figsize=(8, 5))
    sns.histplot(train[score_col].dropna(), bins=30, kde=True, color='salmon')
    plt.title('콘텐츠 평점 분포')
    plt.xlabel('평점')
    plt.ylabel('콘텐츠 수')
    plt.tight_layout()
    plt.show()

# 5. (선택) 연령대/성별별 선호 장르
if 'age_group' in train.columns and 'genre' in train.columns:
    age_genre = train.groupby('age_group')['genre'].apply(lambda x: ','.join(x)).str.split(',').explode().str.strip().value_counts().head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=age_genre.index, y=age_genre.values, palette='mako')
    plt.title('연령대별 선호 장르(상위 10개)')
    plt.ylabel('선호도')
    plt.xlabel('장르')
    plt.tight_layout()
    plt.show()

if 'gender' in train.columns and 'genre' in train.columns:
    gender_genre = train.groupby('gender')['genre'].apply(lambda x: ','.join(x)).str.split(',').explode().str.strip().value_counts().unstack().fillna(0)
    gender_genre.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('성별별 선호 장르')
    plt.ylabel('선호도')
    plt.xlabel('장르')
    plt.tight_layout()
    plt.show()
