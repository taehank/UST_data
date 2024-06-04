# LSA(Latent Semantic Analysis; 잠재의미분석) 실습
# modified code from https://wikidocs.net/24949

# 일단 SVD부터 알아보기

import numpy as np
A = np.array([[0,0,0,1,0,1,1,0,0],[0,0,0,1,1,0,1,0,0],[0,1,1,0,2,0,0,0,0],[1,0,0,0,0,0,0,1,1]])
print('DTM의 크기(shape) :', np.shape(A))

# Full SVD(Singular Value Decomposition; 특이값 분해) 수행
U, s, VT = np.linalg.svd(A, full_matrices=True)
print('행렬 U :', U.round(2))
print('행렬 U의 크기(shape) :', np.shape(U))

# U가 직교행렬인지 확인해보자
print(U @ U.transpose()) # 대각선으로는 1, 나머지 원소들은 0에 수렴 (매우 작은 값이 나옴)

# 대각행렬 S 확인
print('특이값 벡터 :', s.round(2)) # numpy의 linalg.svd()는 특이값 분해의 결과로 대각 행렬이 아닌 특이값의 리스트를 반환하므로, "행렬"이 아닌 "벡터"가 출력됨
print('특이값 벡터의 크기(shape) :', np.shape(s))

# s를 대각행렬로 바꿈
S = np.zeros((4,9)) # 대각행렬의 크기인 4x9의 행렬 생성 (일단은 임의로 0으로 채워넣음)
S[:4, :4] = np.diag(s) # 특이값을 대각행렬 S에 삽입
print('대각행렬 S :', S.round(2))
print('대각행렬의 크기(shape) :', np.shape(S))

# 직교행렬 V를 인쇄하고, V가 직교행렬인지 확인
print('직교행렬 VT :', VT.round(2))
print('직교행렬 VT의 크기 :', np.shape(VT))
print(VT @ VT.transpose())

# U X S X VT를 하면 원래의 행렬 A가 나오는지 확인
# numpy의 allclose()는 2개의 행렬이 동일하면 true를 리턴함
# np.dot은 두 행렬을 곱하는 함수
print(np.allclose(A, np.dot(np.dot(U,S), VT)))

# 절단된 SVD (Truncated SVD) (지금까지 수행한 것은 Full SVD였음)

# 대각행렬 S내의 특이값 중 상위 2개만 남기고 제거
S = S[:2, :2]
print('대각행렬 S :', S.round(2))

# 직교행렬 U도 2개의 "열"만 남기고 제거
U = U[:, :2]
print('행렬 U :', U.round(2))
# U의 각 행은 잠재 의미(=토픽)를 표현하기 위한 수치화된 각각의 문서 벡터 (즉, 잠재의미와 원래 문서와의 관계)

# 직교행렬 VT도 2개의 "행"만 남기고 제거 (V 관점에서는 2개의 열만 남김)
VT = VT[:2, :]
print('직교행렬 VT :', VT.round(2))
# VT의 각 열은 잠재 의미(=토픽)를 표현하기 위해 수치화된 각각의 단어 벡터 (즉, 잠재의미와 원래 단어와의 관계)

# U X S X VT 연산 결과는 A와는 다른 결과가 나옴(값이 손실됨), 이 값을 A_prime이라 함
A_prime = np.dot(np.dot(U, S), VT)
print(A)
print(A_prime.round(2))

# scikit-learn 실습을 통한 이해: 뉴스그룹
# scikit-learn에서 제공하는 Twenty Newsgroups(20개의 다른 주제를 가진 뉴스그룹 데이터) 사용
# LSA를 사용해서 문서의 수를 원하는 토픽의 수로 압축한 뒤, 각 토픽당 가장 중요한 단어 5개를 출력

import pandas as pd

from sklearn.datasets import fetch_20newsgroups
# pip install scikit-learn으로 사이킷런 설치할 것

import nltk
# nltk.download('stopwords') # stopwords를 다운로드 해줘야 함 (처음 한번만)
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
print('샘플의 수 :', len(documents))
print('2nd sample', documents[1]) # 두번째 샘플 출력 (샘플의 수가 11314개이므로, documents[0]부터 documents[11313]까지 부여됨
print(dataset.target_names) # 뉴스그룹 데이터 중 target_name에는 본래 이 뉴스그룹 데이터가 어떤 카테고리를 갖고 있었는지가 저장돼 있음

# 텍스트 전처리
# 정규표현식으로 정제, 길이가 짧은 단어 제거, 모든 알파벳을 소문자로 변경하여 단어의 개수를 줄임
news_df = pd.DataFrame({'document': documents})
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ", regex = True) # 특수문자 제거, 뒤에 'regex = True'를 꼭 붙여줘야 함 (정규표현식)
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3])) # 길이가 3 이하인 단어 제거
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower()) # 전체 단어에 대해 소문자로 변경

# 아까 출력했던 두번째 훈련용 샘플 출력 (정제 전 후 차이 비교)
print('2nd sample (cleaned): ', news_df['clean_doc'][1])

# 불용어 제거를 위해 토큰화
# NLTK로부터 불용어를 받아온다
stop_words = stopwords.words('english')
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split()) # 토큰화
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words]) # 불용어 제거

print('2nd sample (cleaned and tokenized): ', tokenized_doc[1])

# TF-IDF 행렬 만들기
# 역토큰화
detokenized_doc = []
for i in range(len(news_df)): # news_df의 길이(len)는 11314개
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)
news_df['clean_doc'] = detokenized_doc

print(news_df['clean_doc'][1]) # 역토큰화가 제대로 되었는지 확인, 다시 두번째 샘플 출력
# 사이킷런의 TfidVectorizer를 통해 단어 1,000개에 대한 TF-IDF 행렬 만들기
vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 1000, max_df = 0.5, smooth_idf = True) # 상위 1,000개 단어를 보존
X = vectorizer.fit_transform(news_df['clean_doc'])
print('TF-IDF 행렬의 크기 : ', X.shape) # TF-IDF 행렬의 크기를 확인, 11,314 x 1,000의 크기를 가진 TF-IDF 행렬이 생성되었음
# 11,314는 뉴스그룹 샘플의 수, 1,000개는 상위 1,000개 단어로 제한하기 위해 위에서 준 값임 (max_features 값)

# 토픽 모델링: TF-IDF 행렬을 분해 (Truncated SVD)
# 원래 뉴스그룹의 데이터가 20개의 카테고리를 갖고 있었으므로 20개의 토픽을 가졌다고 가정
# 토픽의 숫자는 n_components의 파라미터로 지정

svd_model = TruncatedSVD(n_components = 20, algorithm = 'randomized', n_iter = 100, random_state = 122)
svd_model.fit(X)
print(len(svd_model.components_)) # 위에서 n_components를 20개로 제한했으므로 20개가 나오는 것을 확인
print(np.shape(svd_model.components_)) # svd_model.components_는 앞에서 본 SVD 중 VT에 해당, 토픽의 수 x 단어의 수 크기를 가짐

terms = vectorizer.get_feature_names_out() # 1,000개의 단어 집합을 terms에 입력
print(terms) # 1,000개의 단어(=terms)를 확인해보자
def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(5)) for i in topic.argsort()[:-n - 1:-1]])
get_topics(svd_model.components_, terms) # 20개의 Topic별로, 각각 5개의 terms를 출력

