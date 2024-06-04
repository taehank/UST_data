import pandas as pd
import nltk
# nltk.download('stopwords') # 처음 한번만
# nltk.download('punkt') # 처음 한번만
# nltk.download('wordnet') # 처음 한번만
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

data = pd.read_csv('./abcnews-date-text.csv')
print('뉴스 제목 개수 :', len(data))
print(data.head(5)) # 상위 5개만 출력해보자
data = data.head(10000) # 1백만개가 넘어서 오래걸림, 1만개만 사용하자

text = data[['headline_text']] # headline_text(뉴스 제목)만 저장
print(text.head(5))

text['headline_text'] = text.apply(lambda row: nltk.word_tokenize(row['headline_text']), axis=1) # 토큰화
stop_words = stopwords.words('english')
text['headline_text'] = text['headline_text'].apply(lambda x: [word for word in x if word not in (stop_words)]) # 불용어 제거
print(text.head(5))
text['headline_text'] = text['headline_text'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x]) # 표제어 추출 (3인칭, 과거 동사를 현재형 1인칭으로)
print(text.head(5))
tokenized_doc = text['headline_text'].apply(lambda x: [word for word in x if len(word) > 3]) # 길이가 3 이하인 단어 제거
print(tokenized_doc[:5])

# TF-IDF 행렬 만들기
detokenized_doc = [] # 역토큰화
for i in range(len(text)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)
text['headline_text'] = detokenized_doc # 역토큰화된 것을 재저장
print(text['headline_text'][:5]) # 역토큰화 잘 됐나 확인
vectorizer = TfidfVectorizer(stop_words='english', max_features= 1000) # 상위 1,000개 단어 보존
X = vectorizer.fit_transform(text['headline_text']) # TF-IDF 행렬 만들기
print('TF-IDF 행렬의 크기 :',X.shape) # TF-IDF 행렬의 크기 확인

# LDA 수행
lda_model = LatentDirichletAllocation(n_components=10, learning_method='online', random_state=777, max_iter=1)
lda_top = lda_model.fit_transform(X)
print(lda_model.components_.round(3))
print(lda_model.components_.shape)

terms = vectorizer.get_feature_names_out() # 1,000개의 단어들을 terms에 저장
def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n - 1:-1]])
get_topics(lda_model.components_,terms) # 토픽별로 상위 5개 단어(keyword)를 출력
# 좀전에 했던 LSA와 유사
