# 공통적으로 쓰는 라이브러리, 함수들

from konlpy.tag import Okt
# Okt는 한글을 토큰화해주는 라이브러리, Open Korean Text, 트위터에서 만든 오픈소스 한국어 처리기에서 파생
okt = Okt()
# JDK 다운로드 및 JAVA_HOME 변수설정 필요
# JDK 다운로드는 https://www.oracle.com/java/technologies/downloads/ 에서
# 환경변수 - 시스템 변수에서 새로 만들기: 변수명 JAVA_HOME 값 C:\Program Files\Java\jdk-22\bin (예)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
# nltk.download('stopwords') # 처음 실행할때는 stopwords를 다운받아야 함 (두번째부터는 주석처리해서 빼도 됨)
from nltk.corpus import stopwords
import pandas as pd
from math import log

# # One-hot vector 예제
#
# tokens = okt.morphs("나는 자연어 처리를 배우지만 너는 배우지 않는다") # okt.morphs: 형태소(morpheme) 단위로 토큰화
# print(tokens)
# word_to_index = {word : index for index, word in enumerate(tokens)} # 앞에 나오는 순서대로 0번부터 번호붙이기
# # 많이 나오는 순서대로 번호 붙이는걸 권장
# print('단어집합:', word_to_index)
#
# def one_hot_encoding(word, word_to_index):
#   one_hot_vector = [0]*(len(word_to_index)) # 단어수만큼 차원 설정하고 원소를 0으로 초기화
#   index = word_to_index[word] # input으로 주어진 word를 찾아서
#   one_hot_vector[index] = 1 # 해당 위치에만 0을 부여
#   return one_hot_vector
#
# print(one_hot_encoding('자연어', word_to_index))
# print(one_hot_encoding('는', word_to_index))
#
# # N-gram 예제
# # source from https://datasciencebeehive.tistory.com/114
#
# sentence = "나는 자연어 처리를 배우고 있습니다."
#
# # N-gram 계산
# unigrams = nltk.ngrams(sentence.split(), 1)
# bigrams = nltk.ngrams(sentence.split(), 2)
# trigrams = nltk.ngrams(sentence.split(), 3)
#
# # 결과 출력
# print("Unigrams:", unigrams)
# print("Bigrams:", bigrams)
# print("Trigrams:", trigrams)
#
# print("Unigrams:", list(unigrams))
# print("Bigrams:", list(bigrams))
# print("Trigrams:", list(trigrams))
#
# # Bag of Words 예제
#
# def build_bag_of_words(document): # 함수 선언
#   document = document.replace('.', '') # 온점 제거
#   tokenized_document = okt.morphs(document) # 형태소 분석
#   word_to_index = {}
#   bow = []
#   for word in tokenized_document:
#     if word not in word_to_index.keys(): # 처음 등장하는 형태소일 경우
#       word_to_index[word] = len(word_to_index) # word_to_index 하나 증가시키고
#       bow.insert(len(word_to_index) - 1, 1) # 1을 넣음
#     else: # 다시 등장하는 단어일 경우
#       index = word_to_index.get(word) # 그 형태소의 현재 카운트를 읽어서
#       bow[index] = bow[index] + 1 # 1을 더해서 다시 넣어줌
#   return word_to_index, bow # 형태소(중복 제거된...) 묶음과 카운트수를 리턴
#
# doc1 = "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다."
# vocab, bow = build_bag_of_words(doc1)
# print('vocabulary :', vocab)
# print('bag of words vector :', bow)
#
# doc2 = '소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.'
# vocab, bow = build_bag_of_words(doc2)
# print('vocabulary :', vocab)
# print('bag of words vector :', bow)
#
# doc3 = doc1 + ' ' + doc2
# vocab, bow = build_bag_of_words(doc3)
# print('vocabulary :', vocab)
# print('bag of words vector :', bow)
#
# # Bag of words: sklearn에서 제공하는 CountVectorizer 사용
#
# corpus = ['You know I want your love, because I love you.']
# vector = CountVectorizer()
# print('bag of words vector :', vector.fit_transform(corpus).toarray()) # corpus를 입력하고 각 단어의 빈도수를 기록
# print('vocabulary :', vector.vocabulary_) # 각 단어의 인덱스 보기 (I는 CountVectorizer의 전처리 과정에서 사라짐)
#
# # 한국어를 넣어보면
# corpus = ['달디달고 달디달고 달디단 밤양갱 밤양갱 내가 먹고 싶었던 건 달디단 밤양갱 밤양갱이야']
# print('bag of words vector :', vector.fit_transform(corpus).toarray())
# print('vocabulary :', vector.vocabulary_)
# # tokenizer를 지정해보면
# vector = CountVectorizer(tokenizer=okt.morphs) # 형태소 단위로 끊어서 토큰화
# print('bag of words vector :', vector.fit_transform(corpus).toarray())
# print('vocabulary :', vector.vocabulary_)
#
# # 불용어 처리 예제
#
# corpus = ["Family is not an important thing. It's everything."]
#
# vect = CountVectorizer(stop_words=["the", "a", "an", "is", "not"]) # 불용처리할 단어를 열거해 줌
# print('bag of words vector :', vect.fit_transform(corpus))
# print('bag of words vector :', vect.fit_transform(corpus).toarray())
# print('vocabulary :', vect.vocabulary_)
#
# vect = CountVectorizer(stop_words="english") # CountVectorizer에서 제공하는 자체 stopwords 사용
# print('bag of words vector :', vect.fit_transform(corpus).toarray())
# print('vocabulary :', vect.vocabulary_)
#
# stop_words = stopwords.words("english")
# vect = CountVectorizer(stop_words = stop_words) # NLTK에서 제공하는 불용어를 받아서 사용
# print('bag of words vector :', vect.fit_transform(corpus).toarray())
# print('vocabulary :', vect.vocabulary_)

# # TF-IDF 예제
#
# docs = [
#   'sweet like bubble gum bouncing like playing ball',
#   'sweet like bubble yum so smooth soft like a hug',
#   'you\'re my favorite flavor bubble gum',
#   'bubble bubble bubble gum'
# ]
# vocab = list(set(w for doc in docs for w in doc.split()))
# vocab.sort() # 알파벳 순으로 정렬, 한글 있다면 영어 먼저, 이후 한글
# print(vocab)
#
# N = len(docs) # 문서의 수
# def tf(t, d):
#   return d.count(t)
# def idf(t):
#   df = 0
#   for doc in docs:
#     df += t in doc
#   return log(N/(df+1)) # 자연로그
# def tfidf(t, d):
#   return tf(t,d)* idf(t)
#
# # TF 구하기
# result = []
# for i in range(N): # 각 문서에 대해 반복 (여기선 4번 반복)
#   result.append([])
#   d = docs[i] # 계산 대상 문서
#   for j in range(len(vocab)): # 아까 만들었던 vocab 내의 모든 단어에 대해
#     t = vocab[j]
#     result[-1].append(tf(t, d)) # 해당 문서 d에서 해당 단어 t가 등장하는 회수를 기록하여
# tf_ = pd.DataFrame(result, columns = vocab) # Data frame에 저장
# print(tf_) # 단어가 다른 단어안에 포함되는 경우는 두번 카운트되는 문제점이 있음 (예를 들어 a도 단어이지만 ball이 있으면 또다시 카운트됨)
#
# # IDF 구하기
# result = []
# for j in range(len(vocab)): # vocab 내의 모든 단어에 대해
#     t = vocab[j]
#     result.append(idf(t)) # IDF를 구해서 기록하여
# idf_ = pd.DataFrame(result, index=vocab, columns=["IDF"]) # Data frame에 저장
# print(idf_) # 모든 문서에 등장하는 bubble은 마이너스 값을 가짐 (ln(4/5)), 마이너스가 되면 이후 이를 처리할 때 문제가 생길 수 있으므로 0으로 만들어주는게 편함
#
# # TF-IDF 만들기
# result = []
# for i in range(N): # 전체 문서(여기서는 4건)에 대해
#   result.append([])
#   d = docs[i] # TF-IDF 작성대상 문서
#   for j in range(len(vocab)): # vocab 내의 모든 단어에 대해
#     t = vocab[j] # TF-IDF 작성대상 단어
#     result[-1].append(tfidf(t,d)) # TF-IDF 구해서 기록하고
# tfidf_ = pd.DataFrame(result, columns = vocab) # Data frame에 저장
# print(tfidf_)
#
# # Scikit-learn을 이용해서 편하게 작업해보기
# # DTM 만들기
# corpus = [
#     '사건은 다가와 Ah Oh Ay',
#     '거세게 커져가 Ah Oh Ay',
#     '질문은 계속돼 Ah Oh Ay',
#     '우린 어디서 왔나 Oh Ay'
# ]
# vector = CountVectorizer()
# print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도수를 계산해서 출력: DTM
# print(vector.vocabulary_) # 각 단어와 맵핑된 인덱스 출력
# # TF-IDF 만들기
# tfidfv = TfidfVectorizer().fit(corpus) # Scikit-learn의 TfidfVectorizer 사용 (맨 위에서 import 했음)
# print(tfidfv.transform(corpus).toarray())
# print(tfidfv.vocabulary_)
