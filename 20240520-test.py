import numpy as np
import matplotlib.pyplot as plt

# # logistic function 그래프 그려보기
# # Code from https://wikidocs.net/22881
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# x = np.arange(-5.0, 5.0, 0.1) # x축은 -5부터 5까지
# y = sigmoid(x)
# plt.plot(x, y, 'g') # 색깔은 Green
# plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가: (0,1)에서 (1,0)까지
# plt.title('Sigmoid Function')
# plt.show()
#
# # w값 변화시키기
# y1 = sigmoid(0.5*x)
# y2 = sigmoid(x)
# y3 = sigmoid(2*x)
# plt.plot(x, y1, 'r', linestyle='--') # w=0.5일 때
# plt.plot(x, y2, 'g') # w=1 일 때 (아까 그대로)
# plt.plot(x, y3, 'b', linestyle='--') # w=2 일 때
# plt.plot([0,0],[1.0,0.0], ':') # 가운데 선 그려주기
# plt.title('Sigmoid Function')
# plt.show()
# # w가 크면 가운데 부분 경사가 커짐
#
# # b값 변화시키기
# y1 = sigmoid(x+0.5)
# y2 = sigmoid(x+1)
# y3 = sigmoid(x+1.5)
# plt.plot(x, y1, 'r', linestyle='--') # b=0.5 일 때
# plt.plot(x, y2, 'g') # b=1 일 때
# plt.plot(x, y3, 'b', linestyle='--') # b=1.5 일 때
# plt.plot([0,0],[1.0,0.0], ':') # 가운데 선 그려주기
# plt.title('Sigmoid Function')
# plt.show()
# # b값은 그래프를 좌우로 이동시킴 (즉, 0.5가 되는 지점이 달라짐)
#
# # Logistic Regression 예제
# # Modified code originated from https://velog.io/@hyesoup/로지스틱회귀Logistic-Regression-예제
# # and https://itstory1592.tistory.com/10
# # 타이타닉호 승객 survival 예측
# import pandas as pd
# passengers = pd.read_csv('./train.csv')
# # 데이터 설명은 https://www.kaggle.com/c/titanic/data?select=train.csv에 있음
# print(passengers.shape)
# print(passengers.columns)
# print(passengers.head(10))
#
# # 생존 여부는 sex, age, pclass에 영향을 받을 것이라고 예상
#
# # 성별을 1과 0으로 맵핑
# passengers['sex'] = passengers['sex'].map({'female': 1, 'male': 0}) # 여자는 1, 남자는 0으로 바꿔줌
# print(passengers['sex'].head(10)) # 맵핑이 잘 됐는지 확인
#
# # 나이 결측치 평균으로 채워주기
# print(passengers.isnull().sum()) # 칼럼별로 결측치가 몇개나 있는지 확인
# passengers['age'].fillna(value = passengers['age'].mean(), inplace = True) # 평균으로 빠진 age 채워주기
# # na(not available) 데이터에 mean()으로 채워주기, inplace = True로 원본데이터도 변경하기
# # 원본을 변경하는지 copy본을 만드는지에 관해 warning 발생함
# print(passengers.isnull().sum()) # 칼럼별로 결측치가 몇개나 있는지 다시 확인
#
# # 1등석, 2등석, 3등석 변수 별도로 만들어주기
# dummies = pd.get_dummies(passengers['pclass'], dtype = int) # 여기까지하면 pclass에 있는 1, 2, 3이 각각 칼럼으로 되고(값은 0 또는 1) dummies라는 새로운 데이터프레임이 만들어짐
# # dtype = int 빼면 0과 1이 아닌 False와 True가 들어감
# del passengers['pclass'] # pclass 칼럼은 이제 필요없으니 지움
# passengers = pd.concat([passengers, dummies], axis = 1, join = 'inner') # 데이터프레임 합쳐줌
# # axis = 1: 열을 기준으로 합쳐줌, inner join은 교집합 병합, outer join(default값)은 합집합 병합 (여기선 index가 다 같으니 뭘해도 마찬가지)
# passengers.rename(columns = {1: 'FirstClass', 2: 'SecondClass', 3: 'EtcClass'}, inplace = True)
# print(passengers.head(10))
#
# # 데이터 세트 준비하기
# features = passengers[['sex', 'age', 'FirstClass', 'SecondClass']]
# survival = passengers[['survived']]
#
# # 학습용과 평가용으로 데이터 분리하기
# from sklearn.model_selection import train_test_split
# train_features, test_features, train_labels, test_labels = train_test_split(features, survival)
# # split할때 test_size = 0.2는 디폴트값임 (80%는 학습, 20%는 test)
# print(train_features) # train에 사용할 독립변수값
# print(test_features) # test에 사용할 독립변수값
# print(train_labels) # train에 사용할 결과값(여기선 survived 변수)
# print(test_labels) # test에 사용할 결과값
#
# # 데이터 정규화하기
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# train_features = scaler.fit_transform(train_features)
# test_features = scaler.fit_transform((test_features))
# # 평균 0, 표준편차 1로 스케일 변환
# print(train_features)
# print(test_features)
#
# # 모델 생성 및 test하기
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.fit(train_features, train_labels) # Fitting하기
# print(model.score(train_features, train_labels)) # train에 사용됐던 데이터들의 정확도를 출력
# print(model.score(test_features, test_labels)) # test용으로 분리해놨던 데이터로도 정확도 알아보기
# print(model.coef_) # 각 feature들에 대한 가중치를 출력해보기 -> 성별이 가장 큰 영향을 줌, 나이는 음의 관계(많을수록 생존못함)
# print(model.intercept_) # 상수항(절편) 출력
#
# # 새로운 데이터 넣어서 예측해보기
# Jack = np.array([0, 20, 0, 0])
# Rose = np.array([1, 17, 1, 0])
# TH = np.array([0, 50, 0, 1])
# sample_passengers = np.array([Jack, Rose, TH])
# sample_passengers = scaler.transform(sample_passengers) # 정규화도 해 줘야 함
# print(model.predict(sample_passengers)) # 생존했는가?
# print(model.predict_proba(sample_passengers)) # 각각의 승객에 대해, 0과 1이 될 확률이 얼마나 되는지
#
# 유방암 데이터로 Logistic Regression 해보기
# code from https://jimmy-ai.tistory.com/97
# 데이터에 대한 설명은 https://gomguard.tistory.com/52 또는 https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names

import pandas as pd
from sklearn import datasets

# 데이터 불러와서 학습대상 칼럼 설정
data = datasets.load_breast_cancer() # 유방암 데이터셋 가져오기
df = pd.DataFrame(data.data, columns = data.feature_names)
df = df[['mean radius', 'mean texture', 'mean area', 'mean symmetry']] # 일부 feature만 가져옴
df['target'] = data.target # 결과값(label) 열 지정 (1이면 양성 종양, 0이면 악성 종양)
print(df)

# scaling 해주기 (= 정규화)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df = scaler.fit_transform(df) # 정규화 진행 (minmax scaler 사용: 최소값 0, 최대값 1이 되도록 조정)
df = pd.DataFrame(df, columns = ['mean radius', 'mean texture', 'mean area', 'mean symmetry', 'target'])
# scaling 해줬으므로 데이터프레임 다시 만들어줌 (안해주면 index도 없고...)
print(df)

# training 데이터와 test 데이터를 분리
from sklearn.model_selection import train_test_split
X = df[['mean radius', 'mean texture', 'mean area', 'mean symmetry']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) # training 및 test용 데이터셋 분리

# 로지스틱 모형 적용해보기
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty = 'l2')
# 학습시켜나갈때 모델, l2는 Ridge, 일반적으로 사용되는 규제 유형, l1은 Lasso 규제 (필요시 더 찾아볼 것)
model.fit(X_train, y_train) # 로지스틱 회귀 모델 학습

# 로지스틱 모형 성능평가
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test) # 예측 결과 라벨(종속변수)을 y_pred에 저장
print(accuracy_score(y_pred, y_test)) # 정확도 측정 # random 요소로 인해 매번 결과가 달라짐
# 정확도는 90% 정도임

# 성능평가
# modified code from https://teddylee777.github.io/scikit-learn/scikit-learn-logistic-regression-and-metrics/

# confusion matrix 만들고 heatmap 보여주기
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)
# confusion matrix 보는법
# 첫번째 행: 실제 y값이 0일 경우 [True Negative, False Positive(=Type 1 error)]
# 두번째 행: 실제 y값이 1일 경우 [False Positive(=Type 2 error), True Negative]
import seaborn as sns # heatmap 그리기 위해 seaborn import함
sns.heatmap(cm, annot = True, annot_kws = {"size": 20}, cmap = 'YlOrBr') # heatmap 그리기
plt.xlabel('Predicted', fontsize = 20)
plt.ylabel('Actual', fontsize = 20)
plt.show() # heatmap 보이기

matrix = classification_report(y_test, y_pred)
print(matrix) # 성능평가 결과 보여주기
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

# 성능평가: 정확도(Accuracy) = 전체 건수 중 제대로 예측된 비율
print('Accuracy: ', (TP + TN) / (TN + FP + FN + TP))

# 성능평가: 정밀도(Precision) = Positive로 예측한 전체 건수에서 실제 Positive의 비율
print('Precision: ', TP / (FP + TP)) # Y=0인 경우도 precision 측정 가능, classification report에 나옴

# 성능평가: 민감도(Recall) = 실제 Positive 중 제대로 Positive로 예측한 비율
print('Recall: ', TP / (FN + TP)) # Y=1인 경우도 Recall 측정 가능

# 그 외: F1 Score는 Recall과 Precision의 균형을 나타내는 수치임