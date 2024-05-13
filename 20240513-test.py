import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

import scipy
from scipy import linalg as la
import scipy.stats as ss
import scipy.special

import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

# 한글출력
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print(sm.__version__)

# 기초 자료: UCI ML DB의 housing 데이터 이용
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
df = pd.read_csv(url, sep='\\s+', header = None) # 구분자(separator)는 space
# \\ 대신 \ 쓰면 syntax warning 발생
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# 칼럼 정보 입력함 (원 데이터에는 칼럼정보 없음)
# 칼럼 설명은 https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset 에 있음
df.columns = df.columns.str.lower() # 모든 문자를 소문자로 바꿔줌
print(df.head(2))

# 모수 추정 예제
# x = df['rm']
# y = df['medv']
# x = sm.add_constant(x) # 절편항을 추가하기 위해 rm 데이터 칼럼 앞에 1로 이루어진 칼럼을 추가해 줌
# model = sm.OLS(y, x) # 모델을 정의: OLS(Ordinary Least Square)로 회귀모델을 정의
# fit = model.fit() # Fitting(적합)을 시킴
# print(fit.summary()) # fitting된 모델의 요약을 보여줌

# # 표준오차 예제
# # housing 데이터에서 크기가 100인 표본을 추출하여 선형회귀모형을 fitting하고, 잔차의 표준오차를 구함
# sample_size = 100
# np.random.seed(123)
# index = np.random.choice(np.arange(len(df)), size = sample_size) # 데이터 추출 (중복 허용)
# # np.random.choice의 인수로 replace = True를 주면 복원추출임 (디폴트값)
# x = df.loc[index, "rm"]
# y = df.loc[index, "medv"]
# x = sm.add_constant(x)
# dups = x.index.value_counts() # x.index는 샘플 100개의 index를 의미, x.index.value_counts()는 index를 세어서 빈도수를 계산
# print(dups[dups > 1]) # 1개 초과로 선택된 관측값의 번호를 보여줌
# fit = sm.OLS(y, x).fit()
# pred = fit.predict(sm.add_constant(df.loc[index, "rm"])) # 간단히 pred = fit.predict(x)로 해도 동일
# # y_hat을 계산하여 pred에 입력
# MSE = np.sum((y - pred)**2) / (sample_size - 2) # MSE 계산
# RMSE = np.sqrt(MSE) # RSE 계산
# print(RMSE.round(3))
# # 모델에서 제공하고 있는 함수를 사용하면
# print(np.sqrt(fit.mse_resid).round(3)) # 위 값과 동일한지 확인

# # 결정계수와 수정결정계수 예제
# # housing 데이터에서 크기가 100인 표본을 추출하고, 독립변수를 방의 개수(rm), 종속변수를 주택중위가격(medv)으로 하여 fitting할 때, 결정계수와 수정결정계수 구하기
# sample_size = 100
# np.random.seed(123)
# index = np.random.choice(np.arange(len(df)), size = sample_size)
# x = df.loc[index, "rm"]
# y = df.loc[index, "medv"]
# x = sm.add_constant(x)
# fit = sm.OLS(y, x).fit()
# pred = fit.predict(sm.add_constant(df.loc[index, "rm"])) # 여기까지는 위와 같음 (100개 뽑아서 회귀모델로 fitting, 결과값을 pred에 입력)
# SST = np.sum((y - np.mean(y))**2)
# SSE = np.sum((y - pred)**2)
# SSR = SST - SSE
# R2 = SSR/SST # 결정계수
# print('R^2: ', R2.round(3))
# adjR2 = 1 - (1-R2)*(sample_size-1)/(sample_size-2)
# print('R^2_adjustedad: ', adjR2.round(3)) # 열심히 계산한 수정결정계수
# print(fit.rsquared, fit.rsquared_adj) # 모델에서 제공하는 결정계수 및 수정결정계수 (위 값과 일치하는지 확인)

# # 모수에 대한 신뢰구간 구하기 예제
# # housing 자료에서 크기가 100인 표본을 추출, 종속변수를 주택중위가격(medv), 독립변수를 방의 개수(rm)로 함
# # 선형회귀모형으로 fitting시킬때 모수에 대한 신뢰구간을 구하기
# beta = fit.params[1] # 추정된 회귀선의 기울기를 beta에 입력
# alpha = 0.05 # 유의확률
# t = ss.t.ppf(1-alpha/2, df=sample_size-2) # t 분포의 분위수 계산, dof 설정 (n-2)
# sigma = np.sqrt(fit.mse_resid) # MSE에 루트를 씌워서 오차의 표준오차 계산
# x_rm = x.iloc[:, 1]
# c = np.sum((x_rm - np.mean(x_rm))**2) # c = 오차의 제곱합 (유인물에서 sigma (x - x_bar)^2로 표시한 부분)
# bse = sigma / np.sqrt(c) # 표준오차와 유사한 부분 계산
# conf = np.array([beta - t*bse, beta + t*bse]) # 신뢰구간 계산
# print('Confid. Interval: ', conf.round(3))

# # 다중선형회귀모형 예제
# # housing 자료에서, 크기가 100인 표본을 추출, 독립변수는 rm, age, lstat으로, 종속변수는 medv로 하여 선형회귀모형을 fitting
# xvars = ['rm', 'age', 'lstat']
# target = 'medv'
# sdf = df.loc[index, xvars+[target]] # sdf는 자료수 100개, 칼럼수 4개(rm, age, lstat, medv)인 데이터프레임
# print(sdf.shape)
# fmla = target + '~' + '+'.join(xvars) # 기호식을 만든다
# print('만들어진 기호식: ', fmla)
# fit = smf.ols(fmla, data = sdf).fit() # fitting 시킴
# print(fit.params.round(3)) # fitting한 결과 출력
#
# # Q-R 분해로 같은 결과를 얻어보자
# X = sdf[xvars]
# X = sm.add_constant(X).values
# y = sdf[target].values
# Q, R = la.qr(X.T@X)
# beta = la.inv(R) @ Q.T @ X.T @ y
# print(beta.round(3))
