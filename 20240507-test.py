import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy
from scipy import linalg as la
import scipy.stats as ss

# 한글출력
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# print(scipy.__version__)
#
# # 모평균이 알려져 있는 경우 분위수 구하기
# X = ss.norm(0, 1)
# alpha = np.array([0.1, 0.05, 0.01])
# z_alpha_half = X.ppf(1 - alpha/2)
# print(z_alpha_half.round(3))
#
# # 모평균이 알려져 있지 않은 경우 분위수 구하기 (t 분포)
# n = 10
# X = ss.t(df = n-1)
# alpha = np.array([0.1, 0.05, 0.01])
# t_alpha_half = X.ppf(1 - alpha/2)
# print(t_alpha_half.round(3))
#
# # [Bank] 데이터를 이용하여 잔액(balance)에 대한 모평균을 추정
#
# # [Bank] 데이터 읽기
# url = 'https://github.com/bong-ju-kang/kmu-mba-statistics-winter/raw/master/data/bank.zip'
# # 데이터에 대한 설명은 교재 p.231에 있음
# # 원본 데이터는 https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip
# df = pd.read_csv(url)
# print(df.shape) # 45211개가 full data set, 그 중 10% 샘플만 본 파일에 담겨있음
# print(df.columns)
#
# # 기본 통계량 계산하기
# x = df['balance'] # 변수 정의
# print(np.mean(x).round(2)) # 평균 출력
# print(np.std(x, ddof=1).round(2)) # 표준편차 출력
# # ddof: delta degrees of freedom, 독립적이지 않은 파라미터의 크기
# scale = ss.sem(x) # standard error of the mean 계산
# print(scale.round(2))
# print(np.std(x, ddof=1)/np.sqrt(len(x))) # 위와 같은 값
#
# # 신뢰수준 정의
# confidence = 1 - 0.05
# # 신뢰구간 출력
# confint = ss.t.interval(confidence=0.95, df=len(x)-1, loc=np.mean(x), scale=ss.sem(x))
# print(np.round(confint, 2))
# # n이 충분히 크므로, normal distribution을 이용해서 신뢰구간을 추정하면
# confint = ss.norm.interval(confidence=0.95, loc=np.mean(x), scale=ss.sem(x))
# print(np.round(confint, 2)) # 위와 비슷하게 나옴
#
# # 모평균 유의성 검정 예제 (교재 p.265)
# # [Bank] 데이터의 잔액 H0: mu = 1300, H1: mu > 1300의 가설을 검정
# null_mean = 1300
# res = ss.ttest_1samp(x, popmean = null_mean, alternative='two-sided') # 1samp: one sample test
# # popmean: 모집단의 평균, alternative='two-sided': 양측검정(디폴트값), greater 또는 less 사용가능
# print(res) # 검정통계량, 유의확률, 자유도 출력
# print((res.pvalue/2).round(4))
# # p값이 유의수준 0.05보다 작으므로 영가설 기각!

# # 모평균 비교(짝 비교) 예제
# # [Oatbran] 데이터: Oat bran을 섭취한 경우와 corn flake를 섭취한 경우의 serum cholesterol(혈청 콜레스테롤) 측정 결과
# url = 'https://github.com/bong-ju-kang/kmu-mba-statistics-winter/raw/master/data/oatbran.csv'
# df = pd.read_csv(url)
# print(df)
# # 평균 비교
# d = df['CORFLK'] - df['OATBRAN']
# res = ss.ttest_1samp(d, popmean=0) # 귀무가설: delta=0 (콘플레이크와 귀리겨 섭취시 콜레스테롤 차이는 없음)
# print(res)
# print((res.pvalue/2).round(4)) # 디폴트로 양측검정이므로, p-value를 2로 나눠서 유의수준보다 작은지 확인
# # 0.0026은 유의수준 0.05보다 작으므로, 영가설 기각, 혈청 콜레스테롤 차이가 있다고 판정

# Welch's t-test 예제
# [HFWS] 데이터를 이용하여 40대와 50대의 순자산 차이가 있는지 검증
url = 'https://github.com/bong-ju-kang/kmu-mba-statistics-winter/raw/master/data/MDIS_2018_HFWS.txt'
df = pd.read_csv(url, header = None)
print(df.head(3))
print(df.shape)
# 가구주 연령대: 22번째 열, 순자산: 109번째 열
# 40대 순자산 데이터 추출
x = df[df.iloc[:, 21] == 'G04'].iloc[:, 108]
print(x.shape)
print(x.mean().round(3))
print(x.std().round(3))
# 50대 순자산 데이터 추출
y = df[df.iloc[:, 21] == 'G05'].iloc[:, 108]
print(y.shape)
print(y.mean().round(3))
print(y.std().round(3))
# 순자산 차이를 Welch's t-test로 검정
res = ss.ttest_ind(x, y, equal_var = False)
# 두 집단간의 t-test 수행, equal_var=False로 분산이 다름을 알려줌: Welch's t-test를 수행하라는 얘기
print(res)
print(res.pvalue/2) # 양측 검정한 결과를 보여줬으므로, 단측검정으로 바꾸기 위해 p값을 2로 나눠 줌
# p-value가 0에 가까우므로, 두 집단의 순자산 차이는 유의미하게 존재한다고 판정

# 모분산이 동일하다고 가정하면
res = ss.ttest_ind(x, y, equal_var = True)
print(res) # 자유도는 11970 (x와 y 개수의 합 - 2)
print(res.pvalue/2)

# [HFWS] 데이터를 이용하여 순자산에 대한 모분산 추정
x = df.iloc[:, 108] # 순자산 추출
print(x.shape)
print(np.var(x)) # 표본의 분산 출력 (biased)
alpha = 0.5 # 유의수준 정의
n = len(x) # 표본의 크기 입력
s2 = np.var(x, ddof=1) # 표본의 분산(불편추정량, unbiased) 계산
print(s2)
upper = (n-1) * s2 / ss.chi2.ppf(alpha/2, n-1) # 상한값 계산
lower = (n-1) * s2 / ss.chi2.ppf(1-alpha/2, n-1) # 하한값 계산
print(lower, upper)
print(ss.chi2.ppf(alpha/2, n-1), ss.chi2.ppf(1-alpha/2, n-1))