# modified code from https://csshark.tistory.com/110

import pandas as pd
from sklearn import datasets

iris = datasets.load_iris() # 붓꽃 데이터 가져오기
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns = iris.feature_names)
print(df)

# 스케일 맞춰주기 (스케일 다르면 왜곡)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df.loc[:,:] = scaler.fit_transform(df)
print(df)

# K-means 학습(학습인지는 모르겠지만...)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, random_state = 7) # 3개의 그룹(cluster)
kmeans.fit(df)
print(kmeans.labels_) # Clustering 결과가 labels_로 저장됨
print(len(kmeans.labels_)) # 150개의 데이터였으므로 clustering의 결과도 150개의 원소를 가진 벡터

distance_df = pd.DataFrame(kmeans.transform(df), columns=["c0", "c1", "c2"]) # transform은 150개의 데이터와 3개의 centroid와의 거리
print(distance_df)
# 원래의 데이터인 df 대신 새로운 데이터(validation용)를 넣으면 각 centroid와의 거리가 나오고, 이 값으로 validation 데이터를 분류할 수 있음

centroids = pd.DataFrame(kmeans.cluster_centers_, columns=df.columns) # cluster_centers_는 각 cluster의 centroid 정보를 담고 있음
centroids['cluster'] = ['Cluster {}'.format(i) for i in centroids.index]
print(centroids)

# 모델의 성능 평가(타당성 검증)는 본 코드의 출처(https://csshark.tistory.com/110)에 추가로 설명돼 있음