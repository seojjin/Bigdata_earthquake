import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# In[0]: 데이터 불러오기
data = pd.read_csv("20210859박서진_지진데이터.csv", encoding='cp949')
data.info()

# In[1]: 데이터 분석하기

## 1. 연도별 지진 발생 횟수 알아보기
data['연도'] = pd.to_datetime(data['시간']).dt.year

# 연도별 지진 발생 횟수 집계
earthquake_counts = data['연도'].value_counts().sort_index()

# 연도별 지진 발생 횟수 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x=earthquake_counts.index, y=earthquake_counts.values)
plt.rc('font',family='Malgun Gothic')
plt.title('연도별 지진 발생 횟수')
plt.xlabel('연도')
plt.ylabel('지진 발생 횟수')
plt.xticks(rotation=45)
plt.show()

# In[2]: 가장 규모가 높은 지진 출력 
data = pd.read_csv("20210859박서진_지진데이터.csv", encoding='cp949')
max_magnitude = max(data['규모'])
max_magnitude_data = data[data['규모'] == max_magnitude]
print("가장 큰 지진 규모: ", max_magnitude)
print("================= 지진 정보 =================")
print(max_magnitude_data.to_string(index=False))


# In[3]: 규모 3.0이상의 지진 데이터 출력
data = pd.read_csv("20210859박서진_지진데이터.csv", encoding='cp949')
magnitude_3=data[data['규모']>=3.0]
print("규모 3.0이상의 지진 회수:", len(magnitude_3))
print("규모 3.0이상의 데이터")
print(magnitude_3.to_string(index=False))

# 규모 3.0이상의 지진 연도별 분석
magnitude_3['연도'] = pd.to_datetime(data['시간']).dt.year

# 연도별 지진 발생 횟수 집계
earthquake3_counts = magnitude_3['연도'].value_counts().sort_index()

# 연도별 지진 발생 횟수 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x=earthquake3_counts.index, y=earthquake3_counts.values)
plt.rc('font',family='Malgun Gothic')
plt.title('연도별 규모 3.0 이상의 지진 발생 횟수')
plt.xlabel('연도')
plt.ylabel('지진 발생 횟수')
plt.xticks(rotation=45)
plt.show()


# In[4]: 지진 발생이 많은 지역 추출
data['지역'] = data['위치'].str[:2]

# 지역별 지진 발생 횟수 계산
count_by_region = data.groupby('지역').size()
print("================지역별 지진 횟수================")
print(count_by_region)

# 시각화
plt.figure(figsize=(10, 6))
count_by_region.plot(kind='bar', color='skyblue')
plt.title('지역별 지진 발생 횟수')
plt.xlabel('지역')
plt.ylabel('지진 발생 횟수')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# In[5]: 경북 지역의 연도별 지진 발생 규모 시각화
data = pd.read_csv("20210859박서진_지진데이터.csv", encoding='cp949')

# 위치 정보에서 지역 추출
data['지역'] = data['위치'].str[:2]

# 발생일시를 datetime 형식으로 변환하여 연도와 월 추출
data['시간'] = pd.to_datetime(data['시간'])
data['연도'] = data['시간'].dt.year
data['월'] = data['시간'].dt.month

# 경북 지역 데이터 추출 (2016년 데이터만)
gyeongbuk_data = data[(data['지역'] == '경북') & (data['연도'] == 2016)]

# 월별 최대 규모 계산
max_magnitude_by_month = gyeongbuk_data.groupby('월')['규모'].max().reset_index()
print("====2016년도 경북 지역 월별 지진 최대 규모====")
print(max_magnitude_by_month.to_string(index=False))

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(max_magnitude_by_month['월'], max_magnitude_by_month['규모'], marker='o', linestyle='-', color='b')
plt.title('2016년 경북 지역 월별 최대 지진 규모')
plt.xlabel('월')
plt.ylabel('최대 지진 규모')
plt.grid(True)
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()


# In[6]: 지역별로 최대 규모의 지진 찾기
data = pd.read_csv("20210859박서진_지진데이터.csv", encoding='cp949')

# 위치 정보에서 지역 추출
data['지역'] = data['위치'].str[:2]
max_magnitude_by_region = data.groupby('지역')['규모'].max()

# 결과 출력
print("================지역별 최대 지진 규모================")
print(max_magnitude_by_region)


# 시각화
plt.figure(figsize=(10, 6))
max_magnitude_by_region.plot(kind='bar', color='pink')
plt.title('지역별 최대 지진 규모')
plt.xlabel('지역')
plt.ylabel('최대 지진 규모')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# In[7]: 회기: 선형 회기 분석을 사용하여 지진의 규모 예측하기
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report

# 데이터프레임 생성
data = pd.read_csv("20210859박서진_지진데이터.csv", encoding='cp949')

# 시간 정보를 연도와 월로 분리
data['시간'] = pd.to_datetime(data['시간'])
data['연도'] = data['시간'].dt.year
data['월'] = data['시간'].dt.month

# 불필요한 열 삭제
data = data.drop(columns=['시간'])

# 회귀 분석 - 지진의 규모 예측
X_reg = data[['진앙(km)', '위도', '경도', '연도', '월']]
y_reg = data['규모']

# 데이터 분할
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# 회귀 모델 학습
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

# 예측 및 평가
y_pred_reg = reg_model.predict(X_test_reg)
mae =  mean_absolute_error(y_test_reg, y_pred_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)

# 결과 출력
print(f'회귀 분석 예측 결과: {y_pred_reg}')
print(f'회귀 분석 MAE: {mae:.2f}')
print(f'회귀 분석 MSE: {mse:.2f}')


# In[8]: 분류: Randomforest를 활용하여 지진의 지역 예측하기(육지/해역)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score

# 데이터프레임 생성
data = pd.read_csv("20210859박서진_지진데이터.csv", encoding='cp949')

# 시간 정보를 연도와 월로 분리
data['시간'] = pd.to_datetime(data['시간'])
data['연도'] = data['시간'].dt.year
data['월'] = data['시간'].dt.month

# 불필요한 열 삭제
data = data.drop(columns=['시간'])

data['위치_라벨'] = data['위치'].apply(lambda x: 1 if x[-2:] == '해역' else 0)  # 해역을 1로, 육지를 0으로 라벨링

X_cls = data[['규모', '위도', '경도', '연도', '월']]
y_cls = data['위치_라벨']

# 데이터 분할
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)

# 분류 모델 학습
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_cls, y_train_cls)

# 예측 및 평가
y_pred_cls = clf_model.predict(X_test_cls)
accuracy = accuracy_score(y_test_cls, y_pred_cls)
precision = precision_score(y_test_cls, y_pred_cls)
recall = recall_score(y_test_cls, y_pred_cls)
f1 = f1_score(y_test_cls, y_pred_cls)
roc_auc = roc_auc_score(y_test_cls, y_pred_cls)
print(classification_report(y_test_cls, y_pred_cls, target_names=['육지', '해역']))

# 결과 출력
print(f'분류 예측 결과: {y_pred_cls}')
print(f'정확도 (Accuracy): {accuracy:.2f}')
print(f'정밀도 (Precision): {precision:.2f}')
print(f'재현율 (Recall): {recall:.2f}')
print(f'F1 스코어 (F1 Score): {f1:.2f}')
print(f'roc_auc 스코어 (roc_auc Score): {roc_auc:.2f}')












