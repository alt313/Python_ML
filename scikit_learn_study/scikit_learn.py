# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 16:12:05 2021

@author: GOOD
"""

import sklearn
sklearn.__version__

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# 붓꽃 데이터 세트 로딩
iris = load_iris()
iris

# iris.data는 iris데이터 세트에서 피처(feature)만으로 된 데이터를 numpy로 가지고 있다.
iris_data = iris.data
iris_data

# iris.target은 붓꽃 데이터 세트에서 레이블(결정 값) 데이터를 numpy로 가지고 있다.
iris_label = iris.target
print('iris target값:', iris_label)
print('iris target명:', iris.target_names)
print()
# iris.target_names # 붓꽃 종류
# iris.feature_names # 컬럼명으로 쓰일 데이터

# 붓꽃 데이터 세트를 자세히 보기 위해 DataFrame으로 변환
iris_df = pd.DataFrame(data = iris_data, columns = iris.feature_names)
iris_df['label'] = iris_label
iris_df.head()

# train, test 데이터셋 분리
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_label, 
                                                    test_size = 0.2, 
                                                    random_state = 11)

# DecisionTreeClassifier 객체 생성
df_clf = DecisionTreeClassifier(random_state=11)

# 학습 수행
df_clf.fit(x_train, y_train)

# 학습이 완료된 DecisionTreeClassifier객체에서 테스트 데이터 세트로 예측 수행
pred = df_clf.predict(x_test)


# 예측 정확도 확인
from sklearn.metrics import accuracy_score
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))
print('예측 정확도:',round(accuracy_score(y_test, pred), 4))
print()

# iris클래스 확인
iris_data = load_iris()
print(type(iris_data))
print()

# iris 데이터 세트 key값 확인
keys = iris.keys()
print('붓꽃 데이터 세트의 키들:', keys)
print()

# iris가 반환하는 객체의 key가 가리키는 값 출력
print('feature_names의 type:', type(iris_data.feature_names))
print('feature_names의 shape:', len(iris_data.feature_names))
print(iris_data.feature_names)
print()

print('target_names의 type:', type(iris_data.target_names))
print('target_names의 shape:', iris_data.target_names.shape)
print(iris_data.target_names)
print()

print('data의 type:', type(iris_data.data))
print('data의 shape:', iris_data.data.shape)
print(iris_data.data)
print()

print('target의 type:', type(iris_data.target))
print('target의 shape:', iris_data.target.shape)
print(iris_data.target)
print()


# Model Selection 모듈 소개
# 학습/테스트 데이터 세트 분리 - train_test_split()
# 학습 데이터 세트로만 학습하고 예측하면 무엇이 문제인지 살펴보자
iris = load_iris()
dt_clf = DecisionTreeClassifier()
train_data = iris.data
train_label = iris.target
dt_clf.fit(train_data, train_label)

# 학습데이터 세트로 예측 수행
pred = dt_clf.predict(train_data)
print('예측 정확도:', accuracy_score(train_label, pred))
print()

# train_test_split() 데이터 세트를 30%로 학습을 70%로 분리 random_state = 121
iris = load_iris()
dt_clf = DecisionTreeClassifier()

train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, 
                                                    test_size = 0.3, 
                                                    random_state = 121)

# 테스트 데이터로 예측 정확도 측정
dt_clf.fit(train_x, train_y)
pred = dt_clf.predict(test_x)
print('예측정확도:', round(accuracy_score(test_y, pred), 4))
print()

# 교차검증

# K폴드 교차 검증

#  5개의 폴드세트로 분리하는 KFold객체를 생성
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state = 156)

# 5개의 폴드로 분리하는 KFold객체와 폴드 세트 별 정확도를 담을 리스트 객체 생성
kfold = KFold(n_splits = 5)

cv_accuracy = []
print('붓꽃 데이터 세트 크기:', features.shape[0])
print()

# 5개의 폴드 세트를 생성하는 KFold객체의 split()을 호출해 교차검증 수행시마다 
# 학습과 검증을 반복해 예측 정확도를 측정 그리고  split()이 어떤 값을 
# 실제로 반환하는지도 확인해보기 위해 검증데이터 세트의 인덱스도 추출
n_iter = 0

# KFold객체의 split()을 호출하면 폴드 별 학습용, 검증용 테스트의 로우 인덱스를 
# array로 변환
for train_index, test_index in kfold.split(features):
    # kfold.split()으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터 추출
    train_x, test_x = features[train_index], features[test_index]
    train_y, test_y = label[train_index], label[test_index]
    
    # 학습 및 예측
    dt_clf.fit(train_x, train_y)
    pred = dt_clf.predict(test_x)
    n_iter += 1
    
    # 반복 시마다 정확도 측정
    accuracys = np.round(accuracy_score(test_y, pred), 4)
    train_size = train_x.shape[0]
    test_size = test_x.shape[0]
    print(f'#{n_iter} 교차 검증 정확도 : {accuracys}, 학습 데이터 크기: {train_size}, 검증 데이터 크기: {test_size}')
    print(f'#{n_iter} 검증 세트 인덱스: {test_index}')
    print()
    cv_accuracy.append(accuracys)
    
# 개별 iteration별 정확도를 합하여 평균 정확도 계산
print('## 평균 검증 정확도 :', np.mean(cv_accuracy))
print()

# K폴드가 어떤 문제를 가지고 있는지 확인해 보고 
# 이를 사이킷런의 StratifiedKFold클래스를 이용해 개선
iris = load_iris()
iris_df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
iris_df['label'] = iris.target
iris_df.value_counts('label')

# 3개의 폴드 세트를 KFold로 생성 교차 검증시마다 생성되는 
# 학습/검증 레이블 데이터 값의 분포도를 확인
kfold = KFold(n_splits = 3)
n_iter = 0

for tr_idx, te_idx in kfold.split(iris_df):
    n_iter += 1
    label_tr = iris_df['label'].iloc[tr_idx]
    label_te = iris_df['label'].iloc[te_idx]
    print(f'## 교차 검증: {n_iter}')
    print('학습 레이블 데이터 분포:\n', label_tr.value_counts())
    print('검증 레이블 데이터 분포:\n', label_te.value_counts())

# StratifiedKFlod사용
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits = 3)
n_iter = 0

for tr_idx, te_idx in skf.split(iris_df, iris_df['label']):
    n_iter += 1
    label_tr = iris_df['label'].iloc[tr_idx]
    label_te = iris_df['label'].iloc[te_idx]
    print(f'## 교차 검증: {n_iter}')
    print('학습 레이블 데이터 분포:\n', label_tr.value_counts())
    print('검증 레이블 데이터 분포:\n', label_te.value_counts())
    
# 다음은 StratifiedKFold를 이용해 붓꽃 데이터를 분리해 보자 
df_clf = DecisionTreeClassifier(random_state = 156)

skfold = StratifiedKFold(n_splits = 3)
n_iter = 0
cv_accuracy = []


iris = load_iris()
features = iris.data
label = iris.target

# StratifiedKFold의 split() 호출시 반드시 레이블 데이터 세트도 추가 입력 필요
for tr_idx, te_idx in skfold.split(features, label):
    # split()으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터 추출
    tr_x, te_x = features[tr_idx], features[te_idx]
    tr_y, te_y = label[tr_idx], label[te_idx]
    # 학습 및 예측
    dt_clf.fit(tr_x, tr_y)
    pred = dt_clf.predict(te_x)
    
    # 반복 시마다 정확도 측정
    n_iter += 1
    accuracy = np.round(accuracy_score(te_y, pred), 4)
    tr_size = tr_x.shape[0]
    te_size = te_x.shape[0]
    print(f'#{n_iter} 교차 검증 정확도 : {accuracy}, 학습 데이터 크기: {tr_size}, 검증 데이터 크기: {te_size}')
    print(f'#{n_iter} 검증 세트 인덱스: {te_idx}')
    cv_accuracy.append(accuracy)
    print()
    
# 교차 검증별 정확도 및 평균 정확도 계산
print('## 교차 검증별 정확도:', np.round(cv_accuracy, 4))
print('## 평균 검증 정확도:', np.mean(cv_accuracy))
print()
    
# 교차 검증을 보다 간편하게 - cross_val_score()
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.datasets import load_iris

iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state = 156)

data = iris_data.data
label = iris_data.target

# 성능지표는 정확도(accuracy), 교차 검증 세트는 3개
scores = cross_val_score(dt_clf, data, label, scoring = 'accuracy', cv = 3)
print('교차 검증별 정확도', np.round(scores, 4))
print('평균 검증 정확도:', np.round(np.mean(scores), 4))
print()


# GridSearchCV - 교차 검증과 최적 하이퍼 파라미터 튜닝을 한 번에
grid_parameters = {'max_dapth': [1, 2, 3],
                   'min_samples_split': [2, 3]}

# 최적화 파라미터를 순차적으로 적용해 붓꽃데이터를 예측 분석하는데 GridSearchCV를 이용
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# 데이터를 로딩하고 학습 데이터와 테스트 데이터 분리
iris_data = load_iris()
tr_x, te_x, tr_t, te_y = train_test_split(iris_data.data, iris_data.target,
                                          test_size = 0.2,
                                          random_state = 121)
dtree = DecisionTreeClassifier()

# 파라미터를 딕셔너리 형태로 설정
paremeters = {'max_depth': [1, 2, 3], 'min_samples_split' : [2, 3]}


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# 데이터를 로딩하고 학습 데이터와 테스트 데이터 분리
iris_data = load_iris()
tr_x, te_x, tr_y, te_y = train_test_split(iris_data.data, iris_data.target,
                                          test_size = 0.2,
                                          random_state = 121)
dtree = DecisionTreeClassifier()

# 파라미터를 딕셔너리 형태로 설정
paremeters = {'max_depth': [1, 2, 3], 'min_samples_split' : [2, 3]}

import pandas as pd
# param_grid의 하이퍼 파라미터를 3개의 train, test set fold로 나누어 테스트 수행 설정
# refit = True가 default True이면 가장 좋은파라미터 설정으로 재학습 시킨다.
grid_dtree = GridSearchCV(dtree, param_grid = paremeters, cv = 3, refit = True)

# 붓꽃 학습 데이터로 param_grid의 하이퍼  파라미터를 순차적으로 학습/평가.
grid_dtree.fit(tr_x, tr_y)

# GridSearchCV 결과를 추출해 DataFrame으로 변환
scores_df = pd.DataFrame(grid_dtree.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']]
