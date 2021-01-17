# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 18:47:09 2021

@author: GOOD
"""

import pandas as pd
import numpy as np

# 판다스 시작

# 위에서 3개만 출력
titanic_df = pd.read_csv('titanic_train.csv')
titanic_df.head(3)
print()

# 타입과 전부출력
print('titanic 변수 type:', type(titanic_df))
titanic_df
print()

# DataFrame의 행과 열의 크기를 알아보는 방법
titanic_df.shape
print()

# 데이터 분포도(info)와 메타데이터(describe) 조회
titanic_df.info()
print()

titanic_df.describe()
print()

# value_counts메서드를 사용하여 Pclass컬럼 유형과 건수 확인
value_counts = titanic_df['Pclass'].value_counts()
value_counts
print()


# numpy ndarray, 리스트, 딕셔너리를, DataFrame으로 변환

# 1차원
col_name1 = ['col1']
list1 = [1, 2, 3]

array1 = np.array(list1)
print('array1 shape:', array1.shape)
print()
# 리스트를 이용해 DataFrame 생성
df_list1 = pd.DataFrame(list1, columns = col_name1)
print('1차원 리스트로 만든 DataFrame:\n', df_list1)
print()
# 넘파이 ndarray를 이용해 DataFrame생성.
df_array1 = pd.DataFrame(array1, columns = col_name1)
print('1차원 ndarray로 만든 DataFrame:\n', df_array1)
print()

# 2차원
# 2행 3열 형태의 리스트와 ndarray생성한 뒤 이를 DataFrame으로 변환
col_name2 = ['col1', 'col2', 'col3']

list2 = [[1, 2, 3],
         [11, 22, 33]]
array2 = np.array(list2)
print('array2 shape:', array2.shape)
print()
df_list2 = pd.DataFrame(list2, columns = col_name2)
print('2차원 리스트로 만든 DataFrame:\n', df_list2)
print()
df_array2 = pd.DataFrame(array2, columns = col_name2)
print('2차원 ndarray로 만든 DataFrame:\n', df_array2)
print()

# 딕셔너리를 DataFrame으로 변환
dict1 = {'col1': [1, 11],
         'col2': [2, 22],
         'col3': [3, 33]}
df_dict = pd.DataFrame(dict1)
print('딕셔너리로 만든 DataFrame:\n', df_dict)
print()


# DataFrame을 넘파이 ndarray, list, 딕셔너리로 변환

# DataFrame을 ndarray로 변환
array3 = df_dict.values
print('df_dict.values 타입:', type(array3), 'df_dict.values shape:', array3.shape)
print(array3)
print()

# DataFrame을 list로 변환
list3 = df_dict.values.tolist()
print('df_dict.values.tolist() 타입:', type(list3))
print(list3)
print()
# DataFrame을 딕셔너리로 변환
dict3 = df_dict.to_dict('list')
print("df_dict.to_dict('list') 타입:", type(dict3))
print(dict3)
print()


# DataFramedml 컬럼 데이터 세트 생성과 수정

# 새로운 컬럼 Age_0을 생성하고 0값 넣기
# 변경 전
titanic_df.head()
print()

# 변경 후
titanic_df['Age_0'] = 0
titanic_df.head()
print()

# 기존 컬럼 Series의 데이터를 이용해 새로운 컬럼 Series를 생성
titanic_df['Age_by_10'] = titanic_df['Age'] * 10
titanic_df['Family_No'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1
titanic_df.head(3)
print()


# DataFrame 데이터 삭제
# 삭제 전
titanic_df.head(3)
print()

# 삭제 후
titanic_drop_df = titanic_df.drop(columns = 'Age_0', axis = 1)
titanic_drop_df.head(3)
print()

# inplace = True로 데이터 삭제
drop_result = titanic_df.drop(['Age_0', 'Age_by_10', 'Family_No'], axis = 1, inplace = True)
print('inplace=True로 drop후 반환된 값:', drop_result)
titanic_df.head(3)
print()

# index 0,1,2 로우삭제
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 15)
print('#### before axis 0 drop ####')
print(titanic_df.head(3))
titanic_df.drop([0, 1, 2], axis = 0, inplace = True)
print()
print('#### after axis 0 drop ####')
print(titanic_df.head(3))
print()


# Index객체
titanic_df = pd.read_csv('titanic_train.csv')

# index객체 추출
indexes = titanic_df.index
print(indexes)
# index 객체를 실제 array로 변환
print('index 객체 array값:\n', indexes.values)
print()

# index단일 값 반환 및 슬라이싱
print(type(indexes.values))
print(indexes.values.shape)
print(indexes[:5].values)
print(indexes.values[:5])
print(indexes[6])
print()

# index변경(불가)
# indexes[0] = 5

# index는 연산에서 제외 된다
series_fair = titanic_df['Fare']
print('Fair Series max값:', series_fair.max())
print('Fair Series min값:', series_fair.min())
print('sum() Fair Series:', sum(series_fair))
print('Fair Series + 3:\n', (series_fair + 3).head(3))
print()

# reset_index()메서드로 인덱스 새롭게 할당
titanic_df.reset_index(inplace = True)
titanic_df.head(3)

# Series에 reset_index()적용
print('### before reset_index ###')
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
print('value_counts 객체 변수 타입:', type(value_counts))
print()
new_value_counts = value_counts.reset_index()
print('### after reset_index ###')
print(new_value_counts)
print('new_valeu_counts 객체 변수 타입:', type(new_value_counts))
print()


# 데이터 셀렉션 및 필터링

# 컬럼 명 지정
print('단일 컬럼 데이터 추출:\n', titanic_df['Pclass'].head(3))
print()
print('여러 컬럼의 데이터 추출:\n', titanic_df[['Survived', 'Pclass']].head(3))
print()
# print('[ ] 안에 숫자 index는 KeyError 오류 발생:\n', titanic_df[0])
# print()

# 인덱스 형태로 변환 가능한 표현식
titanic_df[:2]
print()

# 불린 인덱싱
titanic_df[titanic_df['Pclass'] == 3].head(3)
print()


data = {'Name': ['Chulmin', 'Eunkyung', 'Jinwoong', 'Soobeom'],
        'Yead': [2011, 2016, 2015, 2015],
        'Gender': ['Male', 'Female', 'Male', 'Male']}
data_df = pd.DataFrame(data, index = ['one', 'two', 'three', 'four'])
data_df


# iloc[]연산자
data_df.iloc[0, 0]
print()

# 위치인덱싱이 아닌 명칭입력
# data_df.iloc[0, 'Name']
# print()

# loc[] 연산자
data_df.loc['one', 'Name']
print()

# iloc와 loc의 슬라이싱 차이점
print('위치 기반 iloc slicing\n', data_df.iloc[0:1, 0])
print('명칭 기반 loc slicing\n', data_df.loc['one':'two', 'Name'])
print()

# 불린 인덱싱
# 60세 이상 추출
titanic_boolean = titanic_df.loc[titanic_df['Age'] >= 60, :]
print(type(titanic_boolean))
titanic_boolean
print()

# 60세 이상 나이와 이름 컬럼 위에서 3개만 추출
titanic_df.loc[titanic_df['Age'] >= 60, ['Age', 'Name']].head(3)
print()

# 60세 이상이고 선실 등급은 1등급이며 성별이 여성인 승객을 추출
titanic_df[(titanic_df['Age'] >= 60)  & (titanic_df['Pclass'] == 1) & (titanic_df['Sex'] == 'female')]
print()


# 정렬, Aggregation함수, Group By적용

# DataFrame, Series의 정렬 - sort_values()
titanic_df.sort_values(by = 'Name')
print()

# Pclass, Name컬럼 내림차순으로 정렬
titanic_df.sort_values(by = ['Pclass', 'Name'])
print()

# Aggregation적용
titanic_df.count()
print()

# Age, Fare컬럼 평균값 출력
titanic_df[['Age', 'Fare']].mean()
print()

# group by적용
titanic_group = titanic_df.groupby(by = 'Pclass')
print(type(titanic_group))
print()

# Pclass컬럼을  groupby하여 aggregation으로 count()
titanic_df.groupby(by = 'Pclass').count()
print()

#  Pclass를 groupby하고 PassengerId, Survived컬럼에만 count를 수행
titanic_df.groupby(by = 'Pclass')[['PassengerId', 'Survived']].count()
print()

# Pclass를 groupby하고 Age컬럼에 max와 min 수행
titanic_df.groupby(by = 'Pclass')['Age'].agg([max, min])
# 리스트형식으로 전달한다.
print()

# Pclass를 groupby하고 Age컬럼은 max, SibSp컬럼은  sum, Fare컬럼은 mean 수행
agg_format = {'Age' : 'max', 'SibSp' : 'sum', 'Fare' : 'mean'}
titanic_df.groupby(by = 'Pclass').agg(agg_format)
