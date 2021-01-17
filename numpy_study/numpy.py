# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

array1 = np.array([1, 2, 3])
print('array1 type:', type(array1))
print('array1 array 형태:', array1.shape)

array2 = np.array([[1, 2, 3],
                   [2, 3, 4]])
print('array2 type:', type(array2))
print('array2 array 형태:', array2.shape)

array3 = np.array([[1, 2, 3]])
print('array3 type:', type(array3))
print('array3 array 형태:', array3.shape)
print()


# array 차원 출력
print('array1: {0}차원, array2: {1}차원, array3: {2}차원'.format(array1.ndim, array2.ndim, array3.ndim))
print()


# 데이터 타입
list1 = [1, 2, 3]
print(type(list1))
array1 = np.array(list1)
print(type(array1))
print(array1, array1.dtype)
print()

list2 = [1, 2, 'test']
print(type(list2))
array2 = np.array(list2)
print(type(array2))
print(array2, array2.dtype)
print()

list3 = [1, 2, 3.0]
print(type(list3))
array3 = np.array(list3)
print(type(array3))
print(array3, array3.dtype)
print()

array_int = np.array([1, 2, 3])
array_float = array_int.astype('float64')
print(array_float, array_float.dtype)
print()

array_int1 = array_float.astype('int32')
print(array_int1, array_int1.dtype)
print()

array_float1 = np.array([1.1, 2.1, 3.1])
array_int2 = array_float1.astype('int32')
print(array_int2, array_int2.dtype)
print()


# arange, zeros, ones

#arange()
sequence_array = np.arange(10)
print(sequence_array)
print(sequence_array.dtype, sequence_array.shape)
print()

# zeros(), ones()
zero_array = np.zeros((3, 2), dtype = 'int32')
print(zero_array)
print(zero_array.dtype, zero_array.shape)
print()

one_array = np.ones((3, 2))
print(one_array)
print(one_array.dtype, one_array.shape)
print()


# reshape()
array1 = np.arange(10)
print('array1:\n', array1)
print()

array2 = np.arange(10).reshape(2, 5)
print('array2:\n', array2)
print()

array3 = np.arange(10).reshape(5, 2)
print('array3:\n', array3)
print()

# -1을 일자로 줌
array1 = np.arange(10)
print(array1)
print()

array2 = array1.reshape(-1, 5)
print('array2 shape:', array2.shape)
print()

array3 = array1.reshape(5, -1)
print('array3 shape:', array3.shape)
print()

array1 = np.arange(8)
array3d = array1.reshape((2, 2, 2))
print('array3d:\n', array3d.tolist())
print()

# 3차원 ndarray를 2차원 ndarray로 변환
array5 = array3d.reshape(-1, 1)
print('array5:\n', array5.tolist())
print('array5 shape', array5.shape)
print()

# 1차원 ndarray를 2차원 ndarray로 변환
array6 = array1.reshape(-1, 1)
print('array6:\n', array6.tolist())
print('array6 shape', array6.shape)
print()


# 인덱싱

# 단일값 추출 
array1 = np.arange(1, 10)
# 1부터 9까지의 1차원 ndarray생성
print('array1:', array1)
value = array1[2]
print('value:', value)
print(type(value))
print()

# 2차원(3 x 3)ndarray에서 추출
array1d = np.arange(1, 10)
array2d = array1.reshape(-1, 3) # array1.reshape(3, 3)
print(array2d)
print('(row = 0 , col = 0) index 가리키는 값:', array2d[0, 0])
print('(row = 0 , col = 1) index 가리키는 값:', array2d[0, 1])
print('(row = 1 , col = 0) index 가리키는 값:', array2d[1, 0])
print('(row = 2 , col = 2) index 가리키는 값:', array2d[2, 2])
print()

# 슬라이싱
array1 = np.arange(1, 10)
array3 = array1[0:3]
print(array3)
print(type(array3))
print()

array1 = np.arange(1, 10)

array4 = array1[:3]
print(array4)
# :앞이 생략된다면 맨 처음부터 색인
array5 = array1[3:]
print(array5)
# :뒤를 생략하면 맨 마지막까지 색인
array6 = array1[:]
print(array6)
# :앞뒤 전부 생략하면 처음부터 끝까지 색인
print()

# 2차원 ndarray 슬라이싱 색인
array1d = np.arange(1, 10)
array2d = array1d.reshape(3, 3)
print('array2d:\n', array2d)

print('array2d[0:2, 0:2] \n', array2d[0:2, 0:2])
print('array2d[1:3, 0:3] \n', array2d[1:3, 0:3])
print('array2d[1:3, :] \n', array2d[1:3, :])
print('array2d[:, :] \n', array2d[:, :])
print('array2d[:2, 1:] \n', array2d[:2, 1:])
print('array2d[:2, 0] \n', array2d[:2, 0])
print()

# 펜시 인덱싱
array1d = np.arange(1, 10)
array2d = array1d.reshape(3, 3)
print(array2d)

array3 = array2d[[0, 1], 2]
print('array2d[[0, 1], 2] => ', array3.tolist())

array4 = array2d[[0, 1], 0:2]
print('array2d[[0, 1], 0:2] => ', array4.tolist())

array5 = array2d[[0, 1]]
print('array2d[[0, 1]] => ', array5.tolist())
print()

# 불린 인덱싱
array1d = np.arange(1, 10)
array3 = array1d[array1d > 5]
print('array1d > 5 불린 인덱싱 결과 값 :', array3)
print()


# 정렬(sort(), argsort())
org_array = np.array([3, 1, 9, 5])
print('원본 행렬:', org_array)

# np.sort()로 정렬
sort_array1 = np.sort(org_array)
print('np.sort() 호출 후 반환 된 정렬 행렬:', sort_array1)
print('np.sort() 호출 후 원본 행렬:', org_array)
print()
# ndarray.sort()로 정렬
sort_array2 = org_array.sort()
print('ndarray.srot() 호출 후 반환된 행렬:', sort_array2)
print('ndarray.srot() 호출 후 원본 행렬:', org_array)
print()

# 내림차순 정렬
sort_array1_desc = np.sort(org_array)[::-1]
print('내림차순으로 정렬:', sort_array1_desc)
print()

# 2차원 이상일 경우 정렬
array2d = np.array([[8, 12],
                    [7, 1]])
sort_array2d_axis0 = np.sort(array2d, axis = 0)
print('로우 방향으로 정렬\n', sort_array2d_axis0)
sort_array2d_axis1 = np.sort(array2d, axis = 1)
print('컬럼 방향으로 정렬\n', sort_array2d_axis1)
print()

# 정렬된 행렬의 인덱스 반환
org_array = np.array([3, 1, 9, 5])
sort_indices = np.argsort(org_array)
print(type(sort_indices))
print('행렬 정렬 시 원본 행렬의 인덱스:', sort_indices)
print()

# 성적순으로 학생이름 출력
name_array = np.array(['John', 'Mike', 'Sarah', 'Kate', 'Samuel'])
score_array =np.array([78, 95, 84, 98, 88])

sort_indices_asc = np.argsort(score_array)
print('성적 오름차순 정렬 시 score_array의 인덱스:', sort_indices_asc)
print('성적 오름차순으로 name_array의 이름 출력:', name_array[sort_indices_asc])
print()


# 선형대수 연산

# 행렬 내적(행렬 곱)
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

dot_product = np.dot(A, B) # 또는 A @ B
print('행렬 내적 결과:\n', dot_product)
print()

# 전치 행렬
A = np.array([[1, 2],
              [3, 4]])
transpose_mat = np.transpose(A) # 또는 A.T
print('A의 전치 행렬:\n', transpose_mat)