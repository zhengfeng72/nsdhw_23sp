import _matrix
import numpy as numpy
import pytest
import time

def test_muliply_native():
    m = 100
    n = 100
    k = 100

    m1 = _matrix.Matrix(m, k)
    m2 = _matrix.Matrix(k, n)

    for i in range(m):
        for j in range(k):
            m1[i ,j] = np.random.random()

    m3_native = _matrix.multiply_naive(m1, m2)
    m3_mkl = _matrix.multiply_mkl(m1, m2)
    assert m3_native.nrow == m3_mkl.nrow
    assert m3_native.ncol == m3_mkl.ncol
    for i in range(m):
        for j in range(n):
            assert abs(m3_native[i, j] - m3_mkl[i, k]) < 1e-4

def test_multiply_tile():
    m = 100
    n = 100
    k = 100

    m1 = _matrix.Matrix(m, k)
    m2 = _matrix.Matrix(k, n)

    for i in range(m):
        for j in range(k):
            m1[i ,j] = np.random.random()

    m3_mkl = _matrix.multiply_mkl(m1, m2)

    for t_size in [2, 4, 8, 16]:
        m3_tile = _matrix.multiply_tile(m1, m2, t_size)
        assert m3_native.nrow == m3_mkl.nrow
        assert m3_native.ncol == m3_mkl.ncol
        for i in range(m):
            for j in range(n):
                assert abs(m3_tile[i, j] - m3_mkl[i, k]) < 1e-4

def test_bendmark():
    size = 1000
    m1 = _matrix.Matrix(size, size)
    m2 = _matrix.Matrix(size, size)

    for i in range(size):
        for j in range(size):
            m1[i, j] = i*size + j +1
            m2[i, j] = i*size + j +1

    time_navie = 100000
    for i in range(5):
        start = time.time()
        _matrix.multiply_naive(m1, m2)
        end = time.time()
        time_navie = min(time_navie, end-start)

    time_tile = 100000
    for i in range(5):
        start = time.time()
        _matrix.multiply_tile(m1, m2, 4)
        end = time.time()
        time_tile = min(time_tile, end-start)

    file = open('performance.txt', 'w')
    file.write('naive:' + str(time_navie) + '\n')
    file.write('tile:' + str(time_tile) + '\n')
    file.close()