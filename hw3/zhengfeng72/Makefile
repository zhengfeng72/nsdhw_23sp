CC     = g++
CFLAGS = -O3 -Wall -std=c++11 -shared -fPIC -m64
PY     = python3 -m pybind11 --includes
SUFFIX = python3-config --extension-suffix

MKL_INC =  /usr/include/mkl
MKL_DIR = /usr/lib/x86_64-linux-gnu
MKL_LIB = ${MKL_DIR}/libmkl_def.so \
			${MKL_DIR}/libmkl_avx2.so \
			${MKL_DIR}/libmkl_core.so \
			${MKL_DIR}/libmkl_intel_lp64.so \
			${MKL_DIR}/libmkl_sequential.so
MKL = -I ${MKL_INC} ${MKL_LIB} -lblas -lmkl_intel_lp64 -lmk_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

.PHONY: clean test

all: _matrix.*.so

_matrix.*.so: _matrix.cpp
	$(CC) $(CFLAGS)  `$(PY)` $< -o _matrix`$(SUFFIX)` -I /usr/include/python3.8 ${MKL}
	
clean:
	rm -rf *.so ./__pycache__ ./.pytest_cache

test:
	python3 -m pytest test_matrix.py