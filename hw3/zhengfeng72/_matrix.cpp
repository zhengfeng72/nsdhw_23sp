#include "_matrix.hpp"

#include <cblas.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

Matrix::Matrix(Matrix const & other){
    m_ncol = other.m_ncol;
    m_nrow = other.m_nrow;

    size_t nelement = m_nrow * m_ncol;
    m_buffer = new double[nelement];
    std::copy(std::begin(other.m_buffer), std::end(other.m_buffer), std::begin(m_buffer));
}

Matrix::Matrix(Matrix && other){
    std::swap(m_ncol, other.m_ncol);
    std::swap(m_nrow, other.m_nrow);
    std::swap(m_buffer, other.m_buffer);
}


Matrix & Matrix::operator=(Matrix const & other){
    if(this == &other){ return *this; }

    m_ncol = other.m_ncol;
    m_nrow = other.m_nrow;

    size_t nelement = m_nrow * m_ncol;
    m_buffer = new double[nelement];
    std::copy(std::begin(other.m_buffer), std::end(other.m_buffer), std::begin(m_buffer));

    return *this;
}


Matrix & Matrix::operator=(Matrix && other){
    if(this == &other){ return *this; }

    std::swap(m_ncol, other.m_ncol);
    std::swap(m_nrow, other.m_nrow);
    std::swap(m_buffer, other.m_buffer);

    return *this;
}

Matrix::Matrix(size_t nrow, size_t ncol) : m_nrow(nrow), m_ncol(ncol){
    size_t nelement = nrow * ncol;
    m_buffer = new double[nelement];
}

Matrix::~Matrix(){
    if(m_buffer) delete[] m_buffer;
}

size_t Matrix::nrow() const { return m_nrow; }

size_t Matrix::ncol() const { return m_ncol; }

double* Matrix::buf() const { return m_buffer; }

double Matrix::operator() (size_t row, size_t col) const {
    return m_buffer[row*ncol + col];
}

double & Matrix::operator() (size_t row, size_t col){
    return m_buffer[row*ncol + col];
}

bool operator==(Matrix const & other){
    if((m_ncol!=other.ncol()) || (m_nrow!=other.nrow())) return false;

    for(size_t i=0; i<m_ncol; i++){
        for(size_t j=0; j<m_nrow; j++){
            if((*this)(i, j)!=other(i,j))  return false;
        }
    }
    return true;
}

Matrix multiply_naive(Matrix const &mA, Matrix const &mB){
    if(mA.ncol()!=mB.nrow()){
        throw std::out_of_range("column of Matrix A != row of Matrix B");
    }

    Matrix mul(mA.nrow(), mB.ncol());

    for(size_t i=0; i<mA.nrow(); i++){
        for(size_t j=0; j<mB.ncol(); j++){
            double val = 0;
            for(size_t k=0; k<mA.ncol(); k++){
                val += (mA(i,k) * mB(k, j));
            }

            mul(i, j) = val;
        }
    }
    return mul;
}

Matrix multiply_tile(Matrix const &mA, Matrix const &mB, size_t const t_size){
    if(mA.ncol()!=mB.nrow()){
        throw std::out_of_range("column of Matrix A != row of Matrix B");
    }

    Matrix mul(mA.nrow(), mB.ncol());

    for(size_t i=0; i<mA.nrow(); i+=t_size){
        for(size_t j=0; j<mB.ncol(); j+=t_size){
            for(size_t k=0; k<mA.ncol(); k+=t_size){
                // iterate over the elements of each tile
                for(size_t ti=i; ti<std::min(i+t_size, mA.nrow()); ti++){
                    for(size_t tj=j; tj<std::min(j+t_size, mB.ncol()); tj++){
                        double val = 0;
                        for(size_t tk=k; tk<std::min(k+t_size, mA.nrow()); tk++){
                            val += (mA(ti, tk)* mB(tk, tj));
                        }
                        mul(ti, tj) += val;
                    }
                }
            }
        }
    }
    return mul;
}

Matrix multiply_mkl(Matrix const &mA, Matrix const &mB){
    if(mA.ncol()!=mB.nrow()){
        throw std::out_of_range("column of Matrix A != row of Matrix B");
    }

    Matrix mul(mA.nrow(), mB.ncol());

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        mA.nrow(),
        mB.ncol(),
        mA.ncol(),
        1.0,
        mA.buf(),
        mA.ncol(),
        mB.buf(),
        mB.ncol(),
        0.0,
        mul.buf(),
        mul.ncol()
    );

    return mul;
}

PYBIND11_MODULE(matrix, m){
    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_tile", &multiply_tile);
    m.def("multiply_mkl", &multiply_mkl);

    pybind11::class_<Matrix>(m, "Matrix")
        .def(pybind11::init<size_t, size_t>())
        .def_readwrite("nrow", &Matrix::m_nrow)
        .def_readwrite("ncol", &Matrix::m_ncol)
        .def("__getitem__", [](Matrix &m, std::pair<size_t, size_t> i){
            return m(i.first, i.second);
        },pybind11::is_operator())
        .def("__setitem__", [](Matrix &m, std::pair<size_t, size_t> i, const double v){
            return m(i.first, i.second)=v;
        },pybind11::is_operator())
        .def(pybind11::self==pybind11::self)
        .def("assign", &Matrix::operator=);
}
