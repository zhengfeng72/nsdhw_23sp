#include <iostream>
#include <vector>

class Matrix{
public:
    Matrix()=default;
    Matrix(Matrix const &);
    Matrix(Matrix      &&);
    Matrix & operator=(Matrix const &);
    Matrix & operator=(Matrix      &&);
    Matrix(size_t nrow, size_t ncol);
    ~Matrix();
    size_t nrow() const;
    size_t ncol() const;
    double* buf() const;
    double   operator() (size_t row, size_t col) const;
    double & operator() (size_t row, size_t col);
    bool operator==(Matrix const &);
private:
    size_t m_nrow=0;
    size_t m_ncol=0;
    double * m_buffer=nullptr;
}

Matrix multiply_naive(Matrix const &mA, Matrix const &mB);
Matrix multiply_tile(Matrix const &mA, Matrix const &mB, size_t const t_size);
Matrix multiply_mkl(Matrix const &mA, Matrix const &mB);