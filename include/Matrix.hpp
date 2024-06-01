#ifndef MATRIX_H
#define MATRIX_H

#include <cstdio>
#include <cstring>
#include <random>

#ifndef SAFETY_CHECKS
#define SAFETY_CHECKS 0
#endif

class Matrix {
    private:
        double* data = nullptr;

        size_t rows;
        size_t cols;
        size_t size;

    public:
        Matrix();
        Matrix(const Matrix& rhs);
        Matrix(size_t _rows, size_t _cols);
        ~Matrix();

        double* operator[](size_t row);
        const double* operator[](size_t row) const;

        Matrix& operator=(const Matrix& rhs);
        
        double sum() const;
        double min() const;
        double max() const;

        Matrix& transpose();

        Matrix& add(const double scalar);
        Matrix& sub(const double scalar);
        Matrix& add(const Matrix& rhs);
        Matrix& sub(const Matrix& rhs);

        Matrix& mult(const double scalar);
        Matrix& div(const double scalar);
        Matrix& mult(const Matrix& rhs);
        Matrix& div(const Matrix& rhs);

        Matrix dot(const Matrix& rhs);

        Matrix& zero();
        Matrix& rand(double min = -0.5, double max = 0.5);
        
        void printShape() const;
        void print() const;

        size_t getRows() const;
        size_t getCols() const;
        size_t getSize() const;
};

inline double* Matrix::operator[](size_t row) {
    return &data[row * cols];
}

inline const double* Matrix::operator[](size_t row) const {
    return &data[row * cols];
}

#endif