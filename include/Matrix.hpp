#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <bits/c++config.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <random>

#ifndef SAFETY_CHECKS
#define SAFETY_CHECKS 1
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
        
        double sum();
        Matrix& transpose();

        Matrix& add(const double scalar);
        Matrix& add(const Matrix& rhs);
        Matrix& sub(const Matrix& rhs);

        Matrix& mult(const double scalar);
        Matrix dot(const Matrix& rhs);

        Matrix& zero();
        Matrix& rand();
        
        void print();

        size_t getRows();
        size_t getCols();
        size_t getSize();
};

#endif