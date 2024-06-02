#ifndef MATRIX_H
#define MATRIX_H

#include <cstdio>
#include <cstring>
#include <random>


/** 
 * If SAFETY_CHECKS is defined, the program will check for errors in the Matrix operations and Neural Network operations.
 * If disabled, any misuse will result in undefined behavior and possibly a crash.
 */
#ifndef SAFETY_CHECKS
#define SAFETY_CHECKS true
#endif


/**
 * @brief Matrix class for Neural Network operations.
 * A few optimizations were made, but nothing major. This class is not faster than numpy or Eigen.
 * @attention
 * Most of the operations are implemented in-place (except for dot), 
 * so the return value is the same as the object calling the method.
 * @n
 * This allows for chained operations like:
 * `Matrix(3, 5).rand(1.0, 5.0).add(1.0).transpose().dot(Matrix(3, 3).rand()).div(3.0)`.
 * @n 
 * To perform operations without modifying the original matrix, you can use the copy constructor:
 * `Matrix At = Matrix(A).transpose()` is equivalent to `Matrix At = A.transpose()`, but the latter modifies A.
 * 
 */
class Matrix {
    private:
        double* data = nullptr;

        size_t rows;
        size_t cols;
        size_t size;

    public:
        Matrix();
        ~Matrix();
        Matrix(const Matrix& rhs);
        Matrix(size_t _rows, size_t _cols);

        Matrix& operator=(const Matrix& rhs);

        double* operator[](size_t row);
        const double* operator[](size_t row) const;

        double sum() const;
        double min() const;
        double max() const;

        Matrix& add(const double scalar);
        Matrix& sub(const double scalar);
        Matrix& mult(const double scalar);
        Matrix& div(const double scalar);

        Matrix& add(const Matrix& rhs);
        Matrix& sub(const Matrix& rhs);
        Matrix& mult(const Matrix& rhs);
        Matrix& div(const Matrix& rhs);

        Matrix dot(const Matrix& rhs) const;
        Matrix dotTransposeRight(const Matrix& rhs) const;
        Matrix& transpose();

        Matrix& zero();
        Matrix& rand(double min = -0.5, double max = 0.5);
        
        void printShape() const;
        void print() const;

        size_t getRows() const;
        size_t getCols() const;
        size_t getSize() const;
};

/**
 * @brief Access a row of the matrix for READ and WRITE.
 * This returns a pointer to the row, so you can index the row to access the columns as follows:
 * `A[0][0] = 1.0;`
 * 
 * @param row Row to access.
 * @return double* Pointer to the row.
 */
inline double* Matrix::operator[](size_t row) {
    return &data[row * cols];
}

/**
 * @brief Access a row of the matrix for READ ONLY.
 * This returns a pointer to the row, so you can index the row to access the columns as follows:
 * `double value = A[0][0];`
 * 
 * @param row Row to access.
 * @return double* Pointer to the row.
 */
inline const double* Matrix::operator[](size_t row) const {
    return &data[row * cols];
}

#endif