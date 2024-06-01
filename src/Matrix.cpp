#include "Matrix.hpp"

/**
 * @brief Construct an empty Matrix. 
 * You need to assign an existing Matrix to this one when doing this.
 * 
 */
Matrix::Matrix() {
    data = nullptr;
}


/**
 * @brief Destroy the Matrix object. 
 * Deallocates the data array.
 * 
 */
Matrix::~Matrix() {
    delete[] data;
}


/**
 * @brief Copy constructor. 
 * This can be used to create a copy when modifying the original Matrix is not desired.
 * 
 * @param rhs Matrix to copy.
 */
Matrix::Matrix(const Matrix& rhs) {
    // Copy attr
    rows = rhs.rows;
    cols = rhs.cols;
    size = rhs.size;
    
    // Copy data
    data = new double[rhs.size];

    memcpy(data, rhs.data, size * sizeof(double));
}


/**
 * @brief Construct a Matrix with the given dimensions.
 * Does not initialize the Matrix with any values. Use `.zero()` or `.rand()` or do it manually.
 * 
 * @param _rows Number of rows.
 * @param _cols Number of columns.
 */
Matrix::Matrix(size_t _rows, size_t _cols) {
    if (SAFETY_CHECKS)
        if (_rows == 0 || _cols == 0) {
            fprintf(stderr, "ERROR: Matrix size cannot be zero (%zu, %zu)\n", _rows, _cols);
            exit(1);
        }

    rows = _rows;
    cols = _cols;
    size = _rows * _cols;
    data = new double[size];
}


/**
 * @brief Assign a Matrix to this one.
 * This will copy the data from rhs to this Matrix, erasing all previous data.
 * 
 * @param rhs Matrix to assign.
 * @return Reference to this Matrix.
 */
Matrix& Matrix::operator=(const Matrix& rhs) {
    if (this != &rhs) {
        // Deallocate existing matrix
        if (data != nullptr)
            delete[] data;

        // Copy attr
        rows = rhs.rows;
        cols = rhs.cols;
        size = rhs.size;
        
        // Copy data
        data = new double[rhs.size];

        memcpy(data, rhs.data, size * sizeof(double));
    }

    return *this;
}


/**
 * @brief Sum all elements in the Matrix.
 * 
 * @return sum of all elements.
 */
double Matrix::sum() const {
    double sum = 0.0;
    
    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < size; i++)
        sum += data[i];

    return sum;
}


/**
 * @brief Get the minimum value in the Matrix.
 * 
 * @return minimum value.
 */
double Matrix::min() const {
    double min = data[0];

    #pragma omp simd reduction(min:min)
    for (size_t i = 1; i < size; i++)
        min = std::min(data[i], min);

    return min;
}


/**
 * @brief Get the maximum value in the Matrix.
 * 
 * @return maximum value.
 */
double Matrix::max() const {
    double max = data[0];

    #pragma omp simd reduction(max:max)
    for (size_t i = 1; i < size; i++)
        max = std::max(data[i], max);

    return max;
}


/**
 * @brief Add a scalar to all elements in the Matrix.
 * 
 * @param scalar Scalar to add.
 * @return Reference to this Matrix.
 */
Matrix& Matrix::add(const double scalar) {
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; i++)
        data[i] += scalar;

    return *this;
}


/**
 * @brief Subtract a scalar from all elements in the Matrix.
 * 
 * @param scalar Scalar to subtract.
 * @return Reference to this Matrix.
 */
Matrix& Matrix::sub(const double scalar) {
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; i++)
        data[i] -= scalar;

    return *this;
}


/**
 * @brief Multiply all elements in the Matrix by a scalar.
 * 
 * @param scalar Scalar to multiply.
 * @return Reference to this Matrix.
 */
Matrix& Matrix::mult(const double scalar) {
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; i++)
        data[i] *= scalar;

    return *this;
}


/**
 * @brief Divide all elements in the Matrix by a scalar.
 * 
 * @param scalar Scalar to divide.
 * @return Reference to this Matrix.
 */
Matrix& Matrix::div(const double scalar) {    
    #pragma omp parallel for simd   
    for (size_t i = 0; i < size; i++)
        data[i] /= scalar;

    return *this;
}


/**
 * @brief Add another Matrix to this one.
 * Elements are added element-wise.
 * 
 * @param rhs Matrix to add.
 * @return Reference to this Matrix.
 */
Matrix& Matrix::add(const Matrix& rhs) {
    if (SAFETY_CHECKS)
        if (rows != rhs.rows || cols != rhs.cols) {
            fprintf(stderr, "ERROR: Cannot add matrices of different dimensions: (%zu, %zu) vs. (%zu, %zu)\n",
            rows, cols, rhs.rows, rhs.cols);
            exit(1);
        }

    #pragma omp parallel for simd
    for (size_t i = 0; i < size; i++)
        data[i] += rhs.data[i];

    return *this;
}


/**
 * @brief Subtract another Matrix from this one.
 * Elements are subtracted element-wise.
 * 
 * @param rhs Matrix to subtract.
 * @return Reference to this Matrix.
 */
Matrix& Matrix::sub(const Matrix& rhs) {
    if (SAFETY_CHECKS)
        if (rows != rhs.rows || cols != rhs.cols) {
            fprintf(stderr, "ERROR: Cannot sub matrices of different dimensions: (%zu, %zu) vs. (%zu, %zu)\n",
            rows, cols, rhs.rows, rhs.cols);
            exit(1);
        }

    #pragma omp parallel for simd
    for (size_t i = 0; i < size; i++)
        data[i] -= rhs.data[i];

    return *this;
}


/**
 * @brief Multiply this Matrix by another one.
 * Elements are multiplied element-wise.
 * 
 * @param rhs Matrix to multiply.
 * @return Reference to this Matrix.
 */
Matrix& Matrix::mult(const Matrix& rhs) {
    if (SAFETY_CHECKS)
        if (rows != rhs.rows || cols != rhs.cols) {
            fprintf(stderr, "ERROR: Cannot multiply matrices of different dimensions: (%zu, %zu) vs. (%zu, %zu)\n",
            rows, cols, rhs.rows, rhs.cols);
            exit(1);
        }
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; i++)
        data[i] *= rhs.data[i];

    return *this;
}


/**
 * @brief Divide this Matrix by another one.
 * Elements are divided element-wise.
 * 
 * @param rhs Matrix to divide.
 * @return Reference to this Matrix.
 */
Matrix& Matrix::div(const Matrix& rhs) {
    if (SAFETY_CHECKS)
        if (rows != rhs.rows || cols != rhs.cols) {
            fprintf(stderr, "ERROR: Cannot divide matrices of different dimensions: (%zu, %zu) vs. (%zu, %zu)\n",
            rows, cols, rhs.rows, rhs.cols);
            exit(1);
        }

    #pragma omp parallel for simd
    for (size_t i = 0; i < size; i++)
        data[i] /= rhs.data[i];

    return *this;
}


/**
 * @brief Perform a dot product with another Matrix. 
 * Does not modify this Matrix or rhs.
 * 
 * @param rhs Matrix to dot product with.
 * @return A new Matrix resulting from the dot product.
 */
Matrix Matrix::dot(const Matrix& rhs) const {
    if (SAFETY_CHECKS)
        if (cols != rhs.rows) {
            fprintf(stderr, "ERROR: Cannot dot product matrices of different dimensions: (%zu, %zu *) vs. (%zu *, %zu)\n",
            rows, cols, rhs.rows, rhs.cols);
            exit(1);
        }

    Matrix res = Matrix(rows, rhs.cols);

    #pragma omp parallel for
    for (size_t col = 0; col < rhs.cols; col++) {
        double* rhsCol = new double[rhs.rows];

        // Copy rhs column to avoid jumps in memory (more cache-friendly)
        for (size_t k = 0; k < rhs.rows; k++)
            rhsCol[k] = rhs[k][col];

        // Multiply rows of this matrix with the column of rhs
        for (size_t row = 0; row < rows; row++) {
            double sum = 0.0;

            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < cols; k++)
                sum += (*this)[row][k] * rhsCol[k];

            res[row][col] = sum;
        }

        delete[] rhsCol;
    }
  
    return res;
}


/**
 * @brief Transpose the Matrix. This will modify the Matrix in-place.
 * @attention
 * To transpose a Matrix without modifying it, use the copy constructor as follows:
 * `Matrix At = Matrix(A).transpose()`
 * 
 * @return Reference to this Matrix.
 */
Matrix& Matrix::transpose() {
    // 1D-like matrix
    if (rows == 1 || cols == 1) {
        std::swap<size_t>(rows, cols);

        return *this;
    }

    // Square matrix
    if (rows == cols) {
        for (size_t row = 1; row < rows; row++) {
            for (size_t col = 0; col < row; col++)
                std::swap<double>((*this)[row][col], (*this)[col][row]);
        }

        return *this;
    }

    // Non-square matrix
    double* dataT = new double[size];

    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++)
            dataT[col * rows + row] = data[row * cols + col];
    }

    // Swap data and dimensions
    delete[] data;
    data = dataT;

    std::swap<size_t>(rows, cols);    

    return *this;
}


/**
 * @brief Fill the Matrix with zeros.
 * 
 * @return Reference to this Matrix.
 */
Matrix& Matrix::zero() {
    for (size_t i = 0; i < size; i++)
        data[i] = 0.0;

    return *this;
}


/**
 * @brief Fill the Matrix with random values.
 * 
 * @param min Minimum value.
 * @param max Maximum value.
 * @return Reference to this Matrix.
 */
Matrix& Matrix::rand(double min, double max) {
    std::random_device randomDevice;
    std::default_random_engine randomEngine(randomDevice());
    std::uniform_real_distribution<double> uniformDist(min, max);

    for (size_t i = 0; i < size; i++)
        data[i] = uniformDist(randomEngine);

    return *this;
}


/**
 * @brief Print the shape of the Matrix.
 * 
 */
void Matrix::printShape() const {
    printf("(%zu, %zu)\n", rows, cols);
}


/**
 * @brief Print the shape and Matrix values.
 * 
 */
void Matrix::print() const {
    printShape();

    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++)
            printf("%lf  ", (*this)[row][col]);

        printf("\n");
    }

    printf("\n");
}


/**
 * @brief Get the number of rows in the Matrix.
 * 
 * @return Number of rows.
 */
size_t Matrix::getRows() const {
    return rows;
}


/**
 * @brief Get the number of columns in the Matrix.
 * 
 * @return Number of columns.
 */
size_t Matrix::getCols() const {
    return cols;
}


/**
 * @brief Get the total number of elements in the Matrix (rows * cols).
 * 
 * @return Total number of elements.
 */
size_t Matrix::getSize() const {
    return size;
}