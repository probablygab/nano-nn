#include "Matrix.hpp"

Matrix::Matrix() {
    data = nullptr;
}

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

Matrix::Matrix(const Matrix& rhs) {
    // Copy attr
    rows = rhs.rows;
    cols = rhs.cols;
    size = rhs.size;
    
    // Copy data
    data = new double[rhs.size];

    memcpy(data, rhs.data, size * sizeof(double));
}

Matrix::~Matrix() {
    delete[] data;
}

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

double Matrix::sum() const {
    double sum = 0.0;
    
    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < size; i++)
        sum += data[i];

    return sum;
}

double Matrix::min() const {
    double min = data[0];

    #pragma omp simd reduction(min:min)
    for (size_t i = 1; i < size; i++)
        min = std::min(data[i], min);

    return min;
}

double Matrix::max() const {
    double max = data[0];

    #pragma omp simd reduction(max:max)
    for (size_t i = 1; i < size; i++)
        max = std::max(data[i], max);

    return max;
}

Matrix& Matrix::add(const double scalar) {
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; i++)
        data[i] += scalar;

    return *this;
}

Matrix& Matrix::sub(const double scalar) {
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; i++)
        data[i] -= scalar;

    return *this;
}

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

Matrix& Matrix::mult(const double scalar) {
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; i++)
        data[i] *= scalar;

    return *this;
}

Matrix& Matrix::div(const double scalar) {    
    #pragma omp parallel for simd   
    for (size_t i = 0; i < size; i++)
        data[i] /= scalar;

    return *this;
}

Matrix Matrix::dot(const Matrix& rhs) {
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

        // Copy rhs column to avoid jumps in memory (cache-friendly)
        for (size_t k = 0; k < rhs.rows; k++)
            rhsCol[k] = rhs[k][col];

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

Matrix& Matrix::zero() {
    for (size_t i = 0; i < size; i++)
        data[i] = 0.0;

    return *this;
}

Matrix& Matrix::rand(double min, double max) {
    std::random_device randomDevice;
    std::default_random_engine randomEngine(randomDevice());
    std::uniform_real_distribution<double> uniformDist(min, max);

    for (size_t i = 0; i < size; i++)
        data[i] = uniformDist(randomEngine);

    return *this;
}

void Matrix::printShape() const {
    printf("(%zu, %zu)\n", rows, cols);
}

void Matrix::print() const {
    printShape();

    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++)
            printf("%lf  ", (*this)[row][col]);

        printf("\n");
    }

    printf("\n");
}

size_t Matrix::getRows() const {
    return rows;
}

size_t Matrix::getCols() const {
    return cols;
}

size_t Matrix::getSize() const {
    return size;
}