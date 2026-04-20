#include "../Header/Matrix.h"


//Matrix
/*========================================================================================================================================
 *                                                       Basic function
  ========================================================================================================================================*/
Matrix::Matrix(size_t r, size_t c) : row(r), column(c) {
    matrix.resize(row, std::vector<double>(column, 0.));
}

size_t Matrix::getRows() const {
    return row;
}

size_t Matrix::getCols() const {
       return column;
}

void Matrix::addRow() {
    std::vector<double> newRow(column, 0); // Create and initialize a new row
    matrix.push_back(newRow); // Add a new row at the end of the matrix
    ++row; // Update the row count
}

void Matrix::addColumn() {
    for (size_t i = 0; i < row; ++i) {
        matrix[i].push_back(0);   // Create and initialize a new column
    }
    ++column; // Update the column count
}

void Matrix::removeRow(size_t rowIndex) {
    if (rowIndex >= row) {
        throw std::out_of_range("The row index is out of range");
    }
    matrix.erase(matrix.begin() + rowIndex); // Delete the specified row
    --row; // Update the row count
}

void Matrix::removeColumn(size_t colIndex) {
    if (colIndex >= column) {
        throw std::out_of_range("The column index is out of range");
    }
    for (size_t i = 0; i < row; ++i) {
        matrix[i].erase(matrix[i].begin() + colIndex); //  Delete the specified column
    }
    --column; // Update the column count
}

void Matrix::print() const {
    for (size_t i = 0; i < row; ++i) {
        for (size_t j = 0; j < column; ++j) {
            std::cout << matrix[i][j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

 /* ==========================================================================================================================================
  *                                                   Overload matrix operators
  * ==========================================================================================================================================    */

std::vector <double>& Matrix::operator[](size_t i) {
    return matrix[i];
}

const std::vector <double>& Matrix::operator[](size_t i) const {
    return matrix[i];
}

Matrix Matrix::operator+ (const Matrix& other) const {
    if (row != other.getRows() || column != other.getCols()) throw std::invalid_argument("error: The matrix sizes are inconsistent and cannot be added");

    Matrix result(row, column);

    for (size_t i = 0; i < row; ++i) {
        for (size_t j = 0; j < column; ++j) {
            result[i][j] = matrix[i][j] + other[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator- (const Matrix& other) const {
    if (row != other.getRows() || column != other.getCols()) throw std::invalid_argument("error: The matrix sizes are inconsistent and cannot be subtracted");

    Matrix result(row, column);

    for (size_t i = 0; i < row; ++i) {
        for (size_t j = 0; j < column; ++j) {
            result[i][j] = matrix[i][j] - other[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (column != other.row) throw std::invalid_argument("error: The number of columns in the left matrix does not match the number of rows in the right matrix, so they cannot be multiplied");

    Matrix result(row, other.column);

    for (size_t i = 0; i < row; ++i) {
        for (size_t j = 0; j < other.column; ++j) {
            result[i][j] = 0;
            for (size_t k = 0; k < column; ++k) {
                result[i][j] += matrix[i][k] * other[k][j];
            }
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(row, column);

    for (size_t i = 0; i < row; ++i) {
        for (size_t j = 0; j < column; ++j) {
            result[i][j] = matrix[i][j] * scalar;
        }
    }
    return result;
}

Matrix operator* (double scalar, const Matrix& mat) {
    Matrix result(mat.row, mat.column);

    for (size_t i = 0; i < mat.row; ++i) {
        for (size_t j = 0; j < mat.column; ++j) {
            result[i][j] = mat.matrix[i][j] * scalar;
        }
    }
    return result;
}

/* ==========================================================================================================================================
 *                                                 Matrix correlation function
 * ==========================================================================================================================================    */

Matrix Matrix::transpose() const {
    Matrix result(column, row);
    for (size_t i = 0; i < row; ++i) {
        for (size_t j = 0; j < column; ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

double Matrix::determinant() const {
    if (row != column) throw std::invalid_argument("error: The matrix is not a square matrix, the determinant cannot be calculated");

    if (row == 1) return matrix[0][0];

    if (row == 2) return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

    double det = 0;
    Matrix submatrix(row - 1, row - 1);
    for (size_t k = 0; k < row; ++k) {
        for (size_t i = 1; i < row; ++i) {
            size_t subcol = 0;
            for (size_t j = 0; j < row; ++j) {
                if (j == k) continue;
                submatrix[i - 1][subcol++] = matrix[i][j];
            }
        }
        det += (k % 2 == 0 ? 1 : -1) * matrix[0][k] * submatrix.determinant();
    }
    return det;
}

Matrix Matrix::adjugate() const {
    Matrix adj(row, column);

    for (size_t i = 0; i < row; ++i) {
        for (size_t j = 0; j < column; ++j) {
            Matrix submatrix(row - 1, column - 1);

            for (size_t x = 0; x < row; ++x) {
                if (x == i) continue;
                for (size_t y = 0; y < column; ++y) {
                    if (y == j) continue;
                    submatrix[x < i ? x : x - 1][y < j ? y : y - 1] = matrix[x][y];
                }
            }

            adj[j][i] = ((i + j) % 2 == 0 ? 1 : -1) * submatrix.determinant();
        }
    }
    return adj;
}

Matrix Matrix::inverse() const {
    if (row != column) throw std::invalid_argument("error: The matrix is not square, and the inverse matrix cannot be calculated");
    double tem_1 = 0, tem_2 = 0, tem_3 = 0;
    Matrix inv(row, row);
    Matrix transition(row, 2 * row);

    // Amplify the right half of the matrix
    for (size_t i = 0; i < row; ++i) {
        for (size_t j = 0; j < 2 * row; ++j) {
            if (j < row) {
                transition[i][j] = matrix[i][j];
            }
            else {
                if ((j - row) == i) transition[i][j] = 1;
            }
        }
    }
    for (size_t i = 0; i < row; ++i) {
        // Determine whether the elements at the diagonal of the matrix are 0. If they are 0, continue to determine the elements in the next row until they are no longer 0, and then add them to this row
        if ((transition[i][i]) == 0) {
            for (size_t j = i + 1; j < row; ++j) {
                if ((transition[j][i]) != 0) {
                    // Add this row to the row that is 0
                    for (size_t k = 0; k < 2 * row; ++k) {
                        transition[i][k] += transition[j][k];
                    }
                    break;
                }
                if (j == row) {
                    std::cout << "This matrix cannot be inverted" << std::endl;
                    break;
                }
            }
        }
        tem_1 = transition[i][i];   // Set the first element of the previous row to 1
        for (size_t j = i; j < 2 * row; ++j) {
            transition[i][j] = transition[i][j] / tem_1;
        }
        // Set the first element of all subsequent lines to 0
        for (size_t j = i + 1; j < row; ++j) {
            tem_2 = transition[j][i];
            for (size_t k = i; k < 2 * row; ++k) {
                transition[j][k] = transition[j][k] - tem_2 * transition[i][k];
            }
        }
    }
    // Standardize the first half of the matrix
    for (int i = row - 1; i >= 0; --i) {
        for (int j = i - 1; j >= 0; --j) {
            tem_3 = transition[j][i];
            for (size_t k = i; k < 2 * row; ++k) {
                transition[j][k] = transition[j][k] - tem_3 * transition[i][k];
            }
        }
    }
    // Obtain the inverse matrix
    for (size_t i = 0; i < row; ++i) {
        for (size_t j = row; j < 2 * row; ++j) {
            inv[i][j - row] = transition[i][j];
        }
    }
    return inv;
}

Matrix Matrix::cholesky() const {
    const double epsilon = 1e-8;
    if (row != column) throw std::invalid_argument("error: The matrix is not square and the cholesky decomposition cannot be calculated");
    for (size_t i = 0; i < row; ++i) {
        for (size_t j = 0; j < i + 1; ++j) {
            double error_matrix = std::abs((matrix[i][j] - matrix[j][i]) / matrix[i][j]);
            if (error_matrix >= epsilon) {
                    std::cout << "error: The matrix is asymmetric, located at " <<  i + 1 << "line" << "\t" << j + 1 << "column" << std::endl;
                    std::cout << matrix[i][j] << std::endl;     std::cout << matrix[j][i] << std::endl;
                    std::cout << std::abs(matrix[i][j] - matrix[j][i]) << std::endl;
                    throw std::invalid_argument("Matrix must be symmetric");
            }
        }
    }
    Matrix cholesky(row, column);
    for (size_t j = 0; j < row; ++j) {
        double sum = 0.;
        for (size_t k = 0; k < j; ++k) sum += cholesky[j][k] * cholesky[j][k];
        double diag_val = matrix[j][j] - sum;
        if (diag_val <= -epsilon) throw std::runtime_error("Matrix is not positive definite");
        else if (diag_val < epsilon) diag_val = 0.0;  // Handle situations close to zero
        cholesky[j][j] = std::sqrt(matrix[j][j] - sum);
        for (size_t i = j + 1; i < column; ++i) {
            sum = 0;
            for (size_t k = 0; k < j; ++k) sum += cholesky[i][k] * cholesky[j][k];
            cholesky[i][j] = (matrix[i][j] - sum) / cholesky[j][j];
        }
    }
    return cholesky;
}

double Matrix::trace() const {
    if (row != column)
        throw std::invalid_argument("error:Trace requires square matrix");

    double sum = 0;
    for (size_t i = 0; i < row; ++i) {
        sum += matrix[i][i];
    }
    return sum;
}

Matrix Matrix::sqrtm() const {
    // Pre-check
    if (row != column)
        throw std::invalid_argument("sqrtm requires square matrix");
    if (determinant() < 0)
        throw std::invalid_argument("Matrix may have negative eigenvalues");

    const size_t n = row;
    const double epsilon = 1e-10;
    const size_t max_iter = 100;

    Matrix X = *this;  // Initial guess
    Matrix X_prev(n, n);

    // Å£¶Ùµü´ú·¨
    for (size_t iter = 0; iter < max_iter; ++iter) {
        Matrix inv_X(n, n);
        try {
            inv_X = X.inverse();
        }
        catch (...) {
            throw std::runtime_error("Matrix inversion latitudeled during sqrtm iteration");
        }

        X_prev = X;
        X = 0.5 * (X + (*this) * inv_X);

        // Check convergence
        double diff = 0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                diff += std::abs(X[i][j] - X_prev[i][j]);
            }
        }
        if (diff < epsilon) break;

        if (iter == max_iter - 1) {
            throw std::runtime_error("sqrtm did not converge");
        }
    }

    return X;
}

Matrix Matrix::symmetry_judgment() const {
    if (row != column) throw std::invalid_argument("error: The matrix is neither a square matrix nor a symmetric matrix.");
    Matrix judgment(row, column);
    for (size_t i = 0; i < row; ++i) {
        for (size_t j = 0; j < i + 1; ++j) {
            judgment[i][j] = matrix[i][j] - matrix[j][i];
        }
    }
    return judgment;
}

Matrix Matrix::symmetry_pruning() const {
    const double epsilon = 1e-8;
    if (row != column) throw std::invalid_argument("error: The matrix is neither a square matrix nor a symmetric matrix and cannot be pruned");
    Matrix pruning(row, column);
    for (size_t i = 0; i < row; ++i) {
        for (size_t j = i; j < column; ++j) {
            if(matrix[i][j] != 0){
                double error_matrix = std::abs((matrix[i][j] - matrix[j][i]) / matrix[i][j]);
                if (error_matrix < epsilon) {
                    pruning[i][j] = (matrix[i][j] + matrix[j][i]) / 2.;
                    if (i != j) pruning[j][i] = (matrix[i][j] + matrix[j][i]) / 2.;
                }
                else {
                    std::cout << "error: The matrix is asymmetric, located at " << i + 1 << "line" << "\t" << j + 1 << "column" << std::endl;
                    std::cout << matrix[i][j] << std::endl;     std::cout << matrix[j][i] << std::endl;
                    std::cout << std::abs(matrix[i][j] - matrix[j][i]) << std::endl;
                    throw std::invalid_argument("error: The error is too large to prune");
                }
            }
            
        }
    }
    return pruning;
}


// Define vector classes
/*========================================================================================================================================
                                                     Basic function              
  ========================================================================================================================================*/

Vector::Vector(size_t n) : dimension(n) {
    vec.resize(dimension, 0.);
}

size_t Vector::getDimension() const {
    return dimension;
}

void Vector::print() const {
    std::cout << "(";
    for (size_t i = 0; i < dimension - 1; ++i) std::cout << std::fixed << std::setw(12) << std::setprecision(8) << vec[i] << "," << "\t";
    std::cout << std::fixed << std::setw(12) << std::setprecision(8) << vec[dimension - 1];
    std::cout << ")T" << std::endl;
    std::cout << std::endl;
}

void Vector::print(size_t precision) const {
    std::cout << "(";
    for (size_t i = 0; i < dimension - 1; ++i) std::cout << std::fixed << std::setw(precision + 3) << std::setprecision(precision) << vec[i] << "," << "\t";
    std::cout << std::fixed << std::setw(precision + 3) << std::setprecision(precision) << vec[dimension - 1];
    std::cout << ")T" << std::endl;
    std::cout << std::endl;
}
/*========================================================================================================================================
 *                                                 Overload vector operators
  ========================================================================================================================================*/

double& Vector::operator[](size_t n) {
    if (n >= dimension) throw std::out_of_range("The index exceeds the vector size");
    return vec[n];
}

const double& Vector::operator[](size_t n) const {
    if (n >= dimension) throw std::out_of_range("The index exceeds the vector size");
    return vec[n];
}

Vector Vector::operator+ (const Vector& other) const {
    if (dimension != other.getDimension()) throw std::invalid_argument("error: The dimensions of the vectors are inconsistent and cannot be added");

    Vector result(dimension);

    for (size_t i = 0; i < dimension; ++i) {
        result[i] = vec[i] + other[i];
    }
    return result;
}

Vector Vector::operator- (const Vector& other) const {
    if (dimension != other.getDimension()) throw std::invalid_argument("error: The dimensions of the vectors are inconsistent and cannot be subtracted");

    Vector result(dimension);

    for (size_t i = 0; i < dimension; ++i) {
        result[i] = vec[i] - other[i];
    }
    return result;
}

double Vector::operator* (const Vector& other) const {
    if (dimension != other.getDimension()) throw std::invalid_argument("error: The vector dimensions are inconsistent and cannot be dot-multiplied");

    double result = 0.;

    for (size_t i = 0; i < dimension; ++i) {
        result += vec[i] * other[i];
    }
    return result;
}

Vector Vector::operator*(double scalar) const {
    Vector result(dimension);

    for (size_t i = 0; i < dimension; ++i) {
        result[i] = vec[i] * scalar;
    }
    return result;
}

Vector operator*(double scalar, const Vector& vec) {
    Vector result(vec.dimension);

    for (size_t i = 0; i < vec.dimension; ++i) {
        result[i] = vec.vec[i] * scalar;
    }
    return result;
}

Vector operator* (const Matrix& mat, const Vector& vec) {
    if (mat.getCols() != vec.getDimension()) throw std::invalid_argument("Size mismatch");
    Vector result(mat.getRows());
    for (size_t i = 0; i < mat.getRows(); ++i) {
        for (size_t j = 0; j < mat.getCols(); ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }
    return result;
}

Matrix Vector::transpose() const{
    Matrix result(1, dimension);
    for (size_t i = 0; i < dimension; ++i) result[0][i] = vec[i];
    return result;
}
    
/* ==========================================================================================================================================
 *                                            Vector correlation function      
 * ==========================================================================================================================================    */

double Vector::norm() const {
    double norm = 0.;
    for (size_t i = 0; i < dimension; ++i) {
        norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    return norm;
}


Vector Vector::cross(const Vector& other) {
    size_t n = dimension;
    Vector result(n);
    if (n != other.getDimension()) throw std::invalid_argument("vectors must have the same dimension.");
    if (n == 2) {
        Vector result1(n - 1);
        result1 = vec[0] * other[1] - vec[1] * other[0];
        return result1;
    }
    else if (n == 3) {
        result[0] = vec[1] * other[2] - vec[2] * other[1];
        result[1] = vec[2] * other[0] - vec[0] * other[2];
        result[2] = vec[0] * other[1] - vec[1] * other[0];
        return result;
    }
    // For cases with dimensions higher than 3, use an auxiliary function to calculate the determinant
    else {
        Matrix mat(n, n);
        for (size_t i = 0; i < n; ++i) {
            mat[0][i] = vec[i];
            mat[1][i] = other[i];
        }
        for (size_t i = 0; i < n; ++i) {
            Matrix submatrix(n - 1, n - 1);
            for (size_t r = 1; r < n; ++r) {
                size_t col2 = 0;
                for (size_t c = 0; c < n; ++c) {
                    if (c == i) continue;
                    submatrix[r - 1][col2++] = mat[r][c];
                }
            }
            result[i] = ((i % 2 == 0) ? 1 : -1) * submatrix.determinant();
        }
        return result;
    }
}





/*
------------------------------------------------------------------------------------------------------------------------------------------
*/

Matrix vector_mulTranspose(const Vector& vec) {
    Matrix result(vec.getDimension(), vec.getDimension());
    for (size_t i = 0; i < vec.getDimension(); ++i) {
        for (size_t j = 0; j < vec.getDimension(); ++j) {
            result[i][j] = vec[i] * vec[j];
        }
    }
    return result;
}

Matrix vector_mulTranspose1(const Vector& vecA, const Vector& vecB) {
    size_t rows = vecA.getDimension();
    size_t cols = vecB.getDimension();
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = vecA[i] * vecB[j];
        }
    }
    return result;
}

void insert(Matrix& matrix, const Vector& vector, size_t rowIndex) {
    if (vector.getDimension() != matrix.getRows()) throw std::invalid_argument("error: The dimension of a vector is not equal to the number of rows of the assigned matrix");
    if (rowIndex >= matrix.getCols()) throw std::out_of_range("error: The row index is beyond the range of the matrix");

    for (size_t col = 0; col < matrix.getRows(); ++col) {
        matrix[col][rowIndex] = vector[col];
    }
}

Vector Matrix_read(Matrix matrix, size_t Index, size_t model){
    size_t line_dim = (model == 0) ? matrix.getCols() :
        (model == 1) ? matrix.getRows() : throw std::invalid_argument("error:model");

    Vector index_vec(line_dim);
    if (model == 0) 
        for (size_t row = 0; row < matrix.getRows(); ++row) 
            index_vec[row] = matrix[Index][row];
    if (model == 1)
        for (size_t col = 0; col < matrix.getRows(); ++col)
            index_vec[col] = matrix[col][Index];
    
    return index_vec;
}

Matrix diag_creat(const Vector& v) {
    size_t n = v.getDimension();
    Matrix result(n, n);
    for (size_t i = 0; i < n; ++i) {
        result[i][i] = v[i];
    }
    return result;
}

static Vector diag(const Matrix& A) {
    size_t len = std::min(A.getRows(), A.getCols());
    Vector result(len);
    for (size_t i = 0; i < len; ++i) {
        result[i] = A[i][i];
    }
    return result;
}

double maxDiagRatio(const Matrix& A, const Matrix& B) {
    Vector diagA = diag(A);
    Vector diagB = diag(B);

    if (diagA.getDimension() != diagB.getDimension()) throw std::invalid_argument("maxDiagRatio -> error: The dimensions of the two matrices are different");

    double max_val = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < diagA.getDimension(); ++i) {
        if (diagB[i] == 0) throw std::runtime_error("maxDiagRatio -> error: The diagonal elements of matrix B contain 0");
        double ratio = diagA[i] / diagB[i];
        if (ratio > max_val) {
            max_val = ratio;
        }
    }
    return max_val;
}

std::istream& operator>>(std::istream& in, Matrix& mat) {
    for (size_t i = 0; i < mat.getRows(); ++i) {
        for (size_t j = 0; j < mat.getCols(); ++j) {
            in >> mat[i][j];
        }
    }
    return in;
}

std::istream& operator>>(std::istream& in, Vector& vec) {
    for (size_t i = 0; i < vec.getDimension(); ++i) {
        in >> vec[i];
    }
    return in;
}
