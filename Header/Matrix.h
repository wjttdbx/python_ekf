#ifndef _MATRIX_H_	
#define _MATRIX_H_

#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>

//Define the matrix class
class Matrix {
private:
    friend class Vector;
    std::vector <std::vector <double>> matrix;
    size_t row = 0;
    size_t column = 0;

public:
    Matrix() = default;
    /*========================================================================================================================================
    *                                                       Basic function
    ========================================================================================================================================*/
    //Construct and initialize the matrix
    Matrix(size_t r, size_t c);   

    // Obtain the rows of the matrix
    size_t getRows() const;

    // Obtain the columns of the matrix
    size_t getCols() const;

    // Add a new row
    void addRow();

    // Add a new columns
    void addColumn();

    // Delete the specified row
    void removeRow(size_t rowIndex);

    // Delete the specified columns
    void removeColumn(size_t colIndex);

    // Print matrix
    void print() const;

    /* ==========================================================================================================================================
    *                                                   Overload matrix operators
    * ==========================================================================================================================================    */

    // Overload the "[]" operator to enable element access using matrix objects(Modifiable)
    std::vector <double>& operator[](size_t i);

    // Overload the "[]" operator to enable element access using matrix objects(Read-only)
    const std::vector <double>& operator[](size_t i) const;

    // Overload the "+" operator for matrix addition
    Matrix operator+ (const Matrix& other) const;

    // Overload the "-" operator for matrix subtraction
    Matrix operator- (const Matrix& other) const;

    // Overload the "*" operator for matrix multiplication
    Matrix operator*(const Matrix& other) const;

    // Overloaded the "*" operator is used for multiplying constants and matrices
    Matrix operator*(double scalar) const;

    friend Matrix operator* (double scalar, const Matrix& mat);

    /* ==========================================================================================================================================
    *                                                  Matrix correlation function
    * ==========================================================================================================================================    */
    // Matrix transpose
    Matrix transpose() const;

    // Calculate the matrix determinant
    double determinant() const;

    // Calculate the adjoint matrix
    Matrix adjugate() const;

    // Calculate the inverse matrix(Gauss method)
    Matrix inverse() const;

    // Cholesky decomposition
    Matrix cholesky() const;

    // Calculate the matrix trace(the sum of the diagonal elements)
    double trace() const;

    // Calculate the square root of a matrix(similar to MATLAB's sqrtm())
    Matrix sqrtm() const;

    // Matrix symmetry judgment
    Matrix symmetry_judgment() const;

    // Matrix symmetry pruning
    Matrix symmetry_pruning() const;
};

// Define vector classes
class Vector {
    friend class Matrix;
private:
    std::vector <double> vec;
    size_t dimension = 0;   // The dimension of the vector

public:
    /*========================================================================================================================================
                                                             Basic function
   ========================================================================================================================================*/
   // Construct the vector and initialize it
    Vector() = default;
    Vector(size_t n);

    // Obtain the dimension of the vector
    size_t getDimension() const;

    // Print vector
    void print() const;
    // Print vector(specifies the number of decimal places)
    void print(size_t precision) const;

    /*========================================================================================================================================
   *                                                     Overload vector operators
   ========================================================================================================================================*/
   // Overload the "[]" operator to enable element access using vector objects (Modifiable)
    double& operator[](size_t n);

    // Overload the [] operator to enable element access using vector objects (Read-only)
    const double& operator[](size_t n) const;

    // Overloading the "+" operator for vector addition
    Vector operator+ (const Vector& other) const;

    // Overload the "-" operator for vector subtraction
    Vector operator- (const Vector& other) const;

    // Overloading the "*" operator for vector dot multiplication
    double operator* (const Vector& other) const;

    // Overloading the "*" operator is used for multiplying constants and vectors
    Vector operator*(double scalar) const;
    friend Vector operator*(double scalar, const Vector& vec);

    // Overloading the "*" operator is used for multiplying matrices and vectors
    friend Vector operator* (const Matrix& mat, const Vector& vec);
    /* ==========================================================================================================================================
   *                                                   Vector correlation function
   * ==========================================================================================================================================    */
   // The modulus of the vector
    double norm() const;

    // The function is used to calculate the cross product of two vectors
    Vector cross(const Vector& other);

    // Vector transposition
    Matrix transpose() const;
};

/* ==========================================================================================================================================
*                                                  向量相关函数
  *==========================================================================================================================================*/

// The vector A is multiplied by the transpose vector AT
Matrix vector_mulTranspose(const Vector& vec);

// The product of the transpose of vector A and vector B
Matrix vector_mulTranspose1(const Vector& vecA, const Vector& vecB);

// Insert the vector into the matrix
void insert(Matrix& matrix, const Vector& vector, size_t rowIndex);

//Output the rows/columns of the matrix as vectors
//(model:0 -> specified row;    1 -> Specified column)
Vector Matrix_read(Matrix matrix, size_t Index, size_t model);

// Generate the diagonal matrix (input vector)
Matrix diag_creat(const Vector& v);

// Extract the diagonal elements (input matrix)
Vector diag(const Matrix& A);

// Calculation max(diag(A)/diag(B)) 
double maxDiagRatio(const Matrix& A, const Matrix& B);


// Overload the input operator for the input matrix
std::istream& operator>>(std::istream& in, Matrix& mat);

// Overload the input operator for input vectors
std::istream& operator>>(std::istream& in, Vector& vec);




#endif