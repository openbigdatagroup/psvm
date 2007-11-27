/*
Copyright 2007 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef MATRIX_H__
#define MATRIX_H__

#include "io.h"

// ParallelMatrix: Matrix data structure for parallel version
// LLMatrix: Matrix data structure for single processor version
// MatrixManipulation: static data structure for operations on matrix
//   including ICF(Incomplete cholesky factorization), CF(Cholesky
//   Factorization), ProductMM(Compute HDH'), CholBackwardSub(resolve A'x=b)
//   CholForwardSub(resolve Ax=b)

namespace psvm {
struct PrimalDualIPMParameter;
class Sample;
class Document;
class Kernel;

// ParallelMatrix: n*p parallel matrix.
// Storage: each computer stores some rows of the global matrix.
//   So data on local is in fact a matrix which is nlocal*p where
//   nlocal = n / N(machine number). Data on local computer is column
//   based.
// Example: There are two processors and global matrix (4*3) is
//   1 2 3
//   4 5 6
//   7 8 9
//   1 5 9
// First computer elements_  (2*3) : 1 7 / 2 8 / 3 9
// Second computer elements_ (2*3) : 4 1 / 5 5 / 6 9
class ParallelMatrix {
 public:
  ParallelMatrix();

  // A wrap for Init(num_rows, num_cols).
  // Look for Init(int, int) for details.
  ParallelMatrix(int num_rows, int num_cols);

  ~ParallelMatrix();

  // Initilizes the matrix: original matrix is num_rows_*num_cols_
  // But the data stored locally is in fact num_local_rows_*num_cols_
  // So element_ is two-dimension array of num_local_rows_*num_cols_
  // where num_local_rows_ = num_rows / N;
  void Init(int num_rows, int num_cols);

  // Destroys the matrix by freeing space of element_ if space has
  // been allocated.
  void Destroy();

  // Gets the number of rows of the global matrix.
  inline int GetNumRows() const { return num_rows_; }

  // Gets the number of columns of the global/local matrix.
  inline int GetNumCols() const { return num_cols_; }

  // Gets the number of rows which are stored on local.
  inline int GetNumLocalRows() const { return num_local_rows_; }

  // Resets the number of columns to a smallerer value. Sometimes we
  // would like to decrease the number of columns (such as in
  // MatrixManipulation::ICF), so we have to free unused columns
  // and update the value of num_cols_.
  void SetNumCols(int num_cols);

  // Get the matrix element value on row:x, column:y from local machine.
  // Because element_ is column based stored, it is the value of
  // element_[y][x].
  // Row index x is the row index of local matrix, so if you want to
  // get row:x column:y of the original matrix, you must check whether
  // row:x is stored on this machine, only if it is true, then you must
  // convert global row index to local row index by calling
  //   ParallelInterface::ComputeGlobaToLocal()
  // and use the return value as the function parameter x.
  inline double Get(int x, int y) const { return element_[y][x]; }

  // Sets a matrix element value on row:x, column:y on local machine.
  inline void Set(int x, int y, double value) { element_[y][x] = value; }

  // Saves local matrix chunk to file. The whole matrix is distributed by
  // rows, Processor # will write its chunk into file "path/file_name.#".
  void Save(const char* path, const char* file_name);

  // Load matrix chunks from "path" directory. If successfully loaded, return
  // true, otherwise return false. If current number of processors is different
  // from the number of processors when saving the matrix chunks, matrix chunks
  // are merged together, and splitted into proper number of chunks, at last
  // these chunks are loaded by function ReadChunks().
  bool Load(const char* path, const char* file_name);

  // Loads matrix chunks files from directory "path", processor # will load
  // "path/file_name.#"  This function assumes that the number of processors is
  // the same with the number of chunks. If successfully read, return true;
  // otherwise return false. This function allocate memory using Init().
  bool ReadChunks(const char *path, const char* file_name);

 private:
  // Returns true if exactly size bytes are read
  bool Read(File* inbuf, void* buf, size_t size);

  // Returns true if exactly size bytes are written
  bool Write(File* obuf, void* buf, size_t size);

  // The number of rows of the global matrix.
  int num_rows_;

  // The number of columns of the global matrix.
  int num_cols_;

  // The number of rows stored on local machine.
  int num_local_rows_;

  // Two-dimension array for matrix elements.
  double** element_;
};

// LLMatrix: n*n lower triangular/symmetrical matrix.
// Storage: This is a single processor matrix and is not distributed among
//          machines. Because the matrix is either lower triangular or
//          symmentric, we only store lower half part of the matrix and the
//          storage is column based.
// Example: symmetric matrix (3*3) is
//            1 2 3
//            2 5 6
//            3 6 9
//            elements_: 1 2 3 / 5 6 / 9
//          lower triangular (3*3) is
//            1 0 0
//            2 5 0
//            3 6 9
//            elements_: 1 2 3 / 5 6 / 9
class LLMatrix {
 public:
  LLMatrix();

  // Initialize the matrix by calling Init(int)
  explicit LLMatrix(int dim);
  ~LLMatrix();

  // Destroy the matrix by freeing allocated space
  void Destroy();

  // Initialize and allocate space for the matrix. The space allocated
  // is in fact dim+(dim-1)+(dim-2)+....+2+1 because we only stored lower
  // half part of it. element_ is column based, so element_ is a pointer
  // array of size dim and element_[i] is an array of size n-i+1.
  void Init(int dim);

  // Return the dimension of the symmetric/lower triangular matrix
  inline int GetDim() const { return dim_; }

  // Return the row:x  column:y element of the matrix, because we only
  // stored lower half part, x must be no greater than y. The real storage
  // location for the element is in fact element_[y][x-y] as the storage
  // format introduced above.
  inline double Get(int x, int y) const { return element_[y][x-y]; }

  // Set a matrix element on row:x, column:y.
  inline void Set(int x, int y, double value) { element_[y][x-y] = value; }
 private:
  // The dimension of the matrix
  int dim_;

  // Two dimension array for storing matrix elements.
  double** element_;
};

struct Pivot {
  double pivot_value;
  int pivot_index;
};
}

#endif
