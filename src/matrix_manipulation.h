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

#ifndef MATRIX_MANIPULATION_H__
#define MATRIX_MANIPULATION_H__

#include <string>

using namespace std;

namespace psvm {
struct PrimalDualIPMParameter;
class Sample;
class Document;
class Kernel;
class LLMatrix;
class ParallelMatrix;

// Provides static methods for manipulating matrix operations, including
// ICF (Incomplete Cholesky Factorization), CF (Cholesky Factorization),
// ProductMM (computes HDH'), CholBackwardSub (solves A'x=b) and
// CholForwardSub (solves Ax=b).
class MatrixManipulation {
 public:
  // Performs Cholesky factorization on 'symmetric_matrix'.
  // The result is a lower triangular matrix 'result', which satisfies
  //           symetric_matrix = result * result'
  //
  // The memory block of 'result' will be allocated inside the function.
  // It is the caller's responsibility to free the memory.
  //
  // If CF is successfully done, return true; otherwise, return false.
  static bool CF(const LLMatrix &symmetric_matrix, LLMatrix *result);

  // Performs incomplete Cholesky factorization incrementally on the kernel
  // matrix, and the resulting matrix is stored distributedly.
  //
  // ICF performs a incremental version of incomplete cholesky factorization.
  //   Firstly, original ICF matrix is loaded from directory
  // 'parameter.pc_path'. If no such matrix avaiable, we regard a 0*0 matrix
  // is loaded.
  //   After the original matrix is loaded (assume a n-by-p matrix), an
  // incremental incomplete cholesky factorization is performed, which expands
  // the original n-by-p matrix to a (n+n_plus)-by-(p+p_plus) matrix. n_plus
  // is the number of added samples provided by 'doc', p_plus is the number of
  // columns need to be attached to the original matrix. If n and p are zeros,
  // an ordinary incomplete cholesky factorization is performed.
  //
  // Kernel matrix is not actually formed and stored in memory. When a
  // matrix element is required, it is calculated based on corresponding
  // samples (provided by 'doc', or broadcasted from other machines),
  // kernel parameters ('kernel').
  //
  // The resulting ICF matrix is stored in 'icf', which is distributed by rows.
  // The memory blocks of 'icf' is allocated in this function, it's the caller's
  // obligation to free these memory blocks, which can be done in the
  // destructor of class ParallelMatrix.
  static void ICF(const Document &doc, const Kernel &kernel,
                  int rows, int columns, double threshold,
                  ParallelMatrix *icf);

  // Computes l = h'*d*h where d is a diagonal matrix. h is parallel matrix and
  // rows are distributed among machines, so reductions between computers are
  // necessary. The result is a symmetric matrix stored on machine ID:0 and
  // other computers do not have this result. l's space will be allocated
  // inside the function.
  static void ProductMM(const ParallelMatrix &icf, const double *diagonal,
                        LLMatrix *product);

  // Resolves a'x = b where a is lower triangular matrix, x and b are vectors.
  // It is easy to be solved by backward substitutions.
  static void CholBackwardSub(const LLMatrix &a, const double *b, double *x);

  // Resolves ax = b where a is lower triangular matrix, x and b are vectors.
  // It is easy to be solved by forward substitutions.
  static void CholForwardSub(const LLMatrix &a, const double *b, double *x);

 private:
  // Broadcasts the pivot sample from local machine to other machines.
  // This function is used in ICF.
  static void BroadcastPivotSample(const Document &doc,
                                   int pivotSampleGlobalID,
                                   Sample **pivotSample);

  // Frees pivot sample memory space (pointed by pPivotSample).
  // The pivot index is also provided to verify whether the current processor
  // needs to free the pivot sample. This function is used in ICF.
  static void FreePivotSample(Sample *pPivotSample, int pivotIndex);

};
}  // namespace psvm

#endif
