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

#include <float.h>
#include <cmath>

#include "matrix_manipulation.h"

#include "io.h"
#include "util.h"

#include "matrix.h"
#include "parallel_interface.h"
#include "document.h"
#include "kernel.h"
#include "pd_ipm_parm.h"

namespace psvm {

// Cholesky factorization: factorize matrix A into LL^T.
// "original" represents A, low_triangular represents L.
// The memory block of L is allocated in this function, which will be freed
// by calling Destory() (which is done in the destructor of LLMatrix).
bool MatrixManipulation::CF(const LLMatrix &original,
                            LLMatrix *low_triangular) {
  CHECK(low_triangular);
  int dim = original.GetDim();
  low_triangular->Init(dim);
  for (int i = 0; i < dim; ++i) {
    for (int j = i; j < dim; ++j) {
      double sum = original.Get(j, i);
      for (int k = i-1; k >= 0; --k) {
        sum -= low_triangular->Get(i, k) * low_triangular->Get(j, k);
      }
      if (i == j) {
        if (sum <= 0) {  // sum should be larger than 0
          cerr << "Only symmetric positive definite matrix can perform"
                       " Cholesky factorization.";
          return false;
        }
        low_triangular->Set(i, i, sqrt(sum));
      } else {
        low_triangular->Set(j, i, sum/low_triangular->Get(i, i));
      }
    }
  }
  return true;
}

// For the processor storing the pivot sample in local machine (here, 'doc'),
// *pivot_sample will be assigned to be the pointer to the pivot sample.
// For other processors, pivot sample data will be packed and broadcasted from
// the processor owning it, and a Sample '**pivot_sample' will be allocated to
// store pivot sample data.
//
// 'pivot_global_index' is used to verify whether the pivot sample is stored
// in current processor (in 'doc').
void MatrixManipulation::BroadcastPivotSample(const Document &doc,
                                              int pivot_global_index,
                                              Sample **pivot_sample) {
  CHECK(pivot_sample);
  ParallelInterface *mpi = ParallelInterface::GetParallelInterface();
  int parallel_id = mpi->GetProcId();
  int pnum = mpi->GetNumProcs();
  int pivot_local_index = mpi->ComputeGlobalToLocal(pivot_global_index,
                                                           parallel_id);
  int buff_size = 0;
  char *buffer = NULL;
  if (pivot_local_index != -1) {
    // Pack the pivot sample data
    *pivot_sample = const_cast<Sample*>(doc.GetLocalSample(pivot_local_index));
    buff_size = Document::PackSample(buffer, **pivot_sample);

    // Broadcast package size
    mpi->Bcast(&buff_size, 1, MPI_INT, parallel_id, MPI_COMM_WORLD);

    // Broadcast package
    mpi->Bcast(buffer, buff_size, MPI_BYTE, parallel_id, MPI_COMM_WORLD);
  } else {
    int root = pivot_global_index % pnum;
    // Broadcast package size
    mpi->Bcast(&buff_size, 1, MPI_INT, root, MPI_COMM_WORLD);

    // Prepare a suitable memory block to receive package.
    buffer = new char[buff_size];

    // Broadcast package
    mpi->Bcast(buffer, buff_size, MPI_BYTE, root, MPI_COMM_WORLD);

    Document::UnpackSample(*pivot_sample, buffer);
  }
  delete [] buffer;
}

void MatrixManipulation::FreePivotSample(Sample *pivot_sample,
                                         int pivot_index) {
  ParallelInterface *mpi = ParallelInterface::GetParallelInterface();
  int parallel_id = mpi->GetProcId();
  if (mpi->ComputeGlobalToLocal(pivot_index, parallel_id) == -1) {
    delete pivot_sample;
  }
}

void MatrixManipulation::ICF(const Document &doc, const Kernel& kernel,
                             int rows, int columns, double threshold,
                             ParallelMatrix *icf) {
  CHECK(icf);
  ParallelInterface *mpi = ParallelInterface::GetParallelInterface();
  int parallel_id = mpi->GetProcId();
  int pnum = mpi->GetNumProcs();
  icf->Init(rows, columns);
  int local_rows = mpi->ComputeNumLocal(rows);
  int* pivot = new int[columns];
  bool* pivot_selected = new bool[local_rows];

  // diag1: the diagonal part of Q (the kernal matrix diagonal
  // diag2: the quadratic sum of a row of the ICF matrix
  double* diag1 = new double[local_rows];
  double* diag2 = new double[local_rows];
  for (int i = 0; i < local_rows; ++i) {
    diag1[i] = kernel.CalcKernelWithLabel(*doc.GetLocalSample(i),
                                          *doc.GetLocalSample(i));
    diag2[i] = 0;
    pivot_selected[i] = 0;
  }

  double* header_row = new double[columns];
  for (int column = 0; column < columns; column++) {
    // Get global trace
    double local_trace = 0;
    for (int i = 0; i < local_rows; i++) {
      // If pivot_selected[i] == true, diag1[i] - diag2[i] == 0,
      // summation is not needed.
      if (pivot_selected[i] == false) local_trace += diag1[i] - diag2[i];
    }
    double global_trace = DBL_MAX;
    mpi->AllReduce(&local_trace, &global_trace, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    // Test stop criterion
    if (global_trace < threshold) {
      icf->SetNumCols(column);
      if (parallel_id == 0) {
        cout << "reset columns from " << columns
                  << " to " << icf->GetNumCols();
      }
      break;
    }

    // Find local pivot
    Pivot local_pivot;
    local_pivot.pivot_value = -DBL_MAX;
    local_pivot.pivot_index = -1;
    for (int i = 0; i < local_rows; ++i) {
      double tmp = diag1[i] - diag2[i];
      if (pivot_selected[i] == false && tmp > local_pivot.pivot_value) {
        local_pivot.pivot_index = mpi->ComputeLocalToGlobal(i, parallel_id);
        local_pivot.pivot_value = tmp;
      }
    }

    // Get global pivot (MPI_Reduce is used)
    Pivot global_pivot;
    mpi->AllReduce(&local_pivot, &global_pivot, 1,
                         MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    // Update pivot vector
    pivot[column] = global_pivot.pivot_index;

    // Broadcast pivot sample
    Sample *pivot_sample = NULL;
    BroadcastPivotSample(doc, global_pivot.pivot_index, &pivot_sample);

    // Broadcast the row corresponds to pivot.
    int header_row_size = column + 1;
    int localRowID = mpi->ComputeGlobalToLocal(global_pivot.pivot_index,
                                                      parallel_id);
    if (localRowID != -1) {
      icf->Set(localRowID, column, sqrt(global_pivot.pivot_value));
      for (int j = 0; j < header_row_size; ++j) {
        header_row[j] = icf->Get(localRowID, j);
      }

      mpi->Bcast(header_row, header_row_size,
                 MPI_DOUBLE, parallel_id, MPI_COMM_WORLD);

      // Update pivot flag vector
      pivot_selected[localRowID] = true;
    } else {
      int root = global_pivot.pivot_index % pnum;
      mpi->Bcast(header_row, header_row_size,
                 MPI_DOUBLE, root, MPI_COMM_WORLD);
    }

    // Calculate the column'th column
    // Note: 1. This order can improve numerical accuracy.
    //       2. Cache is used, will be faster too.
    for (int i = 0; i < local_rows; ++i) {
      if (pivot_selected[i] == false) {
        icf->Set(i, column, 0);
      }
    }
    for (int k = 0; k < column; ++k) {
      for (int i = 0; i < local_rows; ++i) {
        if (pivot_selected[i] == false) {
          icf->Set(i, column, icf->Get(i, column) -
                          icf->Get(i, k) * header_row[k]);
        }
      }
    }
    for (int i = 0; i < local_rows; ++i) {
      if (pivot_selected[i] == false) {
        icf->Set(i, column, icf->Get(i, column) +
          kernel.CalcKernelWithLabel(*doc.GetLocalSample(i), *pivot_sample));
      }
    }
    for (int i = 0; i < local_rows; ++i) {
      if (pivot_selected[i] == false) {
        icf->Set(i, column, icf->Get(i, column)/header_row[column]);
      }
    }

    // Free pivot sample
    FreePivotSample(pivot_sample, global_pivot.pivot_index);

    // Update diagonal
    for (int i = 0; i < local_rows; ++i) {
      diag2[i] += icf->Get(i, column) * icf->Get(i, column);
    }
  }
  delete[] pivot;
  delete[] pivot_selected;
  delete[] diag1;
  delete[] diag2;
  delete[] header_row;
}

void MatrixManipulation::ProductMM(const ParallelMatrix &icf,
                                   const double *diagonal,
                                   LLMatrix *product) {
  CHECK(product);
  int row = icf.GetNumLocalRows();
  int column = icf.GetNumCols();
  double* buff = new double[max(row, (column + 1) * column / 2)];
  double* result = new double[(column + 1) * column / 2];
  int offset = 0;
  for (int i = 0; i < column; ++i) {
    offset += i;
    for (int p = 0; p < row; ++p) {
      buff[p] = icf.Get(p, i) * diagonal[p];
    }
    for (int j = 0; j <= i; ++j) {
      double tmp = 0;
      for (int p = 0; p < row; ++p) {
        tmp += buff[p] * icf.Get(p, j);
      }
      result[offset+j] = tmp;
    }
  }
  ParallelInterface *pinterface = ParallelInterface::GetParallelInterface();
  // Sum reduce the column*column matrix on each computer to Computer of ID:0
  pinterface->Reduce(result, buff, (column + 1) * column / 2,
                     MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (pinterface->GetProcId() == 0) {
    int disp = 0;
    for (int i = 0; i < column; ++i) {
      for (int j = 0; j <= i; ++j) {
        product->Set(i, j, buff[disp++] + (i == j ? 1 : 0));
      }
    }
  }
  delete[] buff;
  delete[] result;
}

void MatrixManipulation::CholBackwardSub(const LLMatrix &a, const double *b,
                                         double *x) {
  int dim = a.GetDim();
  for (int k = dim - 1; k >= 0; --k) {
    double tmp = b[k];
    for (int i = k + 1; i < dim; ++i) {
      tmp -= x[i] * a.Get(i, k);
    }
    x[k] = tmp / a.Get(k, k);
  }
}

void MatrixManipulation::CholForwardSub(const LLMatrix &a, const double *b,
                                        double *x) {
  int dim = a.GetDim();
  for (int k = 0; k < dim; ++k) {
    double tmp = b[k];
    for (int i = 0; i < k; ++i) {
      tmp -= x[i] * a.Get(k, i);
    }
    x[k] = tmp / a.Get(k, k);
  }
}

}  // namespace psvm
