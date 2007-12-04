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

#include <cmath>

#include "matrix.h"
#include "document.h"
#include "kernel.h"
#include "pd_ipm_parm.h"
#include "util.h"
#include "io.h"
#include "parallel_interface.h"

using namespace std;

namespace psvm {
ParallelMatrix::ParallelMatrix()
    : num_rows_(0),
      num_cols_(0),
      element_(NULL) {
}

ParallelMatrix::ParallelMatrix(int num_rows, int num_cols)
    : num_rows_(num_rows),
      num_cols_(num_cols),
      element_(NULL) {
  Init(num_rows, num_cols);
}

ParallelMatrix::~ParallelMatrix() {
  Destroy();
}

void ParallelMatrix::SetNumCols(int num_cols) {
  CHECK_LE(num_cols, num_cols_);  // We only support shinking columns
  if (element_ != NULL) {
    double** newcol = new double*[num_cols];
    for (int i = 0; i < num_cols_; ++i) {
      if (i < num_cols) {
        newcol[i] = element_[i];
      } else {
        delete[] element_[i];
      }
    }
    delete[] element_;
    element_ = newcol;
  }
  num_cols_ = num_cols;
}
void ParallelMatrix::Init(int num_rows, int num_cols) {
  Destroy();
  num_rows_ = num_rows;
  num_cols_ = num_cols;
  ParallelInterface* pinterface = ParallelInterface::GetParallelInterface();
  num_local_rows_ = pinterface->ComputeNumLocal(num_rows_);
  element_ = new double* [num_cols_];
  for (int i = 0; i < num_cols_; ++i) {
    element_[i] = new double[num_local_rows_];
    memset(element_[i], 0, sizeof(**element_) * num_local_rows_);
  }
}
void ParallelMatrix::Destroy() {
  if (element_) {
    for (int i = 0; i < num_cols_; ++i) {
      delete[] element_[i];
    }
    delete[] element_;
    element_ = NULL;
  }
}

// Save local matrix into a file.
// For processor #, "path/file_name.#" stores the local matrix
// belonging to processor #.
void ParallelMatrix::Save(const char *path, const char* file_name) {
  ParallelInterface *mpi = ParallelInterface::GetParallelInterface();
  int proc_id   = mpi->GetProcId();
  int num_procs = mpi->GetNumProcs();

  // Open file file_name.# for writing
  char filename[4096];
  snprintf(filename, sizeof(filename), "%s/%s.%d", path, file_name, proc_id);
  File* obuf = File::OpenOrDie(filename, "w");

  // Write header info
  cout << "Saving matrix header to " << filename << "... " << endl;
  CHECK(Write(obuf, &num_procs, sizeof(num_procs)));
  CHECK(Write(obuf, &num_rows_, sizeof(num_rows_)));
  CHECK(Write(obuf, &num_cols_, sizeof(num_cols_)));

  // Write local matrix content
  cout << "Saving matrix content to " << filename <<"... " << endl;
  for (int i = 0; i < num_local_rows_; i++) {
    for (int j = 0; j < num_cols_; j++) {
      double h_ij = Get(i, j);
      CHECK(Write(obuf, &h_ij, sizeof(h_ij)));
    }
  }
  cout << "done" << endl;

  CHECK(obuf->Flush());
  CHECK(obuf->Close());
  delete obuf;
}

// Load matrix from directory "path".
// If no available matrix chunks, return false.
bool ParallelMatrix::Load(const char *path, const char* file_name) {
  ParallelInterface *mpi = ParallelInterface::GetParallelInterface();
  int num_procs = mpi->GetNumProcs();

  // If no available matrix chunks, return false.
  char filename[4096];
  if (mpi->GetProcId() == 0) cout << "Tring to load " << file_name << endl;
  snprintf(filename, sizeof(filename), "%s/%s.0", path, file_name);
  File *file_0 = File::Open(filename, "rb");
  if (file_0 == NULL) {
    if (mpi->GetProcId() == 0) {
      cout << "matrix chunck file\""
                << filename
                << "\" does not exist!"
                << endl;
    }
    return false;
  }

  // Read header info
  int num_original_procs;
  int num_original_rows;
  int num_original_cols;
  if (mpi->GetProcId() == 0) cout << "Reading header info ... " << endl;
  if (file_0->Size() < sizeof(num_original_procs) * 3) {
    if (mpi->GetProcId() == 0) {
      cout << "matrix chunck file\""
                << filename
                << "\" is too small in file size, maybe have been damaged!"
                << endl;
    }
    return false;
  } else {
    if (!Read(file_0, &num_original_procs, sizeof(num_original_procs)) ||
        !Read(file_0, &num_original_rows, sizeof(num_original_rows)) ||
        !Read(file_0, &num_original_cols, sizeof(num_original_cols))) {
      CHECK(file_0->Close());
      delete file_0;
      if (mpi->GetProcId() == 0) {
        cout << "Truncated matrix file '" << filename << "'. "
                  << "Failed to load" << endl;
      }
      return false;
    }
  }
  CHECK(file_0->Close());
  delete file_0;

  if (num_original_procs == num_procs) {
    // The simplest case, the number of processors is the same as the number
    // of processors. Read matrix chunks directly.
    if (mpi->GetProcId() == 0) cout << "Reading trunks ... " << endl;
    return ReadChunks(path, file_name);
  } else {
    cerr << "The number of processes used to predict is different from"
            "the number of processes used to train" << endl;
    exit(1);
    return false;
  }
}

// 1. Open chunk file
// 2. Check the file size
// 3. Initialize matrix
// 4. Read corresponding chunk
bool ParallelMatrix::ReadChunks(const char *path, const char* file_name) {
  ParallelInterface *mpi = ParallelInterface::GetParallelInterface();
  int proc_id   = mpi->GetProcId();

  // Open matrix chunks
  char filename[4096];
  snprintf(filename, sizeof(filename), "%s/%s.%d", path, file_name, proc_id);
  File *file = File::Open(filename, "rb");
  if (file == NULL) {
    if (mpi->GetProcId() == 0) {
      cout << "matrix chunck file \"" << filename
                << "\" does not exist" << endl;
    }
    return false;
  }

  // Read chunk header.
  int num_procs;
  int num_rows;
  int num_cols;
  if (!Read(file, &num_procs, sizeof(num_procs)) ||
      !Read(file, &num_rows, sizeof(num_rows)) ||
      !Read(file, &num_cols, sizeof(num_cols))) {
    if (mpi->GetProcId() == 0) {
      cout << "matrix chunck file \"" << filename
                << "\" is too small in file size, maybe have been damaged!"
                << endl;
    }
    return false;
  }

  // Check number of processors
  if (num_procs != mpi->GetNumProcs()) {
    if (mpi->GetProcId() == 0) cout << "Number of processors differs!";
    return false;
  }

  // Check chunk size
  int num_local_rows = mpi->ComputeNumLocal(num_rows);
  long long num_file_size = sizeof(num_procs) * 3 +
      num_local_rows * num_cols * sizeof(element_[0][0]);
  if (file->Size() != num_file_size) {
    cout << "The size of matrix chunck file \""
              << filename
              << "\" is not correct!" << endl
              << " Expected Size: " << num_file_size
              << " Real Size:" << file->Size() << endl;
    return false;
  }


  // Initialize matrix
  Init(num_rows, num_cols);

  // Read matrix chunk
  for (int i = 0; i < num_local_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      double h_ij;
      if (!Read(file, &h_ij, sizeof(h_ij)))
        return false;
      Set(i, j, h_ij);
    }
  }
  file->Close();
  delete[] file;

  return true;
}

bool ParallelMatrix::Read(File* inbuf, void* buf, size_t size) {
  return (inbuf->Read(buf, size) == size);
}

bool ParallelMatrix::Write(File* obuf, void* buf, size_t size) {
  return (obuf->Write(buf, size) == size);
}

LLMatrix::LLMatrix()
    : dim_(0),
      element_(NULL) {
}

LLMatrix::LLMatrix(int dim)
    : dim_(dim),
      element_(NULL) {
  Init(dim);
}

LLMatrix::~LLMatrix() {
  Destroy();
}

void LLMatrix::Destroy() {
  if (element_ != NULL) {
    for (int i = 0; i < dim_; ++i) {
      delete[] element_[i];
    }
    delete[] element_;
    element_ = NULL;
  }
}

void LLMatrix::Init(int dim) {
  Destroy();
  dim_ = dim;
  element_ = new double*[dim];
  for (int i = 0; i < dim; ++i) {
    element_[i] = new double[dim-i];
    memset(element_[i], 0, sizeof(**element_) * (dim-i));
  }
}
}
