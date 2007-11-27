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

#include <iostream>
#include <vector>
#include <utility>
#include <string>

#include "model.h"

#include "document.h"
#include "pd_ipm_parm.h"
#include "kernel.h"
#include "matrix.h"
#include "io.h"
#include "util.h"
#include "parallel_interface.h"

namespace psvm {
// Initializes mpi_.
Model::Model() {
  mpi_ = ParallelInterface::GetParallelInterface();
}

Model::~Model() {
}

// Checks whether the alpha values (stores in double *alpha) are support
// vecotrs. And fills the support vector object (support_vector_) of
// class Model.
//
// If alpha[i] < epsilon_sv, then the i'th sample is not considered as a
// support vector. if (alpha[i] - C) is smaller than epsilon_sv, than
// alpha[i] is considered as a support vector and regulate as C. Any other
// alpha values between 0 and C will be considered as support vector.
//
// In the weighted case, in which positive and negative samples possess
// different weight, then C equals hyper_parm * weight.
void Model::CheckSupportVector(double *alpha,
                               const Document& doc,
                               const PrimalDualIPMParameter& ipm_parameter) {
  int num_svs = 0;  // number of local support vectors
  int num_bsv = 0;  // number of boundary support vectors
  int num_local_rows = doc.GetLocalNumberRows();  // number of local samples
  // pos_sv[i] stores the index of the i'th support vector.
  int *pos_sv = new int[num_local_rows];

  // Get weighted c for positive and negative samples.
  double c_pos = ipm_parameter.hyper_parm * ipm_parameter.weight_positive;
  double c_neg = ipm_parameter.hyper_parm * ipm_parameter.weight_negative;

  // Check and regulate support vector values.
  for (int i = 0; i < num_local_rows; ++i) {
    if (alpha[i] <= ipm_parameter.epsilon_sv) {
      // If alpha[i] is smaller than epsilon_sv, then assign alpha[i] to be 0,
      // which means samples with small alpha values are considered as
      // non-support-vectors.
      alpha[i] = 0;
    } else {
      // If alpha[i] is near the weighted hyper parameter, than regulate
      // alpha[i] to be the weighted hyper parameter.
      pos_sv[num_svs++] = i;
      const Sample *ptr_sample = doc.GetLocalSample(i);
      double c = (ptr_sample->label > 0) ? c_pos : c_neg;
      if ((c - alpha[i]) <= ipm_parameter.epsilon_sv) {
        alpha[i] = c;
        ++num_bsv;
      }
    }
  }

  // Store support vector information
  support_vector_.num_sv      = num_svs;
  support_vector_.num_bsv     = num_bsv;
  for (int i = 0; i < num_svs; i++) {
    // sv_alpha stores the production of alpha[i] and label[i].
    // sv_alpha[i] = alpha[i] * label[i].
    const Sample *ptr_sample = doc.GetLocalSample(pos_sv[i]);
    if (ptr_sample->label > 0) {
      support_vector_.sv_alpha.push_back(alpha[pos_sv[i]]);
    }
    if (ptr_sample->label < 0) {
      support_vector_.sv_alpha.push_back(-alpha[pos_sv[i]]);
    }
  }
  // If document has samples, then assign sample pointers.
  if (doc.GetLocalNumberRows() != 0) {
    for (int i = 0; i < num_svs; i++) {
      // sv_data stores pointers to support vectors.
      support_vector_.sv_data.push_back(doc.GetLocalSample(pos_sv[i]));
    }
  }

  // Store total number of support vectors.
  mpi_->AllReduce(&support_vector_.num_sv, &num_total_sv_, 1,
                 MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (ipm_parameter.verb >= 1 && mpi_->GetProcId() == 0) {
    cout << "Number of Support Vectors = " << num_total_sv_ << endl;
  }

  // Clean up
  delete [] pos_sv;
}

// Saves model in the "str_directory/" directory.
// several files will be created.
//   model.header: stores general information, including:
//        kernel parameters.
//        the b value of the obtained SVM.
//        the number of support vectors.
//   model.#: "#" will be replaced by 0,1,2,...,etc.
//            each file stores the support vectors of precessor "#" owns.
//            The format is as follows:
//        alpha_value [feature_id:feature_value, ...]
void Model::Save(const char* str_directory, const char* model_name) {
  if (mpi_->GetProcId() ==0) {
    cout << "========== Store Model ==========" << endl;
  }
  // Save model header
  SaveHeader(str_directory, model_name);
  // Make the output order looks good
  mpi_->Barrier(MPI_COMM_WORLD);
  // Output support vector alpha and data
  SaveChunks(str_directory, model_name);
}

void Model::SaveHeader(const char* str_directory, const char* model_name) {
  // Store model header, which will be done by processor #0.
  if (mpi_->GetProcId() == 0) {
    char str_file_name[4096];
    int num_chunks_ = mpi_->GetNumProcs();

    // Create a file for storing model.header
    snprintf(str_file_name, sizeof(str_file_name),
             "%s/%s.header", str_directory, model_name);
    File* obuf = File::OpenOrDie(str_file_name, "w");

    // Output kernel parameters
    cout << "Storing " << model_name << ".header ... " << endl;
    obuf->WriteString(StringPrintf("%d", kernel_.kernel_type()));
    switch (kernel_.kernel_type()) {
      case Kernel::LINEAR:
        obuf->WriteString(StringPrintf("\n"));
        break;
      case Kernel::POLYNOMIAL:
        obuf->WriteString(StringPrintf(" %.8lf %.8lf %d\n",
                         kernel_.coef_lin(),
                         kernel_.coef_const(),
                         kernel_.poly_degree()));
        break;
      case Kernel::GAUSSIAN:
        obuf->WriteString(StringPrintf(" %.8lf\n", kernel_.rbf_gamma()));
        break;
      case Kernel::LAPLACIAN:
        obuf->WriteString(StringPrintf(" %.8lf\n", kernel_.rbf_gamma()));
        break;
      default:
        cerr << "Error: Unknown kernel_ function\n";
        exit(1);
    }
    // Output b value and number of support vectors
    obuf->WriteString(StringPrintf("%.8lf\n", support_vector_.b));
    obuf->WriteString(StringPrintf("%d %d\n", num_total_sv_, num_chunks_));
    CHECK(obuf->Flush());
    CHECK(obuf->Close());
    delete obuf;
  }
}

void Model::SaveChunks(const char* str_directory, const char* model_name) {
  cout << "Storing " << model_name << "." << mpi_->GetProcId() << endl;
  char str_file_name[4096];
  snprintf(str_file_name, sizeof(str_file_name),
           "%s/%s.%d", str_directory, model_name, mpi_->GetProcId());
  File* obuf = File::OpenOrDie(str_file_name, "w");

  // Output the number of support vectors of this model chunk.
  obuf->WriteLine(StringPrintf("%d", support_vector_.num_sv));
  const Sample *ptr_sample;
  for (int i = 0; i < support_vector_.num_sv; i++) {
    ptr_sample = support_vector_.sv_data[i];
    // first, alpha value with label
    obuf->WriteString(StringPrintf("%.8lf", support_vector_.sv_alpha[i]));
    // then, support vector sample
    size_t num_words = ptr_sample->features.size();
    for (size_t j = 0; j < num_words; j++) {
      obuf->WriteString(
          StringPrintf(" %d:%.8lf",
                       ptr_sample->features[j].id,
                       ptr_sample->features[j].weight));
    }
    obuf->WriteLine(string(""));
  }

  CHECK(obuf->Flush());
  CHECK(obuf->Close());
  delete obuf;
}

// Loads model from the "str_directory/" directory.
// For processor # (here # can be 0, 1, 2, ..., etc), two files will be read
//   model.header: stores general information, including:
//        kernel parameters.
//        the b value of the obtained SVM.
//        the number of support vectors.
//   model.#: the support vectors corresponding to processor #.
//            The format is as follows:
//        alpha_value [feature_id:feature_value, ...]
void Model::Load(const char* str_directory, const char* model_name) {
  // Load model header file
  LoadHeader(str_directory, model_name);

  if (mpi_->GetNumProcs() == num_chunks_) {
    // Read model chunks directly
    LoadChunks(str_directory, model_name);
  } else {
    cerr << "The number of processes used to predict is different from the number of processes used to train" << endl;
    exit(1);
  }
}

void Model::LoadHeader(const char*str_directory, const char* model_name) {
  char str_file_name[4096];
  // Load model header file
  snprintf(str_file_name, sizeof(str_file_name),
           "%s/%s.header", str_directory, model_name);
  string line;
  File *reader = File::OpenOrDie(str_file_name, "r");

  // Read kernel_ parameters
  Kernel::KernelType kernel_type;
  int  poly_degree, int_kernel_type;
  double rbf_gamma;
  double coef_lin;
  double coef_const;

  reader->ReadLine(&line);
  const char *buf = line.c_str();
  SplitOneIntToken(&buf, " ", &int_kernel_type);
  kernel_type = static_cast<Kernel::KernelType>(int_kernel_type);
  kernel_.set_kernel_type(kernel_type);
  switch (kernel_type) {
    case Kernel::LINEAR:
      break;
    case Kernel::POLYNOMIAL:
      // polynomial kernel
      SplitOneDoubleToken(&buf, " ", &coef_lin);
      SplitOneDoubleToken(&buf, " ", &coef_const);
      SplitOneIntToken(&buf, "\n", &poly_degree);

      kernel_.set_coef_lin(coef_lin);
      kernel_.set_coef_const(coef_const);
      kernel_.set_poly_degree(poly_degree);
      break;
    case Kernel::GAUSSIAN:
      // gaussian kernel
      SplitOneDoubleToken(&buf, "\n", &rbf_gamma);
      kernel_.set_rbf_gamma(rbf_gamma);
      break;
    case Kernel::LAPLACIAN:
      // laplacian kernel
      SplitOneDoubleToken(&buf, "\n", &rbf_gamma);
      kernel_.set_rbf_gamma(rbf_gamma);
      break;
    default:
      cerr << "Fatal Error: Unknown kernel_ function" << endl;
      exit(1);
  }
  // Read b
  reader->ReadLine(&line);
  buf = line.c_str();
  SplitOneDoubleToken(&buf, "\n", &(support_vector_.b));

  // Read total number of support vectors, and number of chunks
  reader->ReadLine(&line);
  buf = line.c_str();
  SplitOneIntToken(&buf, " ", &num_total_sv_);
  SplitOneIntToken(&buf, "\n", &num_chunks_);

  // Clean up
  reader->Close();
  delete reader;
}

void Model::LoadChunks(const char*str_directory, const char* model_name) {
  char str_file_name[4096];
  snprintf(str_file_name, sizeof(str_file_name),
           "%s/%s.%d", str_directory, model_name, mpi_->GetProcId());
  const char* buf;

  // Check the total number of support vectors
  File *reader = File::OpenOrDie(str_file_name, "r");
  string line;
  reader->ReadLine(&line);
  buf = line.c_str();
  int num_global_sv = 0;
  int num_local_sv = 0;
  SplitOneIntToken(&buf, " ", &num_local_sv);
  support_vector_.num_sv = num_local_sv;
  mpi_->Reduce(&num_local_sv, &num_global_sv, 1,
               MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (mpi_->GetProcId() == 0) {
    if (num_global_sv != num_total_sv_) {
      cerr << model_name << ".# do not compatible with "
                 << model_name << ".dat" << endl
                 << "Expected total #SV:" << num_total_sv_
                 << "Actual total #SV:" << num_global_sv;
      exit(1);
    }
  }

  // Read model.# into sv_data_test
  int num_actual_local_sv = 0;
  while (reader->ReadLine(&line)) {
    buf = line.c_str();
    double alpha;
    SplitOneDoubleToken(&buf, " ", &alpha);
    support_vector_.sv_alpha.push_back(alpha);

    support_vector_.sv_data_test.push_back(Sample());
    Sample& sample = support_vector_.sv_data_test.back();

    sample.id = num_local_sv;
    alpha > 0 ? sample.label = 1 : sample.label = -1;
    sample.two_norm_sq = 0;
    if (buf != NULL) {
      vector<pair<string, string> > kv_pairs;
      SplitStringIntoKeyValuePairs(string(buf), ":", " ", &kv_pairs);
      vector<pair<string, string> >::const_iterator pair_iter;
      for (pair_iter = kv_pairs.begin();
           pair_iter != kv_pairs.end();
           ++pair_iter) {
        Feature feature;
        feature.id = atoi(pair_iter->first.c_str());
        feature.weight = atof(pair_iter->second.c_str());
        sample.features.push_back(feature);
        sample.two_norm_sq += feature.weight * feature.weight;
      }
    }
    ++num_actual_local_sv;
  }

  // Check local number of support vectors
  if (num_actual_local_sv != num_local_sv) {
    cerr << str_file_name << " is broken!"
               << "Expected #SV:" << num_local_sv
               << "\tActual #SV:" << num_actual_local_sv;
    exit(1);
  }

  // Print local support vector number
  cout << mpi_->GetProcId() << ": #support_vector_:"
            << support_vector_.num_sv << endl;

  // Clean up
  CHECK(reader->Close());
  delete reader;
}

// Compute b of the support vector machine.
// Theory:
//   f(x) = sum{alpha[i] * kernel(x, x_i)} + b
//   where x_i is the i'th support vector.
//   Therefore, b can be abtained by substituting x by a support vector.
//   In order to get a robust estimation, we estimate b a few times, and use
//   the average value as an optimal estimation.
//
// Implementation Details:
//   First, each machine provides a few support vectors, these support vectors
// are broadcasted to other machines. All the provided support vectors formed
// a "selected support vector dataset".
//   Second, each computer compute a local b
// value using its local support vectors (stored in local machine),
// that is: sum{alpha[k] * kernel(x,x_k)}
// where x_k is the k'th local support vector. Note that each support vector
// in the selected support vector dataset can obtain such a local b, thus a
// local b value array can be obtained.
//   Third, sum local b value arrays of different machines, a global b value
// array can be obtained, which equals: sum{alpha[i] * kernel(x, x_i)}
//   Finally, the #0 machine get the average b value, save it in
// support_vector_.b
void Model::ComputeB(const PrimalDualIPMParameter& ipm_parameter) {
  int proc_id = mpi_->GetProcId();
  int pnum = mpi_->GetNumProcs();

  if (proc_id == 0) {
    cout << "========== Compute b ==========" << endl;
  }


  if (proc_id == 0) {
    cout << "Computing b ................ " << endl;
  }

  // selected support_vector_ count in computing b.
  int num_selected_sv = std::min(num_total_sv_, 1000);

  // the number of support_vector_ provided by the this computer.
  int num_provided_sv = (num_selected_sv * support_vector_.num_sv) /
                         num_total_sv_;

  // Get localPackSize for packing the support vectors provided by this
  // computer, then get global pack size array by reduction. the global
  // size array is used for calculating memory begin position and total
  // pack size.
  int *local_pack_size  = new int[pnum + 1];
  int *global_pack_size = new int[pnum + 1];
  memset(local_pack_size, 0, (pnum + 1) * sizeof(local_pack_size[0]));
  local_pack_size[pnum] = num_provided_sv;
  int tmp = 0;
  for (int i = 0; i < num_provided_sv; i++) {
    tmp += Document::GetPackSize(*support_vector_.sv_data[i]);
  }
  local_pack_size[proc_id] = tmp;
  mpi_->AllReduce(local_pack_size, global_pack_size, pnum + 1, MPI_INT,
                 MPI_SUM, MPI_COMM_WORLD);
  int num_actual_selected_sv = global_pack_size[pnum];

  // Get global memory size and begin postion for this computer
  int begin_pos = 0;
  int num_total_memory_size = 0;
  for (int i = 0; i < pnum; i++) {
    if (i < proc_id) begin_pos += global_pack_size[i];
    num_total_memory_size += global_pack_size[i];
  }

  // Allocate memory for selected support vectors.
  char *pc_local_memory = new char[num_total_memory_size];
  char *pc_global_memory = new char[num_total_memory_size];
  memset(pc_local_memory, 0, sizeof(pc_local_memory[0])
         * num_total_memory_size);

  // Pack local provided support vectors into pc_local_memory
  int offset = begin_pos;
  char *pc_temp;
  for (int i = 0; i < num_provided_sv; i++) {
    pc_temp = pc_local_memory + offset;
    offset += Document::PackSample(pc_temp, *support_vector_.sv_data[i]);
  }

  // Get global memory by reduction
  mpi_->AllReduce(pc_local_memory, pc_global_memory, num_total_memory_size,
                 MPI_BYTE, MPI_BOR, MPI_COMM_WORLD);

  // Get seleced support vectors by unpacking the memory block.
  Sample **selected_sv = new Sample*[num_actual_selected_sv];
  memset(selected_sv, 0, sizeof(selected_sv[0]) * num_actual_selected_sv);
  offset = 0;
  for (int i = 0; i < num_actual_selected_sv; i++) {
    pc_temp = pc_global_memory + offset;
    offset += Document::UnpackSample(selected_sv[i], pc_temp);
  }

  // Compute b
  double *local_b  = new double[num_actual_selected_sv];
  double *global_b = new double[num_actual_selected_sv];
  for (int i = 0; i < num_actual_selected_sv; i++) {
    // Get local b values
    double sum = 0;
    for (int k = 0; k < support_vector_.num_sv; k++) {
      sum += support_vector_.sv_alpha[k] *
          kernel_.CalcKernel(*support_vector_.sv_data[k],
                             *selected_sv[i]);
    }
    local_b[i] = sum;
  }

  // Get global b values
  mpi_->Reduce(local_b, global_b, num_actual_selected_sv,
              MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // Processor #0 gets the average b value.
  if (proc_id == 0) {
    double b = 0;
    for (int i = 0; i < num_actual_selected_sv; i++)
      b += selected_sv[i]->label - global_b[i];
    b /= num_actual_selected_sv;

    // Output b value.
    support_vector_.b = b;
    cout << "          b = "
              << support_vector_.b
              << endl;
  }

  // Clean up
  delete [] local_pack_size;
  delete [] global_pack_size;
  delete [] pc_local_memory;
  delete [] pc_global_memory;
  delete [] local_b;
  delete [] global_b;

  // Free selected samples
  for (int i = 0; i < num_actual_selected_sv; i++) {
    delete selected_sv[i];
  }
  delete []  selected_sv;
}
}
