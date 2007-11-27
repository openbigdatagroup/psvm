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

#ifndef MODEL_H__
#define MODEL_H__

#include <vector>
#include "kernel.h"

using namespace std;

namespace psvm {
struct Sample;    // The structure that stores a sample.
class Document;   // The class that encapsulates a dataset.
class PrimalDualIPMParameter;  // The parameters for the primal-dual problem.
class ParallelInterface;  // The mpi parallel interface

// Stores the Support Vector information.
struct SupportVector {
  int num_sv;           // number of support vectors
  int num_bsv;          // number of support vectors at boundary
  double b;             // b value of classification function in SVM model
  vector<double> sv_alpha;        // the alpha values of the support vectors
  vector<const Sample*> sv_data;  // the pointers to support vectors,
                                  // used only in training phase.
  vector<Sample> sv_data_test;    // support vetor samples,
                                  // used only in  predicting phase.
};

// Stores the training result of pd-ipm. usage:
//    Model model;
//    model.SetKernel(kernel);
//    model.CheckSupportVector(alpha, doc, ipm_parameter);
//    model.ComputeB(ipm_parameter, &num_total_sv);
//    model.save("model_directory");
class Model {
 public:
  Model();
  virtual ~Model();

  // Uses alpha values to decide which samples are support vectors and stores
  // their information.
  void CheckSupportVector(double* alpha,
                          const Document& doc,
                          const PrimalDualIPMParameter& ipm_parameter);

  const SupportVector* support_vector() const { return &support_vector_; }

  // Accessors to kernel_.
  const Kernel* kernel() const { return &kernel_; }
  void set_kernel(const Kernel& kernel) { kernel_ = kernel; }

  // Saves the model to the directory specified by str_directory.
  void Save(const char* str_directory, const char* model_name);
  void SaveHeader(const char* str_directory, const char* model_name);
  void SaveChunks(const char* str_directory, const char* model_name);

  // Loads the model from the directory specified by str_directory.
  void Load(const char* str_directory, const char* model_name);
  void LoadHeader(const char* str_directory, const char* model_name);
  void LoadChunks(const char* str_directory, const char* model_name);

  // Computes the b value of the SVM's classification function.
  void ComputeB(const PrimalDualIPMParameter& ipm_parameter);

 private:
  // The parallel interface pointer, for example MPICH2. This class data
  // member is initialized in the construtor. The interface should provide
  // message passing interfaces, such as Broadcast(), Reduce(), etc.
  ParallelInterface *mpi_;

  // kernel_ stores kernel type, kernel parameter information,
  // and calculates kernel function accordingly.
  Kernel kernel_;

  // The number of support vectors in all.
  int num_total_sv_;

  // The number of chunks/processors.
  int num_chunks_;

  // support_vector_ stores support vector information.
  // In training phase, it stores the pointers to the suppor vectors,
  // In testing phase, it stores the support vectors read from model files.
  SupportVector support_vector_;
};
}

#endif
