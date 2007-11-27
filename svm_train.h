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

#ifndef SVM_TRAIN_H__
#define SVM_TRAIN_H__

#include <string>

namespace psvm {
class Document;
class Kernel;
class Model;
class PrimalDualIPMParameter;

// Trains a document with given kernel and IPM parameters.  The result is a
// model structure including kernel and support vectors. Sample usage:
//    See main in svm_train.cc
class SvmTrainer {
 public:
  void TrainModel(const Document& doc, const Kernel& kernel,
                  const PrimalDualIPMParameter& p, Model* model,
                  bool failsafe);

  // Format the training time info of current processor to a string.
  std::string PrintTimeInfo();

  // Save time statistic information into a file.
  // For processor #, it is stored in "path/file_name.#"
  void SaveTimeInfo(const char *path, const char* file_name);
};
}
#endif
