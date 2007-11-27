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

#ifndef SVM_PREDICT_H__
#define SVM_PREDICT_H__

#include <string>

#include "model.h"

namespace psvm {
// Structure for evaluating the prediction result. The calculation is
// based on the four values shown below:
//    Real class\Predicted class   POSITIVE  NEGATIVE
//    Positive                     a         b
//    Negative                     c         d
struct EvaluationResult {
  int num_total;
  int num_pos;
  int num_neg;
  int num_pos_pos;
  int num_pos_neg;
  int num_neg_pos;
  int num_neg_neg;

  double positive_precision;  // a/(a+c)
  double positive_recall;     // a/(a+b)
  double negative_precision;  // d/(b+d)
  double negative_recall;     // d/(c+d)
  double accuracy;            // (a+d)/(a+b+c+d)
};

// Predicts the class labels of documents according to the model file.
// Sample usage:
//    SvmPredictor predictor;
//    predictor.ReadModel(modelfilename);
//    EvaluationResult result;
//    predictor.PredictDocument(document, chunksize, &result);
class SvmPredictor {
 public:
  // Loads the model from the model_file
  void ReadModel(const char* model_file);

  // Predicts the class label for the samples in data_file. The result is
  // stored in 'result' on return of the method.
  void PredictDocument(const char* testdata_filename,
                       const char* predict_filename,
                       int chunk_size,
                       EvaluationResult *result);

  // Prints the time information of current processor.
  std::string PrintTimeInfo();

  // Save time statistic information into a file.
  // For processor #, it is stored in "path/file_name.#"
  void SaveTimeInfo(const char *path, const char* file_name);
 private:
  // Stores model information including kernel info and support vectors.
  Model model_;
};
}
#endif
