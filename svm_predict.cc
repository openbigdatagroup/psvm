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

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <utility>
#include <string>
#include <vector>
#include "svm_predict.h"
#include "document.h"
#include "timer.h"
#include "util.h"
#include "io.h"
#include "parallel_interface.h"

namespace psvm {
void SvmPredictor::ReadModel(const char* model_file) {
  model_.Load(model_file, "model");
}

std::string SvmPredictor::PrintTimeInfo() {
  std::string str = "========== Predicting Time Statistics ==========\n";
  str += "Total            : "
         + PredictingTimeProfile::total.PrintInfo() + "\n";
  str += "1. Load Model    : "
         + PredictingTimeProfile::read_model.PrintInfo() + "\n";
  str += "2. Load Test Data: "
         + PredictingTimeProfile::read_test_doc.PrintInfo() + "\n";
  str += "3. Predict       : "
         + PredictingTimeProfile::predict.PrintInfo() + "\n";

  return str;
}

void SvmPredictor::SaveTimeInfo(const char *path, const char* file_name) {
  ParallelInterface *mpi = ParallelInterface::GetParallelInterface();
  int proc_id   = mpi->GetProcId();

  // Open file file_name.# for writing
  char filename[4096];
  snprintf(filename, sizeof(filename), "%s/%s.%d", path, file_name, proc_id);
  File* obuf = File::OpenOrDie(filename, "w");

  std::string str = PrintTimeInfo();
  CHECK(obuf->WriteString(str) == str.length());

  CHECK(obuf->Flush());
  CHECK(obuf->Close());
  delete obuf;
}

void SvmPredictor::PredictDocument(const char* testdata_filename,
                                   const char* predict_filename,
                                   int chunk_size,
                                   EvaluationResult *result) {
  ParallelInterface* interface = ParallelInterface::GetParallelInterface();
  int myid = interface->GetProcId();

  const Kernel *kernel = model_.kernel();
  const SupportVector *support_vector = model_.support_vector();

  double *local_prediction = new double[chunk_size];
  int  *label = new int[chunk_size];
  double *global_prediction = NULL;
  if (myid == 0) global_prediction = new double[chunk_size];

  int num_total_document = 0;
  int num_positive_positive = 0;  // Number of samples whose original class
  // label is positive and predicted as
  // positive.
  int num_positive_negative = 0;
  int num_negative_positive = 0;
  int num_negative_negative = 0;
  int num_parsed_samples = 0;

  string line;

  File *outputbuffer_predict = NULL;

  if (myid == 0) {
    // Open the predict file for writing.
    outputbuffer_predict = File::Open(predict_filename, "w");
  }

  File* reader = File::OpenOrDie(testdata_filename, "r");
  bool end_of_file = false;
  char sz_line[256];
  while (!end_of_file) {
    // Read a line
    PredictingTimeProfile::read_test_doc.Start();
    end_of_file = !reader->ReadLine(&line);
    PredictingTimeProfile::read_test_doc.Stop();

    // If the local prediction of chunk_size samples is ready or if there are no
    // samples left to predict, performs a reduce to obtain global prediction
    // result.
    if (num_parsed_samples == chunk_size || end_of_file) {
      // Processor 0 will have the global prediction result.
      interface->Reduce(local_prediction, global_prediction, num_parsed_samples,
                        MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      if (myid == 0) {
        for (int i = 0; i < num_parsed_samples; i++) {
          double predicted_label = global_prediction[i] + support_vector->b;

          // Output predicted values.
          size_t b = snprintf(sz_line, sizeof(sz_line), "%d %f\n",
                              label[i], predicted_label);
          CHECK(outputbuffer_predict->Write(sz_line, b) == b);

          // Updates the counters
          if (label[i] > 0) {
            if (predicted_label > 0)
              ++num_positive_positive;
            else
              ++num_positive_negative;
          } else {
            if (predicted_label > 0)
              ++num_negative_positive;
            else
              ++num_negative_negative;
          }
        }

        // Flush buffer per batch
        if (!outputbuffer_predict->Flush()) {
          cerr << "Flushing " << predict_filename << " failed!";
          exit(1);
        }

        // Print the number of predicted samples.
        cout << "Finished: " << num_total_document << endl;
      }
      num_parsed_samples = 0;

    }
    if (end_of_file) break;

    // Parses one sample
    const char* start = line.c_str();
    if (!SplitOneIntToken(&start, " ", &label[num_parsed_samples])) {
      return;
    }
    vector<pair<string, string> > kv_pairs;
    SplitStringIntoKeyValuePairs(string(start), ":", " ",
                                 &kv_pairs);
    Sample sample;
    sample.label = label[num_parsed_samples];
    sample.two_norm_sq = 0.0;
    vector<pair<string, string> >::const_iterator pair_iter;
    for (pair_iter = kv_pairs.begin(); pair_iter != kv_pairs.end();
         ++pair_iter) {
      Feature feature;
      feature.id = atoi(pair_iter->first.c_str());
      feature.weight = atof(pair_iter->second.c_str());
      sample.features.push_back(feature);
      sample.two_norm_sq += feature.weight * feature.weight;
    }

    // Tests the sample based on local support vectors
    double predicted_label = 0.0;
    double temp;
    for (int j = 0; j < support_vector->num_sv; ++j) {
      temp = kernel->CalcKernel(sample, support_vector->sv_data_test[j]);
      predicted_label += support_vector->sv_alpha[j] * temp;
    }
    local_prediction[num_parsed_samples] = predicted_label;

    // Moves on to next sample
    ++num_parsed_samples;
    ++num_total_document;
  }

  // Cleans up
  reader->Close();
  delete reader;
  delete [] local_prediction;
  delete [] label;
  if (myid == 0) {
    delete [] global_prediction;
    CHECK(outputbuffer_predict->Close());
    delete outputbuffer_predict;
  }

  // Computes some statistics
  // Record the numbers
  result->num_pos_pos = num_positive_positive;
  result->num_pos_neg = num_positive_negative;
  result->num_neg_pos = num_negative_positive;
  result->num_neg_neg = num_negative_negative;
  result->num_pos = num_positive_positive + num_positive_negative;
  result->num_neg = num_negative_positive + num_negative_negative;
  result->num_total = result->num_pos + result->num_neg;
  // Calculate the precision/recall and accuracy
  int correct = num_positive_positive + num_negative_negative;
  int incorrect = num_negative_positive + num_positive_negative;
  result->accuracy = static_cast<double>(correct) / (correct + incorrect);
  result->positive_precision = static_cast<double>(num_positive_positive) /
      (num_positive_positive + num_negative_positive);
  result->positive_recall = static_cast<double>(num_positive_positive) /
      (num_positive_positive + num_positive_negative);
  result->negative_precision = static_cast<double>(num_negative_negative) /
      (num_positive_negative + num_negative_negative);
  result->negative_recall = static_cast<double>(num_negative_negative) /
      (num_negative_positive + num_negative_negative);
}
}

using namespace psvm;

//=============================================================================
// Parameter Definitions

string FLAGS_model_path = ".";
string FLAGS_output_path = ".";
int FLAGS_batch_size = 10000;
//=============================================================================

void Usage() {
  const char* msg =
      "svm_predict: This program predicts the class labels of samples. Usage:\n"
      "  svm_predict data_file\n"
      "The predict result is saved in data_file.predict.\n"
      "\n"
      "  Flag descriptions:\n"
      "    -batch_size (How many samples to predict in one reduce) type: int32\n"
      "      default: 10000\n"
      "    -model_path (Directory where to load the SVM model.) type: string\n"
      "      default: .\n"
      "    -output_path (Directory where to save the predict result.) type: string\n"
      "      default: .\n";
  cerr << msg;
}

void ParseCommandLine(int* argc, char*** argv) {
  int i;
  for (i = 1; i < *argc; ++i) {
    if ((*argv)[i][0] != '-') break;
    if (++i >= *argc) {
      Usage();
      exit(1);
    }

    char* param_name = &(*argv)[i-1][1];
    char* param_value = (*argv)[i];
    if (strcmp(param_name, "batch_size") == 0) {
      FLAGS_batch_size = atoi(param_value);
    } else if (strcmp(param_name, "model_path") == 0) {
      FLAGS_model_path = string(param_value);
    } else if (strcmp(param_name, "output_path") == 0) {
      FLAGS_output_path = string(param_value);
    } else {
      cerr << "Unknown parameter " << param_name << endl;
      Usage();
      exit(2);
    }
  }

  for (int j = 1; i + j - 1 < *argc; ++j) {
    (*argv)[j] = (*argv)[i + j - 1];
  }
  *argc -= i - 1;
}


int main(int argc, char** argv) {
  // Initializes the parallel computing environment
  ParallelInterface* interface = ParallelInterface::GetParallelInterface();
  interface->Init(&argc, &argv);

  ParseCommandLine(&argc, &argv);
  if (FLAGS_batch_size <= 0 || argc != 2) {
    Usage();
    return 3;
  }
  string data_file;  // File which contains samples to predict
  data_file = argv[1];

  // Begin Timing
  PredictingTimeProfile::total.Start();

  // Loads the SVM model and predicts the samples
  SvmPredictor predictor;
  PredictingTimeProfile::read_model.Start();
  predictor.ReadModel(FLAGS_model_path.c_str());
  PredictingTimeProfile::read_model.Stop();

  // Predict
  EvaluationResult result;
  PredictingTimeProfile::predict.Start();
  predictor.PredictDocument(data_file.c_str(),
                            (FLAGS_output_path + "/PredictResult").c_str(),
                            FLAGS_batch_size,
                            &result);
  PredictingTimeProfile::predict.Stop();
  PredictingTimeProfile::predict.Minus(PredictingTimeProfile::read_test_doc);

  // End Timing
  PredictingTimeProfile::total.Stop();

  // Save time info
  interface->Barrier(MPI_COMM_WORLD);
  if (interface->GetProcId() == 0) {
    cout << "Saving predicting time statistic info ... " << endl;
  }
  //predictor.SaveTimeInfo(FLAGS_model_path.c_str(), "PredictingTimeInfo");

  // Prints the statistical information.
  interface->Barrier(MPI_COMM_WORLD);
  if (interface->GetProcId() == 0) {
    // Print timing info
    cout << endl
              << predictor.PrintTimeInfo()
              << "========== Predict Matrix ==========" << endl
              << "Total: " << result.num_total << "  "
              << "Positive: " << result.num_pos << "  "
              << "Negative: " << result.num_neg << endl
              << "Real\\Predict\tPositive \tNegative" << endl
              << StringPrintf("Positive     \t%-8d \t%-8d",
                 result.num_pos_pos, result.num_pos_neg) << endl
              << StringPrintf("Negtive      \t%-8d \t%-8d",
                 result.num_neg_pos, result.num_neg_neg) << endl
              << "========== Predict Accuracy ==========" << endl
              << "Accuracy          : " << result.accuracy << endl
              << "Positive Precision: " << result.positive_precision << endl
              << "Positive Recall   : " << result.positive_recall << endl
              << "Negative Precision: " << result.negative_precision << endl
              << "Negative Recall   : " << result.negative_recall << endl;
  }

  // Finalizes the parallel computing environment
  interface->Finalize();
  return 0;
}
