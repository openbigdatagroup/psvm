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

#ifndef DOCUMENT_H__
#define DOCUMENT_H__

#include <vector>

using namespace std;

// Reads samples according to processor id and provides methods for accessing
// them. Suppose there are N processors, the first processor will read the 0th,
// Nth, 2Nth, ... samples, the second processor will read the first, (N+1)th,
// (2N+1)th, ... samples, and so forth. Sample usage:
//    Document document();
//    document.Read("sample.dat");
//    const Sample* sample = document.GetLocalSample(0);
//    const Feature& feature = sample.features[0];
namespace psvm {
class ParallelInterface;

// Stores the properties of a feature, including its word id and its weight.
struct Feature {
  int id;
  double weight;
};

// Stores the properties of a document sample, including its document id,
// class label, the square of its two norm and all its features.
struct Sample {
  int id;
  int label;
  double two_norm_sq;
  vector<Feature> features;
};

// See comment at top of file for a complete description
class Document {
 public:
  Document();

  // Reads samples from the file specified by filename. If the file does not
  // exist or the file format is illegal, false is returned. Otherwise true
  // is returned. The file format whould strickly be:
  //    label word-id:word-weight word-id:word-weight ...
  //    label word-id:word-weight word-id:word-weight ...
  //    ...
  // Each line in the file corresponds to one sample. The samples will be
  // evenly distributed across all the processors. Suppose there are N
  // processors, with processor ids 0, 1, ..., (N-1). Then processor 0 will
  // read the 0th, Nth, 2Nth, ... samples from the file, processor 1 will read
  // the first, (N+1)th, (2N+1)th, ... samples form the file, and so forth.
  bool Read(const char* filename);

  // Returns a const pointer to the local_row_index'th sample. But if
  // local_row_index is less then 0 or points to a non-existent position, NULL
  // will be returned.
  const Sample* GetLocalSample(int local_row_index) const;

  // Returns a const pointer to the global_row_index'th sample. But when any of
  // the following conditions is satisfied, NULL will be returned:
  //    1. global_row_index is less then 0 or points to a non-existent
  //       position.
  //    2. The global_row_index'th sample is not assigned to this processor.
  //       (See comment of method 'Read')
  const Sample* GetGlobalSample(int global_row_index) const;

  // Frees the memory occupied by the samples assigned to this processor.
  void Destroy() { samples_.clear(); }

  // Returns the number of the samples assigned to this processor.
  int GetLocalNumberRows() const { return samples_.size(); }

  // Returns the total number of samples.
  int GetGlobalNumberRows() const { return num_total_; }

  // Returns the total number of positive samples.
  int num_pos() const { return num_pos_; }

  // Returns the total number of negative samples.
  int num_neg() const { return num_neg_; }

  // Copies the labels of the samples assigned to this processor to the array
  // specified by the output parameter 'labels'. The class labels will be
  // stored in the same order as the samples. It is the caller's responsibility
  // to allocate enough memory for the labels.
  void GetLocalLabels(int* labels) const;

  // The following methods are used to encode Sample to or decode Sample from
  // a memory block, which is used to transfer Sample in the network.

  // Computes the size of the memory block needed to encode sample to.
  static size_t GetPackSize(const Sample & sample);

  // Packs a Sample into 'buffer'. If buffer != NULL, it should be a
  // pre-allocated memory block, with proper block size. Otherwise,
  // this method will use GetPackSize to determine how much memory is
  // needed and then allocate enough memory to hold it. It is the caller's
  // responsibility to free the memory. The return value is the number
  // of bytes used in buffer.
  static size_t PackSample(char *&buffer, const Sample &sample);

  // Decodes sample from the memory block pointed to by 'buffer'. If 'sample' is
  // NULL, the method will allocate a new Sample. On return of the method,
  // the decoded Sample is put in the output parameter 'sample'. It's the
  // caller's responsility to free the memory. The method returns how many
  // bytes is decoded from 'buffer'
  static size_t UnpackSample(Sample *&sample, const char *buffer);

 private:
  // Stores the samples assigned to this processor.
  vector<Sample> samples_;

  // keeps track of the total number of samples.
  int num_total_;

  // keeps track of the total number of positive samples.
  int num_pos_;

  // keeps track of the total number of negative samples.
  int num_neg_;

  // An interface to the parallel computing environment.
  ParallelInterface* interface_;
};
}
#endif
