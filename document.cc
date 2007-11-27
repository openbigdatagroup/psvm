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

#include <vector>
#include <string>
#include <utility>
#include <iostream>

#include "document.h"
#include "util.h"
#include "io.h"
#include "parallel_interface.h"

namespace psvm {
Document::Document(): num_total_(0), interface_(NULL) { }

// Stores the class labels of the samples in samples_ to the array specified by
// 'labels'. It is the caller's reposibility to allocate enough memory for the
// labels.
void Document::GetLocalLabels(int* labels) const {
  for (size_t i = 0; i < samples_.size(); ++i) {
    labels[i] = samples_[i].label;
  }
}

// Reads samples from the file specified by filename. The samples are evenly
// distributed accross all the processors. See the comment for 'Read' in the
// header file.
bool Document::Read(const char* filename) {
  num_total_ = 0;
  num_pos_   = 0;
  num_neg_   = 0;
  if (filename == NULL) return false;

  // Obtains the parallel interface
  interface_ = ParallelInterface::GetParallelInterface();
  int myid = interface_->GetProcId();
  int num_processors = interface_->GetNumProcs();
  // Walks through each sample
  File* file = File::Open(filename, "r");
  if (file == NULL) {
    cerr << "Cannot find file " << filename << endl;
    return false;
  }
  string line;
  int num_local_pos = 0;
  int num_local_neg = 0;
  while (file->ReadLine(&line)) {
    // If the sample should be assigned to this processor
    if (num_total_ % num_processors == myid) {
      int label = 0;
      const char* start = line.c_str();
      // Extracts the sample's class label
      if (!SplitOneIntToken(&start, " ", &label)) {
        cerr << "Error parsing line: " << num_total_ + 1 << endl;
        return false;
      }

      // Gets the local number of positive and negative samples
      if (label == 1) {
        ++num_local_pos;
      } else if (label == -1) {
        ++num_local_neg;
      } else {
        cerr << "Unknow label in line: " << num_total_ + 1 << label;
        return false;
      }

      // Creates a "Sample" and add to the end of samples_
      samples_.push_back(Sample());
      Sample& sample = samples_.back();
      sample.label = label;
      sample.id = num_total_;  // Currently num_total_ == sample id
      sample.two_norm_sq = 0.0;

      // Extracts the sample's features
      vector<pair<string, string> > kv_pairs;
      SplitStringIntoKeyValuePairs(string(start), ":", " ", &kv_pairs);
      vector<pair<string, string> >::const_iterator pair_iter;
      for (pair_iter = kv_pairs.begin(); pair_iter != kv_pairs.end();
           ++pair_iter) {
        Feature feature;
        feature.id = atoi(pair_iter->first.c_str());
        feature.weight = atof(pair_iter->second.c_str());
        sample.features.push_back(feature);
        sample.two_norm_sq += feature.weight * feature.weight;
      }
    }
    ++num_total_;
  }
  file->Close();
  delete file;

  // Get the global number of positive and negative samples
  int local[2];
  int global[2];
  local[0] = num_local_pos;
  local[1] = num_local_neg;
  memset(global, 0, sizeof(global[0] * 2));
  interface_->AllReduce(local, global, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  num_pos_ = global[0];
  num_neg_ = global[1];

  return true;
}

// If global_row_index points a valid position and the coressponding sample
// is assigned to this processor, returns a const pointer to it.
const Sample* Document::GetGlobalSample(int global_row_index) const {
  int num_processors = interface_->GetNumProcs();
  int myid = interface_->GetProcId();
  int local_row_index = global_row_index / num_processors;
  int expected_processor_id = global_row_index % num_processors;

  // If local_row_index points to a illegal position, returns NULL.
  if (local_row_index < 0 || local_row_index >= (int)samples_.size())
    return NULL;

  // If the global_row_index'th sample is not assigned to this processor,
  // returns NULL.
  if (expected_processor_id != myid) return NULL;

  // Otherwise returns a const pointer to the sample.
  return &(samples_[local_row_index]);
}

const Sample* Document::GetLocalSample(int local_row_index) const {
  // If local_row_index points to a illegal position, returns NULL.
  if (local_row_index < 0 || local_row_index >= (int)samples_.size()) {
    return NULL;
  }

  // Otherwise returns a const pointer to the sample.
  return &(samples_[local_row_index]);
}

size_t Document::GetPackSize(const Sample & sample) {
  // Size of the first three data members of Sample
  int size_buffer = sizeof(sample.id)
      + sizeof(sample.label)
      + sizeof(sample.two_norm_sq);

  // Size of num_features. We need to encode it to indicate how many
  // features there are in the memory block.
  size_t num_features = sample.features.size();
  size_buffer += sizeof(num_features);

  // Size of sample.features, which is (sizeof(id) + sizeof(weight)
  // * num_features. We do not use sizeof(Feature) since the two
  // are not always equal. See the definition of Feature in the header
  // file.
  size_buffer += (sizeof(sample.features[0].id) +
                  sizeof(sample.features[0].weight)) * num_features;

  return size_buffer;
}

size_t Document::PackSample(char *&buffer, const Sample &sample) {
  // buffer should be a pre-allocated memory block, with proper block size.
  // If buffer is not allocated, say "buffer == NULL", then allocates memory
  if (buffer == NULL) {
    size_t size_buffer = GetPackSize(sample);
    buffer = new char[size_buffer];
  }

  size_t offset = 0;

  // Encodes the first three data members of sample
  memcpy(buffer + offset, &(sample.id), sizeof(sample.id));
  offset += sizeof(sample.id);
  memcpy(buffer + offset, &(sample.label), sizeof(sample.label));
  offset += sizeof(sample.label);
  memcpy(buffer + offset, &(sample.two_norm_sq), sizeof(sample.two_norm_sq));
  offset += sizeof(sample.two_norm_sq);

  // Encodes num_features
  size_t num_features = sample.features.size();
  memcpy(buffer + offset, &num_features, sizeof(num_features));
  offset += sizeof(num_features);

  // Encodes sample.features
  for (size_t i = 0; i < num_features; ++i) {
    // Encodes one feature
    memcpy(buffer + offset,
           &(sample.features[i].id),
           sizeof(sample.features[i].id));
    offset += sizeof(sample.features[i].id);
    memcpy(buffer + offset,
           &(sample.features[i].weight),
           sizeof(sample.features[i].weight));
    offset += sizeof(sample.features[i].weight);
  }

  return offset;
}

size_t Document::UnpackSample(Sample *&sample, const char *buffer) {
  size_t offset = 0;
  if (sample == NULL) sample = new Sample;

  // Decodes the first three data members of sample
  memcpy(&(sample->id), buffer + offset, sizeof(sample->id));
  offset += sizeof(sample->id);
  memcpy(&(sample->label), buffer + offset, sizeof(sample->label));
  offset += sizeof(sample->label);
  memcpy(&(sample->two_norm_sq), buffer + offset, sizeof(sample->two_norm_sq));
  offset += sizeof(sample->two_norm_sq);

  // Decodes num_features
  size_t num_features;
  memcpy(&num_features, buffer + offset, sizeof(num_features));
  offset += sizeof(num_features);

  // Decodes sample.features
  for (size_t i = 0; i < num_features; ++i) {
    Feature feature;

    // Decodes one feature
    memcpy(&(feature.id), buffer + offset, sizeof(feature.id));
    offset += sizeof(feature.id);
    memcpy(&(feature.weight), buffer + offset, sizeof(feature.weight));
    offset += sizeof(feature.weight);

    sample->features.push_back(feature);
  }

  return offset;
}
}
