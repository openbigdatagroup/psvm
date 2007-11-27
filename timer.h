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

#ifndef TIMER_H__
#define TIMER_H__

#include <string>

using std::string;

namespace psvm {
// Records the total time, computational time, and communicational time between
// Start() and Stop(). Typical use sample:
//   Timer t1;
//   t1.Start();
//     ...
//   t1.Stop();
//
// Users should Call communicationStart() and CommunicationStop() at the
// start and stop point of communication, this will make sure the
// communication time is reliable.
class Timer {
 public:
  Timer();
  // Gets current time
  static double GetCurrentTime();

  // Sets start and stop timing points
  void Start();
  void Stop();

  // Sets start and stop communication timing points
  static void CommunicationStart();
  static void CommunicationStop();

  // Sets start and stop synchronization timing points
  static void SynchronizationStart();
  static void SynchronizationStop();

  // Add the time Timer t contains to me
  void Add(const Timer &t);
  // Minus the time Timer t contains from me
  void Minus(const Timer &t);

  // Gets time statistics (total, computation, communication)
  double total() const {return total_;}
  double communication() const {return communication_;}
  double computation() const {return computation_;}
  double sync() const {return sync_;}

  string PrintInfo() const;
 private:
  // The start time of timing
  double start_;

  // The communication time elapsed when Start() is called
  double communication_start_;

  // The synchronization time elapsed when Start() is called
  double sync_start_;

  // Total time between Start() and Stop()
  double total_;

  // Total computation time between Start() and Stop()
  double computation_;

  // Total communicational time between Start() and Stop()
  double communication_;

  // Total synchronization time between Start() and Stop()
  double sync_;

  // The total communication time from Start() is called
  static double communication_elapsed_;

  // The total synchronization time from Start() is called
  static double sync_elapsed_;

  // The time when the last communication begins
  static double last_communication_begin_time_;

  // The time when the last communication begins
  static double last_sync_begin_time_;
};

struct TrainingTimeProfile {
  static Timer total;   // total training time

  // Global timming variables for SVM
  static Timer read_doc;     // reading document
  static Timer train_model;  // traing time
  static Timer store_model;  // store resulting model

  // The major parts in training phase
  static Timer icf;          // ICF factorization
  static Timer ipm;          // IPM
  static Timer compute_b;    // computing $b$

  // IPM details
  static Timer surrogate_gap;
  static Timer partial_z;
  static Timer check_stop;
  static Timer production;
  static Timer cf;
  static Timer update_variables;
  static Timer check_sv;     // check support vector
  static Timer ipm_misc;
};

struct PredictingTimeProfile {
  static Timer total;

  static Timer read_model;
  static Timer read_test_doc;
  static Timer predict;
};
}
#endif
