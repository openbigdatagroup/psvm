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

#include <sys/time.h>
#include <cstdio>
#include <string>
#include "timer.h"

namespace psvm {
double Timer::communication_elapsed_ = 0.0;
double Timer::last_communication_begin_time_ = 0.0;
double Timer::sync_elapsed_ = 0.0;
double Timer::last_sync_begin_time_ = 0.0;

Timer TrainingTimeProfile::total;
Timer TrainingTimeProfile::read_doc;
Timer TrainingTimeProfile::train_model;
Timer TrainingTimeProfile::icf;
Timer TrainingTimeProfile::ipm;
Timer TrainingTimeProfile::compute_b;
Timer TrainingTimeProfile::store_model;
Timer TrainingTimeProfile::surrogate_gap;
Timer TrainingTimeProfile::partial_z;
Timer TrainingTimeProfile::check_stop;
Timer TrainingTimeProfile::production;
Timer TrainingTimeProfile::cf;
Timer TrainingTimeProfile::update_variables;
Timer TrainingTimeProfile::check_sv;
Timer TrainingTimeProfile::ipm_misc;

Timer PredictingTimeProfile::total;
Timer PredictingTimeProfile::read_model;
Timer PredictingTimeProfile::read_test_doc;
Timer PredictingTimeProfile::predict;

Timer::Timer() {
  communication_ = 0.0;
  computation_ = 0.0;
  total_ = 0.0;
  sync_ = 0.0;
}

double Timer::GetCurrentTime() {
  struct timeval start;
  gettimeofday(&start, NULL);
  return  static_cast<double>(start.tv_sec) +
          static_cast<double>(start.tv_usec)/1000000;
}

void Timer::Start() {
  start_ = GetCurrentTime();
  communication_start_ = communication_elapsed_;
  sync_start_ = sync_elapsed_;
}

void Timer::Stop() {
  double sync_add  = sync_elapsed_ - sync_start_;
  double commu_add = communication_elapsed_ - communication_start_;

  total_         += GetCurrentTime() - start_;
  sync_          += sync_add;
  communication_ += commu_add;
  computation_   = total_ - communication_ - sync_;
}

void Timer::CommunicationStart() {
  last_communication_begin_time_ = GetCurrentTime();
}

void Timer::CommunicationStop() {
  communication_elapsed_ += GetCurrentTime() - last_communication_begin_time_;
}

void Timer::SynchronizationStart() {
  last_sync_begin_time_ = GetCurrentTime();
}

void Timer::SynchronizationStop() {
  sync_elapsed_ += GetCurrentTime() - last_sync_begin_time_;
}

void Timer::Add(const Timer &t) {
  total_         += t.total();
  sync_          += t.sync();
  communication_ += t.communication();
  computation_   += t.computation();
}

void Timer::Minus(const Timer &t) {
  total_         -= t.total();
  sync_          -= t.sync();
  communication_ -= t.communication();
  computation_   -= t.computation();
}

string Timer::PrintInfo() const {
  char sz_line[256];
  snprintf(sz_line, sizeof(sz_line),
           "%e [Calc: %e][Commu: %e][Sync: %e]",
           total_, computation_, communication_, sync_);
  return sz_line;
}
}
