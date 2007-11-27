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

#ifndef PD_IPM_PARM_H__
#define PD_IPM_PARM_H__

namespace psvm {
// The parameters for Interior Point Method, see comments
// in pd_ipm.cc for references
struct PrimalDualIPMParameter {
  // ICF factorization resulted $p/n$
  double rank_ratio;
  // ICF stop iteration number
  int iteration;
  // $C$ in optimiation
  double hyper_parm;
  // threshold for primal variable to be deemed as zero
  double epsilon_x;
  // threshold for support vector
  double epsilon_sv;
  // feasibility threshold
  double feas_thresh;
  // stop condition for surrogate gap
  double sgap;
  // maximum iterations of IPM
  int max_iter;
  // increasing factor
  double mu_factor;
  // constant to be substracted from diagonal
  double tradeoff;
  double threshold;
  // the weight of POSITIVE sample in $C$ of optimization
  double weight_positive;
  // the weight of NEGATIVE sample in $C$ of optimization
  double weight_negative;
  // verbose output for diagnostic
  int verb;
  // path to store model result
  char model_path[4096];
  // number of seconds between two save operations
  double save_interval;
};
}

#endif
