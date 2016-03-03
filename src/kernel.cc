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
#include <cmath>
#include <vector>
#include <iostream>
#include "kernel.h"
#include "document.h"

namespace psvm {
Kernel::Kernel()
    : kernel_type_(LINEAR),
      rbf_gamma_(1),
      coef_lin_(1),
      coef_const_(1),
      poly_degree_(3) {
}

// Computes the kernel function value for samples a and b according
// to kernel_type_. See comment at the private section of the header
// file for the formulars of different kernels.
double Kernel::CalcKernel(const Sample& a, const Sample& b) const {
  switch (kernel_type_) {
    case LINEAR:
      return InnerProduct(a, b) / sqrt(a.two_norm_sq * b.two_norm_sq);

    case POLYNOMIAL:
      double val, a_normalizer, b_normalizer;
      val = pow(coef_lin_ * InnerProduct(a, b) + coef_const_,
                static_cast<double>(poly_degree_));
      a_normalizer = pow(coef_lin_ * a.two_norm_sq + coef_const_,
                         static_cast<double>(poly_degree_));
      b_normalizer = pow(coef_lin_ * b.two_norm_sq + coef_const_,
                         static_cast<double>(poly_degree_));
      return val / sqrt(a_normalizer * b_normalizer);

    case GAUSSIAN:
      // Note: ||a - b||^2 = (||a||^2 - 2 * a.b + ||b||^2)
      return exp(-rbf_gamma_ *
                 (a.two_norm_sq - 2 * InnerProduct(a, b) + b.two_norm_sq));

    case LAPLACIAN:
      return exp(-rbf_gamma_ * OneNormSub(a, b));

    default:
      cerr << "Unknown kernel type" << endl;
      exit(1);
      return 0.0;  // This should not be executed because of LOG(FATAL)
  }
}

// Like CalcKernel, but negates the result if a and b have different class
// labels.
double Kernel::CalcKernelWithLabel(const Sample& a, const Sample& b) const {
  return CalcKernel(a, b) * (a.label == b.label ? 1.0 : -1.0);
}

double Kernel::OneNormSub(const Sample& a, const Sample& b) const {
  double norm = 0.0;

  // See comment at the header file. a.features and b.features must be
  // sorted according to the feature-id in the same order. We relies on
  // this property to speed up the computation.
  vector<Feature>::const_iterator it1 = a.features.begin();
  vector<Feature>::const_iterator it2 = b.features.begin();
  while (it1 != a.features.end() && it2 != b.features.end()) {
    if (it1->id == it2->id) {
      norm += fabs(it1->weight - it2->weight);
      ++it1;
      ++it2;
    } else if (it1->id < it2->id) {
      norm += fabs(it1->weight);
      ++it1;
    } else {
      norm += fabs(it2->weight);
      ++it2;
    }
  }
  while (it1 != a.features.end()) {
      norm += fabs(it1->weight);
      ++it1;
  }
  while (it2 != b.features.end()) {
      norm += fabs(it2->weight);
      ++it2;
  }
  return norm;
}

// The implementation of this method is almost identical to that of
// OneNormSub. See comment there.
double Kernel::InnerProduct(const Sample& a, const Sample& b) const {
  double norm = 0.0;

  vector<Feature>::const_iterator it1 = a.features.begin();
  vector<Feature>::const_iterator it2 = b.features.begin();
  while (it1 != a.features.end() && it2 != b.features.end()) {
    if (it1->id == it2->id) {
      norm += it1->weight * it2->weight;
      ++it1;
      ++it2;
    } else if (it1->id < it2->id) {
      ++it1;
    } else {
      ++it2;
    }
  }
  return norm;
}
}
