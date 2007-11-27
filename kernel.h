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

#ifndef KERNEL_H__
#define KERNEL_H__

// Deals with kernel functions of kernel methods. Sample usage:
//    Kernel kernel;
//    kernel.SetKernelType(Kernel::GAUSSIAN);
//    kernel.SetRbfGamma(2);
//    double value = kernel.CalcKernel(a, b);
namespace psvm {
class Sample;

// See comment at top of file for a complete description.
class Kernel {
 public:
  // Defines the supported kernel type.
  enum KernelType { LINEAR = 0, POLYNOMIAL, GAUSSIAN, LAPLACIAN };

  Kernel();

  // Keeps track of the kernel type of this kernel.
  int kernel_type() const { return kernel_type_; }
  void set_kernel_type(KernelType kernel_type) {
    kernel_type_ = kernel_type;
  }

  // Parameters for Gaussian and Laplacian kernel. See comment at the private
  // section.
  double rbf_gamma() const { return rbf_gamma_; }
  void set_rbf_gamma(double rbf_gamma) { rbf_gamma_ = rbf_gamma; }

  // Parameters for Linear kernel. See comment at the private section
  int poly_degree() const { return poly_degree_; }
  void set_poly_degree(int poly_degree) { poly_degree_ = poly_degree; }
  void set_coef_lin(double coef_lin) { coef_lin_ = coef_lin; }
  double coef_lin() const { return coef_lin_; }
  void set_coef_const(double coef_const) { coef_const_ = coef_const; }
  double coef_const() const { return coef_const_; }

  // Computes the kernel function value for samples a and b.
  double CalcKernel(const Sample& a, const Sample& b) const;

  // Like CalcKernel, but negates the result if a and b have different class
  // labels.
  double CalcKernelWithLabel(const Sample& a, const Sample& b) const;

 private:
  // Computes -|a - b|. It is the caller's responsibility to make sure that
  // the features of a and b are sorted according to feature-id in the same
  // order. Otherwise, this method will not return a correct result.
  double OneNormSub(const Sample& a, const Sample& b) const;

  // Computes a.b. It is the caller's responsibility to make sure that
  // the features of a and b are sorted according to feature-id in the same
  // order. Otherwise, this method will not return a correct result.
  double InnerProduct(const Sample& a, const Sample& b) const;

  // Keeps track of the kernel type of this kernel.
  KernelType kernel_type_;

  // The gamma parameter for the Gaussian and Laplacian kernel.
  // In Gaussian kernel
  //    k(a, b) = exp{-rbf_gamma_ * ||a - b||^2}
  // In Laplacian kernel
  //    k(a, b) = exp{-rbf_gamma_ * |a - b|}
  double rbf_gamma_;

  // The three parameters of the polynomial kernel, in which:
  //    k(a, b) = (coef_lin * a.b + coef_const_) ^ poly_degree_
  double coef_lin_;
  double coef_const_;
  int  poly_degree_;
};
}
#endif
