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

#ifndef PD_IPM_H__
#define PD_IPM_H__

namespace psvm {
class PrimalDualIPMParameter;
class ParallelMatrix;
class Document;
class Model;
class LLMatrix;
// Newton method for primal-dual interior point method for SVM optimization,
class PrimalDualIPM {
 public:
  // Using Newton method to solve the optimization problem
  //   parameter: the options of interior point method
  //   h: ICF factorized matrix
  //   doc: data points
  //   model: store the optimization result
  int Solve(const PrimalDualIPMParameter& parameter,
            const ParallelMatrix& h,
            const Document& doc,
            Model* model,
            bool failsafe);

 private:
  // Compute $HH^T\alpha$, which is part of $z$, $\alpha$ is primal variable
  int ComputePartialZ(const ParallelMatrix& icf,
                      const double *x,
                      double to,
                      int local_num_rows,
                      double *z);

  // Compute surrogate gap
  double ComputeSurrogateGap(double c_pos,
                             double c_neg,
                             const double *label,
                             int local_num_rows,
                             const double *x,
                             const double *la,
                             const double *xi);

  // Compute direction of primal vairalbe $x$
  int ComputeDeltaX(const ParallelMatrix& icf,
                    const double *d,
                    const double *label,
                    const double dnu,
                    const LLMatrix& lra,
                    const double *z,
                    const int local_num_rows,
                    double *dx);

  // Compute direction of primal varialbe $\nu$
  int ComputeDeltaNu(const ParallelMatrix& icf,
                     const double *d,
                     const double *label,
                     const double *z,
                     const double *x,
                     const LLMatrix& lra,
                     const int local_num_rows,
                     double *dnu);

  // Solve a special form of linear equation using
  // Sherman-Morrison-Woodbury formula
  int LinearSolveViaICFCol(const ParallelMatrix& icf,
                           const double *d,
                           const double *b,
                           const LLMatrix& lra,
                           const int local_num_rows,
                           double *x);

  // Loads the values of alpha, xi, lambda and nu to resume from an interrupted
  // solving process.
  void LoadVariables(const PrimalDualIPMParameter& parameter,
                     int num_local_doc, int num_total_doc, int *step,
                     double *nu, double *x, double *la, double *xi);

  // Saves the values of alpha, xi, lambda and nu.
  void SaveVariables(const PrimalDualIPMParameter& parameter,
                     int num_local_doc, int num_total_doc, int step,
                     double nu, double *x, double *la, double *xi);
};
}  // namespace psvm

#endif
