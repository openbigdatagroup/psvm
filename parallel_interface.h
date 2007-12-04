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

#ifndef PARALLEL_INTERFACE_H__
#define PARALLEL_INTERFACE_H__

#undef SEEK_CUR
#undef SEEK_SET
#undef SEEK_END
#include "mpi.h"

// Forward declarations of the MPI stuff
namespace psvm {
class ParallelInterface {
 public:
  // Call Init before all other calls to initialize mpi interface.
  void Init(int* argc, char*** argv);

  // A wrapper to call MPI_Finalize. It must be called at the end of program.
  void Finalize();

  // A wrapper to call MPI_Get_proc_name. Refer to MPI documents for usage.
  void GetProcName(char *, int *);

  // A wrapper to call MPI_Barrier. Refer to MPI documents for usage.
  void Reduce(void* , void*, int, MPI_Datatype, MPI_Op, int, MPI_Comm);

  // A wrapper to call MPI_Allreduce. Refer to MPI documents for usage.
  void AllReduce(void* , void*, int, MPI_Datatype, MPI_Op, MPI_Comm);

  // A wrapper to call MPI_Barrier. Refer to MPI documents for usage.
  void Barrier(MPI_Comm);

  // A wrapper to call MPI_Bcast. Refer to MPI documents for usage.
  void Bcast(void*, int, MPI_Datatype, int, MPI_Comm);

  // Get local processor id.
  int GetProcId() const;

  // Get the number of processors;
  int GetNumProcs() const;

  // Computer if there are totally num_total elements which need to be stored
  // distributedly, how many should be stored on local machine.
  // Algorithm: totally n elements, N machines. Machine i should store
  //    n/N + (n%N>i?1:0)
  // elements.
  // Example: 5 elements: 1 2 3 4 5; 3 machines.
  //      machine 0 stored : 1 4
  //      machine 1 stored : 2 5
  //      machine 2 stored : 3
  int ComputeNumLocal(int num_total) const;

  // If the global_idx element is stored on machine proc_id, what is the local
  // index on the machine proc_id.
  // Example: 5 elements: 1 2 3 4 5; 3 machines;
  //          ComputeGlobalToLocal(2, 2) = 0, because machine 2 stored the
  //            third element(3) and it is the first element on machine 2
  //          ComputeGlobalToLocal(3, 0) = 1, because machine 0 stored the
  //            fourth element(4) and it is the second element on machine 0.
  //          ComputeGlobalToLocal(3, 1) = ERROR, because the fourth element is
  //            not stored on machine 1.
  int ComputeGlobalToLocal(int global_idx, int proc_id) const;

  // Compute what is the global_index of the element stored on machine proc_id
  //         with index local_idx
  // Example: 5 elements: 1 2 3 4 5; 3 machines;
  //          ComputeLocalToGlobal(1, 0) = 3, because the 2nd element on
  //            machine 0 is 4 which is the fourth element in original order.
  //          ComputeLocalToGlobal(1, 1) = 4, because the 2nd element on
  //            machine 1 is 5 which is the fifth element in original order.
  //          ComputeLocalToGlobal(1, 2) = ERROR, because there are only
  //            1 elment on machine 2.
  int ComputeLocalToGlobal(int local_idx, int proc_id) const;

  // To get global interface pointer for singleton design patterns
  static ParallelInterface* GetParallelInterface();
 protected:
  ParallelInterface();
 private:
  // global interface for singleton design patterns.
  static ParallelInterface* interface_;
};
}

#endif
