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

#include <string>
#include "parallel_interface.h"
#include "timer.h"

namespace psvm {

ParallelInterface* ParallelInterface::interface_;

ParallelInterface::ParallelInterface() {
}

void ParallelInterface::Init(int* argc, char*** argv) {
  MPI_Init(argc, argv);
}
int ParallelInterface::GetProcId() const {
  int proc_id;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
  return proc_id;
}

int ParallelInterface::GetNumProcs() const {
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
}

int ParallelInterface::ComputeNumLocal(int num_global) const {
  int num_procs = GetNumProcs();
  int proc_id = GetProcId();
  return num_global / num_procs + (num_global % num_procs > proc_id ? 1 : 0);
}

int ParallelInterface::ComputeGlobalToLocal(int global_idx, int proc_id) const {
  int num_procs = GetNumProcs();
  if ((global_idx - proc_id) % num_procs != 0)
    return -1;
  else
    return (global_idx - proc_id) / num_procs;
}

int ParallelInterface::ComputeLocalToGlobal(int local_idx, int proc_id) const {
  int num_procs = GetNumProcs();
  return local_idx * num_procs + proc_id;
}

ParallelInterface* ParallelInterface::GetParallelInterface() {
  if (interface_ == NULL)
    interface_ = new ParallelInterface();
  return interface_;
}

void ParallelInterface::AllReduce(void* buf, void* buf2, int n,
                                  MPI_Datatype datatype, MPI_Op op,
                                  MPI_Comm comm) {
  // Use Barrier() to separate synchronization time from transfermation time
  Barrier(MPI_COMM_WORLD);

  Timer::CommunicationStart();
  MPI_Allreduce(buf, buf2, n, datatype, op, comm);
  Timer::CommunicationStop();
}

void ParallelInterface::Barrier(MPI_Comm comm) {
  Timer::SynchronizationStart();
  MPI_Barrier(comm);
  Timer::SynchronizationStop();
}

void ParallelInterface::Bcast(void* buf, int n, MPI_Datatype datatype, int root,
                              MPI_Comm comm) {
  // Use Barrier() to separate synchronization timefrom transfermation time
  Barrier(MPI_COMM_WORLD);

  Timer::CommunicationStart();
  MPI_Bcast(buf, n, datatype, root, comm);
  Timer::CommunicationStop();
}

void ParallelInterface::Finalize() {
  MPI_Finalize();
}

void ParallelInterface::GetProcName(char *name, int *len) {
  MPI_Get_processor_name(name, len);
}

void ParallelInterface::Reduce(void* buf, void* buf2, int n,
                               MPI_Datatype datatype, MPI_Op op,
                               int root, MPI_Comm comm) {
  // Use Barrier() to separate synchronization timefrom transfermation time
  Barrier(MPI_COMM_WORLD);

  Timer::CommunicationStart();
  MPI_Reduce(buf, buf2, n, datatype, op, root, comm);
  Timer::CommunicationStop();
}
}
