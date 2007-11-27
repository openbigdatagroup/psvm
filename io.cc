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


#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "io.h"
#include "util.h"

using namespace std;
namespace psvm {
bool File::Delete(char* name) {
  return remove(name) == 0;
}

bool File::Exists(char* name) {
  return access(name, F_OK) == 0;
}

size_t File::Size() {
  struct stat f_stat;
  stat(name_, &f_stat);
  return f_stat.st_size;
}

bool File::Flush() {
  return fflush(f_) == 0;
}

bool File::Close() {
  return fclose(f_) == 0;
}

size_t File::ReadOrDie(void* buf, size_t size) {
  return fread(buf, 1, size, f_);
}

size_t File::Read(void* buf, size_t size) {
  return fread(buf, 1, size, f_);
}

size_t File::WriteOrDie(const void* buf, size_t size) {
  return fwrite(buf, 1, size, f_);
}
size_t File::Write(const void* buf, size_t size) {
  return fwrite(buf, 1, size, f_);
}

File* File::OpenOrDie(const char* name, const char* flag) {
  FILE* f_des = fopen(name, flag);
  if (f_des == NULL) {
    cerr << "Cannot open " << name;
    exit(1);
  }
  File* f  = new File;
  f->f_ = f_des;
  strcpy(f->name_, name);
  return f;
}

File* File::Open(const char* name, const char* flag) {
  FILE* f_des = fopen(name, flag);
  if (f_des == NULL) return NULL;
  File* f  = new File;
  f->f_ = f_des;
  strcpy(f->name_, name);
  return f;
}

bool File::ReadLine(string* line) {
  char buff[1000000];
  char* result = fgets(buff, 1000000, f_);
  if (result == NULL) {
    return false;
  } else {
    *line = string(buff);
    return true;
  }
}

size_t File::WriteString(const string& line) {
  return WriteOrDie(line.c_str(), line.size());
}

bool File::WriteLine(const string& line) {
  WriteOrDie(line.c_str(), line.size());
  return WriteOrDie("\n", 1);
}
}
