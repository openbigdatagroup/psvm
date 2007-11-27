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

#ifndef IO_H__
#define IO_H__

#include <cstdlib>
#include <cstdio>
#include <string>

using std::string;

namespace psvm {
class File {
 public:
  static File* Open(const char* name, const char* flag);
  static File* OpenOrDie(const char* name, const char*  flag);
  size_t ReadOrDie(void* buff, size_t size);
  size_t Read(void* buff, size_t size);
  // Each line must be no more than 1000000 bytes
  bool ReadLine(string* line);
  size_t WriteOrDie(const void* buff, size_t size);
  size_t Write(const void* buff, size_t size);
  size_t WriteString(const string& line);
  bool WriteLine(const string& line);
  bool Close();
  bool Flush();
  size_t Size();
  static bool Delete(char* name);
  static bool Exists(char* name);
 protected:
  FILE* f_;
  char name_[1024];
};
}
#endif
