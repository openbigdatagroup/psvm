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

// This file define some IO interfaces to compatible with Google
// IO specifications.
namespace psvm {
class File {
 public:
  // Opens file "name" with flags specified by "flag"
  static File* Open(const char* name, const char* flag);

  // Opens file "name" with flags specified by "flag"
  // If open failed, program will exit.
  static File* OpenOrDie(const char* name, const char*  flag);

  // Reads "size" bytes to buff from file, buff should be pre-allocated.
  size_t Read(void* buff, size_t size);

  // Reads "size" bytes to buff from file, buff should be pre-allocated.
  // If read failed, program will exit.
  void ReadOrDie(void* buff, size_t size);

  // Read a line from file.
  // Each line must be no more than 1000000 bytes
  bool ReadLine(string* line);

  // Writes "size" bytes of buff to file, buff should be pre-allocated.
  size_t Write(const void* buff, size_t size);

  // Writes "size" bytes of buff to file, buff should be pre-allocated.
  // If write failed, program will exit.
  void WriteOrDie(const void* buff, size_t size);

  // Write a string to file.
  size_t WriteString(const string& line);

  // Write a string to file and append a "\n".
  bool WriteLine(const string& line);

  // Close the file.
  bool Close();

  // Flush buffer.
  bool Flush();

  // Return file size.
  size_t Size();

  // Delete a file.
  static bool Delete(char* name);

  // Test if a file exists.
  static bool Exists(char* name);
 protected:
  FILE* f_;
  char name_[1024];
};
}
#endif
