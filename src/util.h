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

#ifndef UTIL_H__
#define UTIL_H__

#include <vector>
#include <string>
#include <iostream>
#include <stdarg.h>

using namespace std;

#define CHECK(condition)  \
                  if (!(condition)) { Log("Check failed: " #condition " ",  __FILE__, __LINE__) ; exit(1);}

#define CHECK_LE(a, b)  \
                  if ((a) > (b)) { Log("Check failed: " #a " <= " #b " ",  __FILE__, __LINE__) ; exit(1);}

#define CHECK_EQ(a, b)  \
                  if ((a) == (b)) { Log("Check failed: " #a " == " #b " ",  __FILE__, __LINE__) ; exit(1);}


namespace psvm {
void Log(const string& msg, const string& file, int linenum);

bool SplitOneIntToken(const char** source, const char* delim,
                      int* value);

bool SplitOneDoubleToken(const char** source, const char* delim,
                         double* value);

void SplitStringIntoKeyValuePairs(const string& line,
                                  const string& key_value_delimiters,
                                  const string& key_value_pair_delimiters,
                                  vector<pair<string, string> >* kv_pairs);

string StringPrintfV(const char* format, va_list ap);

string StringPrintf(const char* format, ...);
}
#endif
