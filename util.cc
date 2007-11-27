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

#include "util.h"
#include <stdio.h>
#include <ctype.h>

namespace psvm {
void Log(const string& msg, const string& file, int linenum) {
  char buff[100];
  sprintf(buff, "%d", linenum);
  cerr << msg << " in file " << file << " at line " << buff << endl;
}

bool SplitOneIntToken(const char** source, const char* delim,
                      int* value) {

  int i = 0;
  const char* buf = *source;

  // Make sure the first character is a digit and parse it
  if (!isdigit(buf[0])) {
    if (buf[0] == '-' || buf[0] == '+') {
      if (!isdigit(buf[1])) return false;
      i = 1;
    } else {
      return false;
    }
  }
  if (sscanf(buf, "%d", value) != 1) return false;

  // Skip the integer
  for (; buf[i] != 0 && isdigit(buf[i]); ++i);

  // Make sure the character after the integer is from delim and skip it
  if (buf[i] != 0) {
    if (strchr(delim, buf[i]) == NULL) return false;
  }
  *source = &(buf[i + 1]);

  return true;
}

bool SplitOneDoubleToken(const char** source, const char* delim,
                         double* value) {
  int i = 0;
  const char* buf = *source;

  // Make sure the first character is a digit and parse it
  if (!isdigit(buf[0])) {
    if (buf[0] == '-' || buf[0] == '+') {
      if (!isdigit(buf[1])) return false;
      i = 1;
    } else {
      return false;
    }
  }
  if (sscanf(buf, "%lf", value) != 1) return false;

  // Skip the number
  while (buf[i] != 0 && strchr(delim, buf[i]) == NULL) ++i;

  // Make sure the character after the number is from delim and skip it
  if (buf[i] != 0) {
    if (strchr(delim, buf[i]) == NULL) return false;
  }
  *source = &(buf[i + 1]);

  return true;
}

// Split a line into a vector of <key, value> pairs. The line has
// the following format:
//
// <kvpsep>*<key1><kvsep>+<value1><kvpsep>+<key2><kvsep>+<value2>...<kvpsep>*
void SplitStringIntoKeyValuePairs(const string& line,
                                  const string& key_value_delimiters,
                                  const string& key_value_pair_delimiters,
                                  vector<pair<string, string> >* kv_pairs) {
  int i, j;
  const char* buf = line.c_str();
  kv_pairs->clear();

  // Skip the preceding kvpsep
  for (i = 0; buf[i] != 0; ++i) {
    if (strchr(key_value_pair_delimiters.c_str(), buf[i]) == NULL) break;
  }

  while (true) {
    // Locate the key
    j = i;
    while (buf[j] != 0 && strchr(key_value_delimiters.c_str(), buf[j]) == NULL) ++j;
    if (buf[j] == 0) break;
    string key(&buf[i], j - i);

    // Skip the kvsep
    i = j;
    while (buf[j] != 0 && strchr(key_value_delimiters.c_str(), buf[j]) != NULL) ++j;
    if (buf[j] == 0 || i == j) break;

    // Locate the value
    i = j;
    while (buf[j] != 0 && strchr(key_value_pair_delimiters.c_str(), buf[j]) == NULL) ++j;
    if (i == j) break;
    string value(&buf[i], j - i);

    kv_pairs->push_back(make_pair(key, value));

    // Skip the kvpsep
    for (i = j; buf[i] != 0; ++i) {
      if (strchr(key_value_pair_delimiters.c_str(), buf[i]) == NULL) break;
    }
  }
}

string StringPrintfV(const char* format, va_list ap) {
  char buffer[1024];
  va_list backup_ap;
  va_copy(backup_ap, ap);
  vsnprintf(buffer, sizeof(buffer), format, backup_ap);
  va_end(backup_ap);
  return string(buffer);
}

string StringPrintf(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  string result = StringPrintfV(format, ap);
  va_end(ap);
  return result;
}
}
