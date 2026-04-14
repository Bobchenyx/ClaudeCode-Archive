#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

// String utilities
char* str_trim(char* str);
int str_count_char(const char* str, char c);
char* str_duplicate(const char* str);

// Array utilities
int int_array_sum(const int* arr, size_t len);
int int_array_max(const int* arr, size_t len, int* out_max);

#endif // UTILS_H
