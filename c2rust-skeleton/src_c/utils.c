#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

char* str_trim(char* str) {
    if (str == NULL) return NULL;

    // Trim leading whitespace
    char* start = str;
    while (isspace((unsigned char)*start)) start++;

    if (*start == 0) {
        *str = 0;
        return str;
    }

    // Trim trailing whitespace
    char* end = start + strlen(start) - 1;
    while (end > start && isspace((unsigned char)*end)) end--;
    *(end + 1) = 0;

    if (start != str) {
        memmove(str, start, (end - start) + 2);
    }
    return str;
}

int str_count_char(const char* str, char c) {
    if (str == NULL) return -1;
    int count = 0;
    while (*str) {
        if (*str == c) count++;
        str++;
    }
    return count;
}

char* str_duplicate(const char* str) {
    if (str == NULL) return NULL;
    size_t len = strlen(str) + 1;
    char* dup = (char*)malloc(len);
    if (dup == NULL) return NULL;
    memcpy(dup, str, len);
    return dup;
}

int int_array_sum(const int* arr, size_t len) {
    if (arr == NULL || len == 0) return 0;
    int sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += arr[i];
    }
    return sum;
}

int int_array_max(const int* arr, size_t len, int* out_max) {
    if (arr == NULL || len == 0 || out_max == NULL) return -1;
    *out_max = arr[0];
    for (size_t i = 1; i < len; i++) {
        if (arr[i] > *out_max) {
            *out_max = arr[i];
        }
    }
    return 0;
}
