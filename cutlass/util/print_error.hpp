// Minimal print_error.hpp stub
#pragma once

#include <cstdio>

inline void print_error(const char* msg) {
    std::fprintf(stderr, "ERROR: %s\n", msg);
}
