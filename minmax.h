//
// Created by mmayer on 11.04.18.
//

#ifndef SIMD_FUN_MINMAX_H
#define SIMD_FUN_MINMAX_H

#include "Benchmark.h"

minmax_t minmax_naive(const std::vector<float> &values);
minmax_t minmax_avx256(const std::vector<float> &values);
minmax_t minmax_avx256_16(const std::vector<float> &values);

#endif //SIMD_FUN_MINMAX_H
