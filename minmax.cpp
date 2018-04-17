//
// Created by mmayer on 11.04.18.
//

#include <immintrin.h> // AVX
#include <limits>
#include "Benchmark.h"

minmax_t minmax_naive(const std::vector<float> &values) {
    static_assert(std::numeric_limits<float>::is_iec559, "Require IEC559 floating point");

    const auto length = values.size();
    auto min = std::numeric_limits<float>::infinity();
    auto max = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < length; ++i) {
        const auto value = values[i];
        if (value > max) max = value;
        if (value < min) min = value;
    }
    return {min, max};
}

union m256 {
    __m256 v;
    float a[8];
};

minmax_t minmax_avx256(const std::vector<float> &values) {
    static_assert(std::numeric_limits<float>::is_iec559, "Require IEC559 floating point");

    const auto length = values.size();
    auto vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    auto vmin = _mm256_set1_ps(std::numeric_limits<float>::infinity());

    const auto raw = &values[0];
    for (int i = 0; i < length; i += 8) {
        const auto value = _mm256_loadu_ps(&raw[i]);
        vmax = _mm256_max_ps(value, vmax);
        vmin = _mm256_min_ps(value, vmin);
    }

    // https://stackoverflow.com/a/33855061/195651
    vmax = _mm256_max_ps(vmax, _mm256_shuffle_ps(vmax, vmax, _MM_SHUFFLE(2U, 3U, 0U, 1U)));
    vmax = _mm256_max_ps(vmax, _mm256_shuffle_ps(vmax, vmax, _MM_SHUFFLE(1U, 0U, 3U, 2U)));
    vmax = _mm256_max_ps(vmax, _mm256_permute2f128_ps(vmax, vmax, 1));

    vmin = _mm256_min_ps(vmin, _mm256_shuffle_ps(vmin, vmin, _MM_SHUFFLE(2U, 3U, 0U, 1U)));
    vmin = _mm256_min_ps(vmin, _mm256_shuffle_ps(vmin, vmin, _MM_SHUFFLE(1U, 0U, 3U, 2U)));
    vmin = _mm256_min_ps(vmin, _mm256_permute2f128_ps(vmin, vmin, 1));

    const m256 min { vmin };
    const m256 max { vmax };
    return {min.a[0], max.a[0]};
}

minmax_t minmax_avx256_16(const std::vector<float> &values) {
    static_assert(std::numeric_limits<float>::is_iec559, "Require IEC559 floating point");

    const auto length = values.size();
    auto vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    auto vmin = _mm256_set1_ps(std::numeric_limits<float>::infinity());

    const auto raw = &values[0];
    for (int i = 0; i < length; i += 16) {
        const auto value_1 = _mm256_loadu_ps(&raw[i]);
        const auto value_2 = _mm256_loadu_ps(&raw[i + 8]);

        const auto vmax_1 = _mm256_max_ps(value_1, vmax);
        const auto vmin_1 = _mm256_min_ps(value_1, vmin);

        const auto vmax_2 = _mm256_max_ps(value_2, vmax);
        const auto vmin_2 = _mm256_min_ps(value_2, vmin);

        vmax = _mm256_max_ps(vmax_1, vmax_2);
        vmin = _mm256_min_ps(vmin_1, vmin_2);
    }

    // https://stackoverflow.com/a/33855061/195651
    vmax = _mm256_max_ps(vmax, _mm256_shuffle_ps(vmax, vmax, _MM_SHUFFLE(2U, 3U, 0U, 1U)));
    vmax = _mm256_max_ps(vmax, _mm256_shuffle_ps(vmax, vmax, _MM_SHUFFLE(1U, 0U, 3U, 2U)));
    vmax = _mm256_max_ps(vmax, _mm256_permute2f128_ps(vmax, vmax, 1));

    vmin = _mm256_min_ps(vmin, _mm256_shuffle_ps(vmin, vmin, _MM_SHUFFLE(2U, 3U, 0U, 1U)));
    vmin = _mm256_min_ps(vmin, _mm256_shuffle_ps(vmin, vmin, _MM_SHUFFLE(1U, 0U, 3U, 2U)));
    vmin = _mm256_min_ps(vmin, _mm256_permute2f128_ps(vmin, vmin, 1));

    const m256 min { vmin };
    const m256 max { vmax };
    return {min.a[0], max.a[0]};
}