//
// Created by mmayer on 11.04.18.
//

#ifndef SIMD_FUN_BENCHMARK_H
#define SIMD_FUN_BENCHMARK_H

#include <vector>

struct minmax_t {
    constexpr minmax_t(const float min, const float max) noexcept
            : min{min}, max{max}
    {}

    constexpr minmax_t(const minmax_t& other) noexcept
            : min{other.min}, max{other.max}
    {}

    constexpr minmax_t& operator=(const minmax_t& other) noexcept {
        min = other.min;
        max = other.max;
    }

    float min;
    float max;
};

class Benchmark {
public:
    explicit Benchmark() noexcept = default;
    virtual ~Benchmark() = default;

    void operator()(const std::vector<float> &values);

protected:
    virtual minmax_t run(std::vector<float> values) const noexcept = 0;
};


#endif //SIMD_FUN_BENCHMARK_H
