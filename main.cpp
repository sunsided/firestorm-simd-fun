#include <random>
#include <limits>
#include <utility>
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <array>
#include "Benchmark.h"
#include "minmax.h"


class NaiveMinMax final : public Benchmark {
protected:
    minmax_t run(std::vector<float> values) const noexcept final {
        return minmax_naive(values);
    }
};

class Avx256MinMax final : public Benchmark {
protected:
    minmax_t run(std::vector<float> values) const noexcept final {
        return minmax_avx256(values);
    }
};

class Avx25616MinMax final : public Benchmark {
protected:
    minmax_t run(std::vector<float> values) const noexcept final {
        return minmax_avx256_16(values);
    }
};

int main() {
    std::random_device rd;
    const auto seed = rd();
    std::mt19937 mt(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

    std::vector<float> values{};
    values.reserve(1024000);
    for (int i=0; i < values.capacity(); ++i) {
        values.emplace_back(dist(mt));
    }

    std::vector<float> sorted{values};
    std::sort(sorted.begin(), sorted.end());

    std::cout << "Naive:" << std::endl;
    NaiveMinMax naive{};
    naive(values);

    std::cout << "Naive (sorted):" << std::endl;
    naive(sorted);

    std::cout << "AVX 256:" << std::endl;
    Avx256MinMax avx256;
    avx256(values);

    std::cout << "AVX 256 (unrolled):" << std::endl;
    Avx25616MinMax avx256_16;
    avx256_16(values);

    return 0;
}