//
// Created by mmayer on 11.04.18.
//

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include "Benchmark.h"

void Benchmark::operator()(const std::vector<float> &values) {
    const int iterations = 100;

    std::array<float, iterations> durations{};
    std::vector<minmax_t> results{};
    results.reserve(iterations);

    auto total_duration = 0.0F;

    // warm-up
    run(values);

    // TODO: Run a second or 100 iterations, whatever takes longer, but at least ten iterations?
    for (auto i =0; i<iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        const auto result = run(values);
        const auto runtime = std::chrono::steady_clock::now() - start;
        const auto runtime_ms = std::chrono::duration<float, std::milli>(runtime).count();

        total_duration += runtime_ms;
        durations[i] = runtime_ms;
        results.emplace_back(result);
    }

    // Obtain sum of squared errors.
    const auto average_duration = total_duration / iterations;
    auto sse = 0.0F;
    for (auto duration : durations) {
        const auto error = (duration - average_duration);
        sse += error * error;
    }

    // Obtain standard deviation.
    const auto mse = sse / iterations;
    const auto standard_deviation = sqrt(mse);

    std::cout << "- Values in range " << results[0].min << " .. " << results[0].max << std::endl;
    std::cout << "- Duration: " << average_duration << " ms +/- " << standard_deviation << " ms out of " << iterations << " trials." << std::endl;
}