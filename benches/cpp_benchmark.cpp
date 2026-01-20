// C++ benchmark for comparing with Rust voronotalt
//
// Compile with OpenMP (recommended):
//   g++ -O3 -fopenmp -DVORONOTALT_OPENMP \
//       -I/path/to/voronota/expansion_lt/src \
//       -o cpp_benchmark benches/cpp_benchmark.cpp
//
// Compile without OpenMP:
//   g++ -O3 -I/path/to/voronota/expansion_lt/src \
//       -o cpp_benchmark benches/cpp_benchmark.cpp
//
// Run:
//   ./cpp_benchmark [data_path]

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "voronotalt/voronotalt.h"
#include "voronotalt/parallelization_configuration.h"

struct Ball {
    double x, y, z, r;
    Ball(double x, double y, double z, double r) : x(x), y(y), z(z), r(r) {}
};

std::vector<Ball> load_xyzr(const std::string& path) {
    std::vector<Ball> balls;
    std::ifstream file(path);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;
        while (iss >> token) {
            tokens.push_back(token);
        }
        if (tokens.size() >= 4) {
            try {
                size_t n = tokens.size();
                double x = std::stod(tokens[n-4]);
                double y = std::stod(tokens[n-3]);
                double z = std::stod(tokens[n-2]);
                double r = std::stod(tokens[n-1]);
                balls.emplace_back(x, y, z, r);
            } catch (...) {}
        }
    }
    return balls;
}

void benchmark(const std::string& name, const std::vector<Ball>& balls, double probe, int iterations) {
    using namespace std::chrono;

    // Warmup
    for (int i = 0; i < 3; i++) {
        voronotalt::RadicalTessellation::Result result;
        voronotalt::RadicalTessellation::construct_full_tessellation(
            voronotalt::get_spheres_from_balls(balls, probe),
            voronotalt::PeriodicBox(),
            result);
    }

    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        voronotalt::RadicalTessellation::Result result;
        voronotalt::RadicalTessellation::construct_full_tessellation(
            voronotalt::get_spheres_from_balls(balls, probe),
            voronotalt::PeriodicBox(),
            result);
    }
    auto end = high_resolution_clock::now();

    auto total_us = duration_cast<microseconds>(end - start).count();
    double avg_us = static_cast<double>(total_us) / iterations;

    std::cout << std::left << std::setw(25) << name
              << std::right << std::setw(10) << std::fixed << std::setprecision(1) << avg_us << " µs"
              << "  (" << balls.size() << " balls, " << iterations << " iters)" << std::endl;
}

void benchmark_periodic(const std::string& name, const std::vector<Ball>& balls, double probe,
                        double x1, double y1, double z1, double x2, double y2, double z2, int iterations) {
    using namespace std::chrono;

    struct Point { double x, y, z; };
    std::vector<Point> corners = {{x1, y1, z1}, {x2, y2, z2}};
    auto pbox = voronotalt::PeriodicBox::create_periodic_box_from_corners(
        voronotalt::get_simple_points_from_points(corners));

    // Warmup
    for (int i = 0; i < 3; i++) {
        voronotalt::RadicalTessellation::Result result;
        voronotalt::RadicalTessellation::construct_full_tessellation(
            voronotalt::get_spheres_from_balls(balls, probe), pbox, result);
    }

    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        voronotalt::RadicalTessellation::Result result;
        voronotalt::RadicalTessellation::construct_full_tessellation(
            voronotalt::get_spheres_from_balls(balls, probe), pbox, result);
    }
    auto end = high_resolution_clock::now();

    auto total_us = duration_cast<microseconds>(end - start).count();
    double avg_us = static_cast<double>(total_us) / iterations;

    std::cout << std::left << std::setw(25) << name
              << std::right << std::setw(10) << std::fixed << std::setprecision(1) << avg_us << " µs"
              << "  (" << balls.size() << " balls, " << iterations << " iters)" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string base_path = "benches/data/";
    if (argc > 1) {
        base_path = argv[1];
        if (base_path.back() != '/') base_path += '/';
    }

    auto balls_cs_1x1 = load_xyzr(base_path + "balls_cs_1x1.xyzr");
    auto balls_2zsk = load_xyzr(base_path + "balls_2zsk.xyzr");
    auto balls_3dlb = load_xyzr(base_path + "balls_3dlb.xyzr");

    std::cout << "C++ voronota-lt API benchmark" << std::endl;
#ifdef _OPENMP
    std::cout << "_OPENMP defined, omp_get_max_threads: " << omp_get_max_threads() << std::endl;
#else
    std::cout << "_OPENMP not defined" << std::endl;
#endif
#ifdef VORONOTALT_OPENMP
    std::cout << "VORONOTALT_OPENMP defined" << std::endl;
#else
    std::cout << "VORONOTALT_OPENMP not defined" << std::endl;
#endif
    std::cout << "voronotalt::openmp_enabled(): " << (voronotalt::openmp_enabled() ? "true" : "false") << std::endl;

    // Set number of threads (this is what the CLI does)
    voronotalt::openmp_set_num_threads_if_possible(10);
    std::cout << "After setting threads: " << voronotalt::openmp_get_max_threads() << std::endl;

    // Test if OpenMP actually runs in parallel
#ifdef _OPENMP
    {
        int max_thread_id = 0;
        #pragma omp parallel
        {
            #pragma omp critical
            {
                int tid = omp_get_thread_num();
                if (tid > max_thread_id) max_thread_id = tid;
            }
        }
        std::cout << "OpenMP test: max thread id seen = " << max_thread_id << " (should be >0 if parallel)" << std::endl;
    }
#endif
    std::cout << std::string(60, '-') << std::endl;

    // Focus on largest dataset
    std::cout << "balls_3dlb: " << balls_3dlb.size() << " balls" << std::endl;

    // Verify correctness
    {
        voronotalt::RadicalTessellation::Result result;
        voronotalt::RadicalTessellation::construct_full_tessellation(
            voronotalt::get_spheres_from_balls(balls_3dlb, 1.4),
            voronotalt::PeriodicBox(),
            result);
        std::cout << "Contacts: " << result.contacts_summaries.size() << std::endl;
        std::cout << "Cells: " << result.cells_summaries.size() << std::endl;
    }

    benchmark("balls_3dlb (10 runs)", balls_3dlb, 1.4, 10);

    return 0;
}
