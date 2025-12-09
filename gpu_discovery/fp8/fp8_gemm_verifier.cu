// fp8_gemm_verifier.cu
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/float8.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_compare.h>

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <random>

#include "gpu_timer.h"
#include "fp8_gemm_verifier.h"


Fp8GemmVerifier::Fp8GemmVerifier(int m, int n, int k, int warmup_iterations, int iterations)
    : M(m), N(n), K(k), WARMUP_ITERATIONS(warmup_iterations), ITERATIONS(iterations) {
    using ElementA = cutlass::float_e4m3_t;
    fp8_min = cutlass::platform::numeric_limits<ElementA>::lowest();
    fp8_max = cutlass::platform::float8_base_numeric_limits<ElementA>::max();
}

std::tuple<bool,int, double> Fp8GemmVerifier::verify(int device_id, bool verbose) {
    int error_code = 0;
    double gFLOPS = 0.0f;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    cudaSetDevice(device_id);

    if (verbose) {
        std::cout << "\n=== Verifying FP8 GEMM on device " << device_id << " (" << prop.name << ") ===\n";
    }

    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementOutput = float;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    constexpr int kStages = 2;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementOutput, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm89,
        cutlass::gemm::GemmShape<16, 8, 64>,
        cutlass::gemm::GemmShape<16, 8, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        kStages>;

    Gemm gemm_op;

    cutlass::HostTensor<ElementA, LayoutA> tensor_a({M, K});
    cutlass::HostTensor<ElementB, LayoutB> tensor_b({K, N});
    cutlass::HostTensor<ElementOutput, LayoutC> tensor_d({M, N});
    cutlass::HostTensor<ElementOutput, LayoutC> tensor_d_ref({M, N});

    // Random generator with true entropy seed
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(fp8_min, fp8_max);

    auto A_host = tensor_a.host_view();
    auto B_host = tensor_b.host_view();

    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            A_host.at({m, k}) = ElementA(dist(gen));

    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            B_host.at({k, n}) = ElementB(dist(gen));

    tensor_a.sync_device();
    tensor_b.sync_device();

    typename Gemm::Arguments args(
        {M, N, K},
        {tensor_a.device_ref(), K},
        {tensor_b.device_ref(), K},
        {tensor_d.device_ref(), N},
        {tensor_d.device_ref(), N},
        {1.0f, 0.0f});

    // Determines whether the GEMM can execute the given problem
    if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) {
        error_code = -10;
        if (verbose) {
            std::cerr << "CUTLASS GEMM can_implement is failed \n";
        }
        return std::make_tuple(false, error_code, gFLOPS);
    }

    // Run GEMM kernel
    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        error_code = -20;
        if (verbose) {
            std::cerr << "CUTLASS GEMM launch failed (maybe FP8 not supported)\n";
        }
        return std::make_tuple(false, error_code, gFLOPS);
    }

    cudaDeviceSynchronize();
    tensor_d.sync_host();

    // Host reference
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<float>(A_host.at({m, k})) * static_cast<float>(B_host.at({k, n}));
            }
            tensor_d_ref.at({m, n}) = acc;
        }
    }

    // Compare results
    float tol = 1.0f;
    bool pass = true;
    for (int m = 0; m < M && pass; ++m) {
        for (int n = 0; n < N; ++n) {
            float diff = fabs(tensor_d.at({m, n}) - tensor_d_ref.at({m, n}));

            if (diff > tol) {
                error_code = -30;
                pass = false;
                if (verbose) {
                    std::cout << "Mismatch at (" << m << "," << n << ") device=" << tensor_d.at({m, n}) << " host=" << tensor_d_ref.at({m, n}) << " diff=" << diff << "\n";
                }
            }
        }
    }

    if (verbose) {
        std::cout << (pass ? "FP8 GEMM verified\n" : "FP8 GEMM mismatch\n");
    }

    if (pass) {
        // Warm up
        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            gemm_op(args);
        }

        // Profiling
        GpuTimer timer;
        timer.start();

        for (int iter = 0; iter < ITERATIONS; ++iter) {
            gemm_op(args);
        }
        timer.stop();

        // Compute average runtime and GFLOPs.
        float elapsed_ms = timer.elapsed_millis();
        double avg_runtime_ms = double(elapsed_ms) / double(ITERATIONS);
        gFLOPS = gflops(avg_runtime_ms / 1000.0);

        if (verbose) {
            std::cout << "  Problem Size: " << M << 'x' << N << 'x' << K << std::endl;
            std::cout << "  Avg runtime: " << avg_runtime_ms << " ms" << std::endl;
            std::cout << "  GFLOPS: " << gFLOPS << std::endl;
        }
    }

    return std::make_tuple(pass, error_code, gFLOPS);
}

// Compute performance in GFLOP/s
double Fp8GemmVerifier::gflops(double runtime_s) const {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * M * N * K;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
}
