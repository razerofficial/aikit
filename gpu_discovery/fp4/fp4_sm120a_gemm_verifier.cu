#include "fp4_sm120a_gemm_verifier.h"

#include <torch/torch.h>
#include <torch/script.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>


// forward declare the kernel entry
void cutlass_scaled_fp4_mm_sm120a(torch::Tensor& D, torch::Tensor const& A,
                                  torch::Tensor const& B, torch::Tensor const& A_sf,
                                  torch::Tensor const& B_sf, torch::Tensor const& alpha, double& gflops, int warmup_iteration = 0, int iteration = 0, bool verbose = false);

Fp4Sm120aGemmVerifier::Fp4Sm120aGemmVerifier(int m, int n, int k, int warmup_iterations, int iterations)
    : M(m), N(n), K(k), WARMUP_ITERATIONS(warmup_iterations), ITERATIONS(iterations) {
}

Fp4Sm120aGemmVerifier::~Fp4Sm120aGemmVerifier() {
}

std::tuple<bool,int, double> Fp4Sm120aGemmVerifier::verify(int device_id, bool verbose) {
    int error_code = 0;
    double gFLOPS = 0.0f;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    std::vector<float> A_fp32(M*K), B_fp32(K*N);
    for (auto &x : A_fp32) x = dist(rng);
    for (auto &x : B_fp32) x = dist(rng);

    auto opts_u8_cpu = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);

    // ---- A: row-major packed (M, K/2)
    torch::Tensor A_bytes_cpu = torch::empty({M, K/2}, opts_u8_cpu);
    {
        auto A_ptr = A_bytes_cpu.data_ptr<uint8_t>();
        for (int m = 0; m < M; ++m) {
            for (int kk = 0; kk < K; kk += 2) {
                uint8_t lo = encode_e2m1_nibble(A_fp32[m*K + kk]);
                uint8_t hi = encode_e2m1_nibble(A_fp32[m*K + kk + 1]);
                A_ptr[m*(K/2) + (kk/2)] = pack2(lo, hi);
            }
        }
    }

    if (verbose) {
        std::cout << "Fill A (CPU) done; A_bytes_cpu.is_contiguous=" << A_bytes_cpu.is_contiguous() << std::endl;
    }

    // ---- B: direct shape (N, K/2), row-major contiguous
    torch::Tensor B_bytes_cpu = torch::empty({N, K/2}, opts_u8_cpu);
    {
        auto B_ptr = B_bytes_cpu.data_ptr<uint8_t>();
        for (int n = 0; n < N; ++n) {
            for (int kk = 0; kk < K; kk += 2) {
                uint8_t lo = encode_e2m1_nibble(B_fp32[(kk + 0)*N + n]);
                uint8_t hi = encode_e2m1_nibble(B_fp32[(kk + 1)*N + n]);
                int kp = kk/2;
                B_ptr[n*(K/2) + kp] = pack2(lo, hi); // rows=n, cols=kp
            }
        }
    }

    if (verbose) {
        std::cout << "Fill B (CPU) done; B_bytes_cpu.is_contiguous=" << B_bytes_cpu.is_contiguous() << std::endl;
    }

    // ---- check whether CUDA is available
    if (!torch::cuda::is_available()) {
        if (verbose) {
            std::cerr << "CUDA not available\n";
        }

        error_code = -10;
        return std::make_tuple(false, error_code, gFLOPS);
    }
 
    try {
        cudaSetDevice(device_id);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        if (verbose) {
            std::cout << "Running FP4 test on GPU " << device_id << " : " << prop.name << " (SM " << prop.major << "," << prop.minor << ")" << std::endl << std::endl;
        }

        torch::Device dev(torch::kCUDA, device_id);
        torch::Tensor A_bytes = A_bytes_cpu.to(dev);
        torch::Tensor B_bytes = B_bytes_cpu.to(dev);

        // ---- Build dequantized A_q, B_q for reference GEMM
        std::vector<float> A_q(M*K), B_q(K*N);
        {
            auto Ap = A_bytes_cpu.data_ptr<uint8_t>();
            for (int m = 0; m < M; ++m) {
                for (int kk = 0; kk < K; kk += 2) {
                    uint8_t byte = Ap[m*(K/2) + (kk/2)];
                    uint8_t lo, hi; 
                    unpack2(byte, lo, hi);

                    A_q[m*K + kk]   = decode_e2m1_nibble(lo);
                    A_q[m*K + kk+1] = decode_e2m1_nibble(hi);
                }
            }

            auto Bp = B_bytes_cpu.data_ptr<uint8_t>();
            for (int n = 0; n < N; ++n) {
                for (int kk = 0; kk < K; kk += 2) {
                    int kp = kk/2;
                    uint8_t byte = Bp[n*(K/2) + kp]; // matches direct layout
                    uint8_t lo, hi;
                    unpack2(byte, lo, hi);

                    B_q[(kk+0)*N + n] = decode_e2m1_nibble(lo);
                    B_q[(kk+1)*N + n] = decode_e2m1_nibble(hi);
                }
            }
        }

        // ---- Scale tensors
        auto round_up = [](int x,int y){ return (x+y-1)/y*y; };
        int rounded_m = round_up(M, 128);
        int rounded_n = round_up(N, 128);
        int rounded_k = round_up(K/16,4);

        auto opts_sf = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(dev);
        torch::Tensor A_sf = torch::ones({rounded_m, rounded_k}, opts_sf).contiguous();
        torch::Tensor B_sf = torch::ones({rounded_n, rounded_k}, opts_sf).contiguous();

        // ---- Output + alpha
        auto out_dtype = torch::kBFloat16;
        torch::Tensor D = torch::empty({M,N}, torch::TensorOptions().dtype(out_dtype).device(dev)).contiguous();
        torch::Tensor alpha = torch::tensor(1.0f, torch::TensorOptions().dtype(torch::kFloat32).device(dev));

        // ---- Debug types
        if (verbose) {
            std::cout << "A_bytes device CUDA ? " << A_bytes.is_cuda() << " contiguous? " << A_bytes.is_contiguous() << " dtype==kUInt8? " << (A_bytes.dtype()==torch::kUInt8) << std::endl;
            std::cout << "B_bytes device CUDA ? " << B_bytes.is_cuda() << " contiguous? " << B_bytes.is_contiguous() << " dtype==kUInt8? " << (B_bytes.dtype()==torch::kUInt8) << std::endl;
        }

        // ---- Call kernel
        if (verbose) {
            std::cout << "Call kernel " << std::endl;
        }

        try {
            cutlass_scaled_fp4_mm_sm120a(D, A_bytes, B_bytes, A_sf, B_sf, alpha, gFLOPS, WARMUP_ITERATIONS, ITERATIONS, verbose);
        }
        catch(const c10::Error& e) {
            if (verbose) {
                std::cerr << "Call Kernel Exception" << std::endl;
                std::cerr << "Torch error: " << e.msg() << std::endl;
            }

            error_code = -20;
            return std::make_tuple(false, error_code, gFLOPS);
        }
        catch (const std::exception& e) {
            if (verbose) {
                std::cerr << "std::exception: " << e.what() << std::endl;
            }

            error_code = -30;
            return std::make_tuple(false, error_code, gFLOPS);
        }
        catch (...) {
            if (verbose) {
                std::cerr << "Unknown exception caught !" << std::endl;
            }

            error_code = -40;
            return std::make_tuple(false, error_code, gFLOPS);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            if (verbose) {
                std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
            }

            error_code = -50;
            return std::make_tuple(false, error_code, gFLOPS);
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            if (verbose) {
                std::cerr << "CUDA runtime error after sync: " << cudaGetErrorString(err) << std::endl;
            }

            error_code = -60;
            return std::make_tuple(false, error_code, gFLOPS);
        }

        // ---- Fetch GPU result and verify
        std::vector<float> C_gpu(M*N);
        {
            torch::Tensor D32 = D.to(torch::kFloat32).cpu();
            std::memcpy(C_gpu.data(), D32.data_ptr<float>(), sizeof(float)*M*N);
        }

        std::vector<float> C_ref(M*N);
        host_gemm_ref(A_q, B_q, C_ref, M, N, K, 1.0f);

        float max_abs;
        float max_rel;
        bool passed = verify_result(C_ref, C_gpu, M, N, max_abs, max_rel, verbose);

        if (verbose) {
            std::cout << "D[0:4,0:4] (GPU):" << std::endl;
            for (int i = 0; i < 4; i++){
                for (int j = 0; j < 4; j++) {
                    std::cout << C_gpu[i*N + j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl << std::endl;

            std::cout << "D[0:4,0:4] (CPU):" << std::endl;
            for (int i = 0; i < 4; i++){
                for (int j = 0; j < 4; j++) {
                    std::cout << C_ref[i*N + j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        if (!passed) {
            error_code = -70;
        }

        return std::make_tuple(passed, error_code, gFLOPS);
    }
    catch(const c10::Error& e) {
        if (verbose) {
            std::cerr << "Torch error: " << e.msg() << std::endl;
            std::cerr << "Detail: " << e.what() << std::endl;
        }
        error_code = -100;
    }
    catch (const std::exception& e) {
        if (verbose) {
            std::cerr << "std::exception: " << e.what() << std::endl;
        }
        error_code = -110;
    }
    catch (...) {
        if (verbose) {
            std::cerr << "Unknown exception at " << __FILE__ << ":" << __LINE__ << std::endl;
        }
        error_code = -120;
    }

    return std::make_tuple(false, error_code, gFLOPS);
}

// FP4 helpers
uint8_t Fp4Sm120aGemmVerifier::encode_e2m1_nibble(float x) {
    cutlass::float_e2m1_t q = cutlass::float_e2m1_t(x);
    uint8_t raw;
    std::memcpy(&raw, &q, 1);
    return static_cast<uint8_t>(raw & 0x0F);
}

float Fp4Sm120aGemmVerifier::decode_e2m1_nibble(uint8_t nib) {
    uint8_t raw = static_cast<uint8_t>(nib & 0x0F);
    cutlass::float_e2m1_t q;
    std::memcpy(&q, &raw, 1);
    return static_cast<float>(q);
}

uint8_t Fp4Sm120aGemmVerifier::pack2(uint8_t lo, uint8_t hi) {
    return static_cast<uint8_t>((hi << 4) | (lo & 0x0F));
}

void Fp4Sm120aGemmVerifier::unpack2(uint8_t byte, uint8_t &lo, uint8_t &hi) {
    lo = byte & 0x0F;
    hi = (byte >> 4) & 0x0F;
}

// Compute on CPU and verify
void Fp4Sm120aGemmVerifier::host_gemm_ref(const std::vector<float>& A_q, const std::vector<float>& B_q, std::vector<float>& C, int M, int N, int K, float alpha) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A_q[m*K + k] * B_q[k*N + n];
            }
            C[m*N + n] = alpha * acc;
        }
    }
}

bool Fp4Sm120aGemmVerifier::verify_result(const std::vector<float>& C_ref, const std::vector<float>& C_gpu, int M, int N, float& max_abs, float& max_rel, bool verbose, float atol, float rtol) {
    max_abs = 0.0f;
    max_rel = 0.0f;

    int wi = -1;
    int wj = -1;

    bool passed = true;
    for (int i = 0; (i < M) && passed; ++i) {
        for (int j = 0; j < N; ++j) {
            float r = C_ref[i*N + j];
            float g = C_gpu[i*N + j];

            float ae = std::fabs(r - g);
            float re = ae / (std::fabs(r) + 1e-6f);

            if (ae > max_abs) {
                max_abs = ae;
                wi = i;
                wj = j;
            }

            if (re > max_rel) {
                max_rel = re;
            }

            if (ae > atol && re > rtol) {
                if (verbose) {
                    std::cerr << "Mismatch at (" << i << "," << j << ") ref=" << r << " gpu=" << g << " abs=" << ae << " rel=" << re << "\n";
                }
                passed = false;
                break;
            }
        }
    }

    if (verbose && passed) {
        std::cout << std::fixed << std::setprecision(8) << std::setw(16) << "Verification PASSED. Max abs=" << max_abs << ", max rel=" << max_rel << " (worst at " << wi << "," << wj << ")\n";
    }

    return passed;
}

// Compute performance in GFLOP/s
double Fp4Sm120aGemmVerifier::gflops(double runtime_s) const {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * M * N * K;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
}
