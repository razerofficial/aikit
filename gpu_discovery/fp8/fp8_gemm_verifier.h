// fp8_gemm_verifier.h
#include <tuple>

class Fp8GemmVerifier {
public:
    Fp8GemmVerifier(int m = 16, int n = 8, int k = 64, int warmup_iterations = 100, int iterations = 200);

    std::tuple<bool,int, double> verify(int device_id, bool verbose = true);

protected:
    // Compute performance in GFLOP/s
    double gflops(double runtime_s) const;

private:
    int M, N, K, WARMUP_ITERATIONS, ITERATIONS;
    float fp8_min, fp8_max;
};
