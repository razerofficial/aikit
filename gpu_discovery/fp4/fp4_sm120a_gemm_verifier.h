// fp4_sm120a_gemm_verifier.h

#include <tuple>
#include <vector>
#include <cutlass/numeric_types.h>

class Fp4Sm120aGemmVerifier {
public:
    Fp4Sm120aGemmVerifier(int m = 128, int n = 128, int k = 128, int warmup_iterations = 100, int iterations = 200);
    virtual ~Fp4Sm120aGemmVerifier();

    std::tuple<bool,int, double> verify(int device_id, bool verbose = true);

protected:
    // FP4 helpers
    uint8_t encode_e2m1_nibble(float x);
    float decode_e2m1_nibble(uint8_t nib);

    uint8_t pack2(uint8_t lo, uint8_t hi);
    void unpack2(uint8_t byte, uint8_t &lo, uint8_t &hi);

    // Compute on CPU and verify
    void host_gemm_ref(const std::vector<float>& A_q, const std::vector<float>& B_q, std::vector<float>& C, int M, int N, int K, float alpha = 1.f);
    bool verify_result(const std::vector<float>& C_ref, const std::vector<float>& C_gpu, int M, int N, float& max_abs, float& max_rel, bool verbose = true, float atol = 2e-2f, float rtol = 2e-2f);

    // Compute performance in GFLOP/s
    double gflops(double runtime_s) const;

private:
    int M, N, K, WARMUP_ITERATIONS, ITERATIONS;
};
