#include <torch/extension.h>

// Include the headers of the classes you want to bind
#include "fp8/fp8_gemm_verifier.h"
#include "fp4/fp4_sm120a_gemm_verifier.h"

// The PYBIND11_MODULE macro creates a function that will be called when the Python
// module is imported. The module name (TORCH_EXTENSION_NAME) is defined in setup.py.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Python bindings for FP8 and FP4 GEMM verifiers";

    // Bind the Fp8GemmVerifier class
    py::class_<Fp8GemmVerifier>(m, "Fp8GemmVerifier")
        // Bind the constructor, including default arguments
        .def(py::init<int, int, int, int, int>(),
             py::arg("m") = 16,
             py::arg("n") = 8,
             py::arg("k") = 64,
             py::arg("warmup_iterations") = 100,
             py::arg("iterations") = 200)
        // Bind the 'verify' method, including its default argument.
        // pybind11 automatically handles the std::tuple -> Python tuple conversion.
        .def("verify", &Fp8GemmVerifier::verify,
             "Runs the FP8 GEMM verification.",
             py::arg("device_id"),
             py::arg("verbose") = true);

    // Bind the Fp4Sm120aGemmVerifier class
    py::class_<Fp4Sm120aGemmVerifier>(m, "Fp4Sm120aGemmVerifier")
        // Bind the constructor with its default arguments
        .def(py::init<int, int, int, int, int>(),
             py::arg("m") = 128,
             py::arg("n") = 128,
             py::arg("k") = 128,
             py::arg("warmup_iterations") = 100,
             py::arg("iterations") = 200)
        // Bind the 'verify' method with its default argument
        .def("verify", &Fp4Sm120aGemmVerifier::verify,
             "Runs the FP4 GEMM verification on SM120a architecture.",
             py::arg("device_id"),
             py::arg("verbose") = true);
}
