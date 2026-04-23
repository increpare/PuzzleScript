#include "runtime/simd.hpp"

namespace puzzlescript {

SimdBackend detectBestBackend() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return SimdBackend::NEON;
#elif defined(__x86_64__) || defined(_M_X64)
#if defined(__clang__) || defined(__GNUC__)
    if (__builtin_cpu_supports("avx2")) {
        return SimdBackend::AVX2;
    }
    if (__builtin_cpu_supports("sse2")) {
        return SimdBackend::SSE2;
    }
#endif
    return SimdBackend::Scalar;
#else
    return SimdBackend::Scalar;
#endif
}

std::string_view simdBackendName(SimdBackend backend) {
    switch (backend) {
        case SimdBackend::Scalar: return "scalar";
        case SimdBackend::SSE2: return "sse2";
        case SimdBackend::AVX2: return "avx2";
        case SimdBackend::NEON: return "neon";
    }
    return "scalar";
}

} // namespace puzzlescript
