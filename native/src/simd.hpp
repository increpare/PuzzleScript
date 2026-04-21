#pragma once

#include <string_view>

namespace puzzlescript {

enum class SimdBackend {
    Scalar,
    SSE2,
    AVX2,
    NEON,
};

SimdBackend detectBestBackend();
std::string_view simdBackendName(SimdBackend backend);

} // namespace puzzlescript
