#include "hash.hpp"

namespace puzzlescript {

uint64_t fnv1a64(const void* data, size_t size, uint64_t seed) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t value = seed;
    for (size_t index = 0; index < size; ++index) {
        value ^= bytes[index];
        value *= 1099511628211ull;
    }
    return value;
}

uint64_t fnv1a64String(std::string_view text, uint64_t seed) {
    return fnv1a64(text.data(), text.size(), seed);
}

ps_hash128 dualHash128(const std::vector<uint8_t>& bytes) {
    ps_hash128 result{};
    result.lo = fnv1a64(bytes.data(), bytes.size(), 1469598103934665603ull);
    result.hi = fnv1a64(bytes.data(), bytes.size(), 7809847782465536322ull);
    return result;
}

} // namespace puzzlescript
