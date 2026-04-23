#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "puzzlescript/puzzlescript.h"

namespace puzzlescript {

uint64_t fnv1a64(const void* data, size_t size, uint64_t seed = 1469598103934665603ull);
uint64_t fnv1a64String(std::string_view text, uint64_t seed = 1469598103934665603ull);
ps_hash128 dualHash128(const std::vector<uint8_t>& bytes);

} // namespace puzzlescript
