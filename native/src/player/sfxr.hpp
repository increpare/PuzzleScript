#pragma once

#include <cstdint>
#include <vector>

namespace puzzlescript::player {

std::vector<float> generateSfxrFromSeed(int32_t seed, int outputSampleRate);

} // namespace puzzlescript::player
