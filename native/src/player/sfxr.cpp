#include "player/sfxr.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace puzzlescript::player {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSoundVol = 0.25;
constexpr int kJsSampleRate = 5512;
constexpr int kBrowserMinSampleRate = 22050;

enum WaveType {
    Square = 0,
    Sawtooth = 1,
    Sine = 2,
    Noise = 3,
    Triangle = 4,
    Breaker = 5,
};

struct Rng {
    std::array<int, 256> s{};
    int i = 0;
    int j = 0;

    explicit Rng(const std::string& seed) {
        std::iota(s.begin(), s.end(), 0);
        if (!seed.empty()) {
            int mixJ = 0;
            for (int index = 0; index < 256; ++index) {
                mixJ = (mixJ + s[static_cast<size_t>(index)] + static_cast<unsigned char>(seed[static_cast<size_t>(index) % seed.size()])) & 255;
                std::swap(s[static_cast<size_t>(index)], s[static_cast<size_t>(mixJ)]);
            }
        }
    }

    int nextByte() {
        i = (i + 1) & 255;
        j = (j + s[static_cast<size_t>(i)]) & 255;
        std::swap(s[static_cast<size_t>(i)], s[static_cast<size_t>(j)]);
        return s[static_cast<size_t>((s[static_cast<size_t>(i)] + s[static_cast<size_t>(j)]) & 255)];
    }

    double uniform() {
        double output = 0.0;
        for (int byte = 0; byte < 7; ++byte) {
            output *= 256.0;
            output += nextByte();
        }
        return output / (std::pow(2.0, 56.0) - 1.0);
    }

    double frnd(double range) { return uniform() * range; }
    int rnd(int max) { return static_cast<int>(std::floor(uniform() * static_cast<double>(max + 1))); }
};

struct Params {
    int wave_type = Square;
    double p_env_attack = 0.0;
    double p_env_sustain = 0.3;
    double p_env_punch = 0.0;
    double p_env_decay = 0.4;
    double p_base_freq = 0.3;
    double p_freq_limit = 0.0;
    double p_freq_ramp = 0.0;
    double p_freq_dramp = 0.0;
    double p_vib_strength = 0.0;
    double p_vib_speed = 0.0;
    double p_arp_mod = 0.0;
    double p_arp_speed = 0.0;
    double p_duty = 0.0;
    double p_duty_ramp = 0.0;
    double p_repeat_speed = 0.0;
    double p_pha_offset = 0.0;
    double p_pha_ramp = 0.0;
    double p_lpf_freq = 1.0;
    double p_lpf_ramp = 0.0;
    double p_lpf_resonance = 0.0;
    double p_hpf_freq = 0.0;
    double p_hpf_ramp = 0.0;
    double sound_vol = 0.5;
};

double cube(double value) {
    return value * value * value;
}

Params pickupCoin(Rng& rng) {
    Params result;
    result.wave_type = static_cast<int>(std::floor(rng.frnd(6)));
    if (result.wave_type == Noise) result.wave_type = Square;
    result.p_base_freq = 0.4 + rng.frnd(0.5);
    result.p_env_attack = 0.0;
    result.p_env_sustain = rng.frnd(0.1);
    result.p_env_decay = 0.1 + rng.frnd(0.4);
    result.p_env_punch = 0.3 + rng.frnd(0.3);
    if (rng.rnd(1)) {
        result.p_arp_speed = 0.5 + rng.frnd(0.2);
        const int num = (static_cast<int>(rng.frnd(7)) | 1) + 1;
        const int den = num + (static_cast<int>(rng.frnd(7)) | 1) + 2;
        result.p_arp_mod = static_cast<double>(num) / static_cast<double>(den);
    }
    return result;
}

Params laserShoot(Rng& rng) {
    Params result;
    result.wave_type = rng.rnd(2);
    if (result.wave_type == Sine && rng.rnd(1)) result.wave_type = rng.rnd(1);
    result.wave_type = static_cast<int>(std::floor(rng.frnd(6)));
    if (result.wave_type == Noise) result.wave_type = Square;
    result.p_base_freq = 0.5 + rng.frnd(0.5);
    result.p_freq_limit = result.p_base_freq - 0.2 - rng.frnd(0.6);
    if (result.p_freq_limit < 0.2) result.p_freq_limit = 0.2;
    result.p_freq_ramp = -0.15 - rng.frnd(0.2);
    if (rng.rnd(2) == 0) {
        result.p_base_freq = 0.3 + rng.frnd(0.6);
        result.p_freq_limit = rng.frnd(0.1);
        result.p_freq_ramp = -0.35 - rng.frnd(0.3);
    }
    if (rng.rnd(1)) {
        result.p_duty = rng.frnd(0.5);
        result.p_duty_ramp = rng.frnd(0.2);
    } else {
        result.p_duty = 0.4 + rng.frnd(0.5);
        result.p_duty_ramp = -rng.frnd(0.7);
    }
    result.p_env_attack = 0.0;
    result.p_env_sustain = 0.1 + rng.frnd(0.2);
    result.p_env_decay = rng.frnd(0.4);
    if (rng.rnd(1)) result.p_env_punch = rng.frnd(0.3);
    if (rng.rnd(2) == 0) {
        result.p_pha_offset = rng.frnd(0.2);
        result.p_pha_ramp = -rng.frnd(0.2);
    }
    if (rng.rnd(1)) result.p_hpf_freq = rng.frnd(0.3);
    return result;
}

Params explosion(Rng& rng) {
    Params result;
    if (rng.rnd(1)) {
        result.p_base_freq = 0.1 + rng.frnd(0.4);
        result.p_freq_ramp = -0.1 + rng.frnd(0.4);
    } else {
        result.p_base_freq = 0.2 + rng.frnd(0.7);
        result.p_freq_ramp = -0.2 - rng.frnd(0.2);
    }
    result.p_base_freq *= result.p_base_freq;
    if (rng.rnd(4) == 0) result.p_freq_ramp = 0.0;
    if (rng.rnd(2) == 0) result.p_repeat_speed = 0.3 + rng.frnd(0.5);
    result.p_env_attack = 0.0;
    result.p_env_sustain = 0.1 + rng.frnd(0.3);
    result.p_env_decay = rng.frnd(0.5);
    if (rng.rnd(1) == 0) {
        result.p_pha_offset = -0.3 + rng.frnd(0.9);
        result.p_pha_ramp = -rng.frnd(0.3);
    }
    result.p_env_punch = 0.2 + rng.frnd(0.6);
    if (rng.rnd(1)) {
        result.p_vib_strength = rng.frnd(0.7);
        result.p_vib_speed = rng.frnd(0.6);
    }
    if (rng.rnd(2) == 0) {
        result.p_arp_speed = 0.6 + rng.frnd(0.3);
        result.p_arp_mod = 0.8 - rng.frnd(1.6);
    }
    return result;
}

Params pushSound(Rng& rng) {
    Params result;
    result.wave_type = static_cast<int>(std::floor(rng.frnd(6)));
    if (result.wave_type == Sine) result.wave_type++;
    if (result.wave_type == Square) result.wave_type = Noise;
    result.p_base_freq = 0.1 + rng.frnd(0.4);
    result.p_freq_ramp = 0.05 + rng.frnd(0.2);
    result.p_env_attack = 0.01 + rng.frnd(0.09);
    result.p_env_sustain = 0.01 + rng.frnd(0.09);
    result.p_env_decay = 0.01 + rng.frnd(0.09);
    result.p_repeat_speed = 0.3 + rng.frnd(0.5);
    result.p_pha_offset = -0.3 + rng.frnd(0.9);
    result.p_pha_ramp = -rng.frnd(0.3);
    result.p_arp_speed = 0.6 + rng.frnd(0.3);
    result.p_arp_mod = 0.8 - rng.frnd(1.6);
    return result;
}

Params powerUp(Rng& rng) {
    Params result;
    if (rng.rnd(1)) result.wave_type = Sawtooth; else result.p_duty = rng.frnd(0.6);
    result.wave_type = static_cast<int>(std::floor(rng.frnd(6)));
    if (result.wave_type == Noise) result.wave_type = Square;
    if (rng.rnd(1)) {
        result.p_base_freq = 0.2 + rng.frnd(0.3);
        result.p_freq_ramp = 0.1 + rng.frnd(0.4);
        result.p_repeat_speed = 0.4 + rng.frnd(0.4);
    } else {
        result.p_base_freq = 0.2 + rng.frnd(0.3);
        result.p_freq_ramp = 0.05 + rng.frnd(0.2);
        if (rng.rnd(1)) {
            result.p_vib_strength = rng.frnd(0.7);
            result.p_vib_speed = rng.frnd(0.6);
        }
    }
    result.p_env_attack = 0.0;
    result.p_env_sustain = rng.frnd(0.4);
    result.p_env_decay = 0.1 + rng.frnd(0.4);
    return result;
}

Params hitHurt(Rng& rng) {
    Params result;
    result.wave_type = rng.rnd(2);
    if (result.wave_type == Sine) result.wave_type = Noise;
    if (result.wave_type == Square) result.p_duty = rng.frnd(0.6);
    result.wave_type = static_cast<int>(std::floor(rng.frnd(6)));
    result.p_base_freq = 0.2 + rng.frnd(0.6);
    result.p_freq_ramp = -0.3 - rng.frnd(0.4);
    result.p_env_attack = 0.0;
    result.p_env_sustain = rng.frnd(0.1);
    result.p_env_decay = 0.1 + rng.frnd(0.2);
    if (rng.rnd(1)) result.p_hpf_freq = rng.frnd(0.3);
    return result;
}

Params jump(Rng& rng) {
    Params result;
    result.wave_type = Square;
    result.wave_type = static_cast<int>(std::floor(rng.frnd(6)));
    if (result.wave_type == Noise) result.wave_type = Square;
    result.p_duty = rng.frnd(0.6);
    result.p_base_freq = 0.3 + rng.frnd(0.3);
    result.p_freq_ramp = 0.1 + rng.frnd(0.2);
    result.p_env_attack = 0.0;
    result.p_env_sustain = 0.1 + rng.frnd(0.3);
    result.p_env_decay = 0.1 + rng.frnd(0.2);
    if (rng.rnd(1)) result.p_hpf_freq = rng.frnd(0.3);
    if (rng.rnd(1)) result.p_lpf_freq = 1.0 - rng.frnd(0.6);
    return result;
}

Params blipSelect(Rng& rng) {
    Params result;
    result.wave_type = rng.rnd(1);
    result.wave_type = static_cast<int>(std::floor(rng.frnd(6)));
    if (result.wave_type == Noise) result.wave_type = rng.rnd(1);
    if (result.wave_type == Square) result.p_duty = rng.frnd(0.6);
    result.p_base_freq = 0.2 + rng.frnd(0.4);
    result.p_env_attack = 0.0;
    result.p_env_sustain = 0.1 + rng.frnd(0.1);
    result.p_env_decay = rng.frnd(0.2);
    result.p_hpf_freq = 0.1;
    return result;
}

Params randomSound(Rng& rng) {
    Params result;
    result.wave_type = static_cast<int>(std::floor(rng.frnd(6)));
    result.p_base_freq = std::pow(rng.frnd(2.0) - 1.0, 2.0);
    if (rng.rnd(1)) result.p_base_freq = cube(rng.frnd(2.0) - 1.0) + 0.5;
    result.p_freq_limit = 0.0;
    result.p_freq_ramp = std::pow(rng.frnd(2.0) - 1.0, 5.0);
    if (result.p_base_freq > 0.7 && result.p_freq_ramp > 0.2) result.p_freq_ramp = -result.p_freq_ramp;
    if (result.p_base_freq < 0.2 && result.p_freq_ramp < -0.05) result.p_freq_ramp = -result.p_freq_ramp;
    result.p_freq_dramp = cube(rng.frnd(2.0) - 1.0);
    result.p_duty = rng.frnd(2.0) - 1.0;
    result.p_duty_ramp = cube(rng.frnd(2.0) - 1.0);
    result.p_vib_strength = cube(rng.frnd(2.0) - 1.0);
    result.p_vib_speed = rng.frnd(2.0) - 1.0;
    result.p_env_attack = cube(rng.frnd(2.0) - 1.0);
    result.p_env_sustain = std::pow(rng.frnd(2.0) - 1.0, 2.0);
    result.p_env_decay = rng.frnd(2.0) - 1.0;
    result.p_env_punch = std::pow(rng.frnd(0.8), 2.0);
    if (result.p_env_attack + result.p_env_sustain + result.p_env_decay < 0.2) {
        result.p_env_sustain += 0.2 + rng.frnd(0.3);
        result.p_env_decay += 0.2 + rng.frnd(0.3);
    }
    result.p_lpf_resonance = rng.frnd(2.0) - 1.0;
    result.p_lpf_freq = 1.0 - cube(rng.frnd(1.0));
    result.p_lpf_ramp = cube(rng.frnd(2.0) - 1.0);
    if (result.p_lpf_freq < 0.1 && result.p_lpf_ramp < -0.05) result.p_lpf_ramp = -result.p_lpf_ramp;
    result.p_hpf_freq = std::pow(rng.frnd(1.0), 5.0);
    result.p_hpf_ramp = std::pow(rng.frnd(2.0) - 1.0, 5.0);
    result.p_pha_offset = cube(rng.frnd(2.0) - 1.0);
    result.p_pha_ramp = cube(rng.frnd(2.0) - 1.0);
    result.p_repeat_speed = rng.frnd(2.0) - 1.0;
    result.p_arp_speed = rng.frnd(2.0) - 1.0;
    result.p_arp_mod = rng.frnd(2.0) - 1.0;
    return result;
}

Params birdSound(Rng& rng) {
    // Keep the same fallback family used by the JS bird generator for its common path.
    Params result;
    result.wave_type = static_cast<int>(std::floor(rng.frnd(6)));
    if (result.wave_type == Sawtooth || result.wave_type == Noise) result.wave_type = Sine;
    result.p_base_freq = 0.85 + rng.frnd(0.15);
    result.p_freq_ramp = 0.3 + rng.frnd(0.15);
    result.p_env_attack = rng.frnd(0.09);
    result.p_env_sustain = 0.2 + rng.frnd(0.3);
    result.p_env_decay = rng.frnd(0.1);
    result.p_duty = rng.frnd(2.0) - 1.0;
    result.p_duty_ramp = cube(rng.frnd(2.0) - 1.0);
    result.p_repeat_speed = 0.5 + rng.frnd(0.1);
    result.p_pha_offset = -0.3 + rng.frnd(0.9);
    result.p_pha_ramp = -rng.frnd(0.3);
    result.p_arp_speed = 0.4 + rng.frnd(0.6);
    result.p_arp_mod = 0.8 + rng.frnd(0.1);
    result.p_lpf_resonance = rng.frnd(2.0) - 1.0;
    result.p_lpf_freq = 1.0 - cube(rng.frnd(1.0));
    result.p_lpf_ramp = cube(rng.frnd(2.0) - 1.0);
    if (result.p_lpf_freq < 0.1 && result.p_lpf_ramp < -0.05) result.p_lpf_ramp = -result.p_lpf_ramp;
    result.p_hpf_freq = std::pow(rng.frnd(1.0), 5.0);
    result.p_hpf_ramp = std::pow(rng.frnd(2.0) - 1.0, 5.0);
    return result;
}

Params generateParamsFromSeed(int32_t seed) {
    const int rngSeed = seed / 100;
    Rng rng(std::to_string(rngSeed));
    const int generatorIndex = ((seed % 100) + 100) % 100;
    Params result;
    switch (generatorIndex % 10) {
        case 0: result = pickupCoin(rng); break;
        case 1: result = laserShoot(rng); break;
        case 2: result = explosion(rng); break;
        case 3: result = powerUp(rng); break;
        case 4: result = hitHurt(rng); break;
        case 5: result = jump(rng); break;
        case 6: result = blipSelect(rng); break;
        case 7: result = pushSound(rng); break;
        case 8: result = randomSound(rng); break;
        case 9: result = birdSound(rng); break;
    }
    result.sound_vol = kSoundVol;
    return result;
}

std::vector<float> generateSamples(const Params& ps, int32_t seed) {
    auto noiseRng = Rng("noise-" + std::to_string(seed));
    auto randNoise = [&]() { return noiseRng.frnd(2.0) - 1.0; };

    int rep_time = 0;
    double fperiod = 0.0;
    int period = 0;
    double fmaxperiod = 0.0;
    double fslide = 0.0;
    double fdslide = 0.0;
    double square_duty = 0.0;
    double square_slide = 0.0;
    double arp_mod = 0.0;
    int arp_time = 0;
    int arp_limit = 0;

    auto repeat = [&]() {
        rep_time = 0;
        fperiod = 100.0 / (ps.p_base_freq * ps.p_base_freq + 0.001);
        period = static_cast<int>(std::floor(fperiod));
        fmaxperiod = 100.0 / (ps.p_freq_limit * ps.p_freq_limit + 0.001);
        fslide = 1.0 - cube(ps.p_freq_ramp) * 0.01;
        fdslide = -cube(ps.p_freq_dramp) * 0.000001;
        square_duty = 0.5 - ps.p_duty * 0.5;
        square_slide = -ps.p_duty_ramp * 0.00005;
        arp_mod = ps.p_arp_mod >= 0.0 ? 1.0 - ps.p_arp_mod * ps.p_arp_mod * 0.9 : 1.0 + ps.p_arp_mod * ps.p_arp_mod * 10.0;
        arp_time = 0;
        arp_limit = static_cast<int>(std::floor(std::pow(1.0 - ps.p_arp_speed, 2.0) * 20000.0 + 32.0));
        if (ps.p_arp_speed == 1.0) arp_limit = 0;
    };
    repeat();

    double fltp = 0.0;
    double fltdp = 0.0;
    double fltw = cube(ps.p_lpf_freq) * 0.1;
    const double fltw_d = 1.0 + ps.p_lpf_ramp * 0.0001;
    double fltdmp = 5.0 / (1.0 + ps.p_lpf_resonance * ps.p_lpf_resonance * 20.0) * (0.01 + fltw);
    if (fltdmp > 0.8) fltdmp = 0.8;
    double fltphp = 0.0;
    double flthp = ps.p_hpf_freq * ps.p_hpf_freq * 0.1;
    const double flthp_d = 1.0 + ps.p_hpf_ramp * 0.0003;

    double vib_phase = 0.0;
    const double vib_speed = ps.p_vib_speed * ps.p_vib_speed * 0.01;
    const double vib_amp = ps.p_vib_strength * 0.5;

    double env_vol = 0.0;
    int env_stage = 0;
    int env_time = 0;
    std::array<int, 3> env_length{
        static_cast<int>(std::floor(ps.p_env_attack * ps.p_env_attack * 100000.0)),
        static_cast<int>(std::floor(ps.p_env_sustain * ps.p_env_sustain * 100000.0)),
        static_cast<int>(std::floor(ps.p_env_decay * ps.p_env_decay * 100000.0)),
    };
    const int env_total_length = env_length[0] + env_length[1] + env_length[2];

    double fphase = ps.p_pha_offset * ps.p_pha_offset * 1020.0;
    if (ps.p_pha_offset < 0.0) fphase = -fphase;
    double fdphase = ps.p_pha_ramp * ps.p_pha_ramp;
    if (ps.p_pha_ramp < 0.0) fdphase = -fdphase;
    int iphase = std::abs(static_cast<int>(std::floor(fphase)));
    int ipp = 0;
    std::array<double, 1024> phaser_buffer{};

    std::array<double, 32> noise_buffer{};
    for (double& value : noise_buffer) value = randNoise();

    int rep_limit = static_cast<int>(std::floor(std::pow(1.0 - ps.p_repeat_speed, 2.0) * 20000.0 + 32.0));
    if (ps.p_repeat_speed == 0.0) rep_limit = 0;
    const double gain = std::exp(ps.sound_vol) - 1.0;
    const int summands = static_cast<int>(std::floor(44100.0 / static_cast<double>(kJsSampleRate)));
    const bool browserUpsample = kJsSampleRate < kBrowserMinSampleRate;
    const int upsampleFactor = browserUpsample ? 4 : 1;
    const int buffer_length = std::max(1, static_cast<int>(std::ceil(static_cast<double>(env_total_length) / static_cast<double>(summands))) * upsampleFactor + upsampleFactor);
    std::vector<float> buffer;
    buffer.reserve(static_cast<size_t>(buffer_length));

    double sample_sum = 0.0;
    int num_summed = 0;
    int phase = 0;
    bool buffer_complete = false;
    for (int t = 0; !buffer_complete; ++t) {
        if (rep_limit != 0 && ++rep_time >= rep_limit) repeat();
        if (arp_limit != 0 && t >= arp_limit) {
            arp_limit = 0;
            fperiod *= arp_mod;
        }
        fslide += fdslide;
        fperiod *= fslide;
        if (fperiod > fmaxperiod) {
            fperiod = fmaxperiod;
            if (ps.p_freq_limit > 0.0) buffer_complete = true;
        }
        double rfperiod = fperiod;
        if (vib_amp > 0.0) {
            vib_phase += vib_speed;
            rfperiod = fperiod * (1.0 + std::sin(vib_phase) * vib_amp);
        }
        period = static_cast<int>(std::floor(rfperiod));
        if (period < 8) period = 8;
        square_duty += square_slide;
        square_duty = std::clamp(square_duty, 0.0, 0.5);

        env_time++;
        if (env_time > env_length[static_cast<size_t>(env_stage)]) {
            env_time = 1;
            env_stage++;
            while (env_stage < 3 && env_length[static_cast<size_t>(env_stage)] == 0) env_stage++;
            if (env_stage == 3) break;
        }
        if (env_stage == 0) {
            env_vol = env_length[0] == 0 ? 0.0 : static_cast<double>(env_time) / static_cast<double>(env_length[0]);
        } else if (env_stage == 1) {
            env_vol = 1.0 + (1.0 - static_cast<double>(env_time) / static_cast<double>(env_length[1])) * 2.0 * ps.p_env_punch;
        } else {
            env_vol = 1.0 - static_cast<double>(env_time) / static_cast<double>(env_length[2]);
        }

        fphase += fdphase;
        iphase = std::abs(static_cast<int>(std::floor(fphase)));
        if (iphase > 1023) iphase = 1023;
        if (flthp_d != 0.0) {
            flthp *= flthp_d;
            flthp = std::clamp(flthp, 0.00001, 0.1);
        }

        double sample = 0.0;
        for (int si = 0; si < 8; ++si) {
            double sub_sample = 0.0;
            phase++;
            if (phase >= period) {
                phase %= period;
                if (ps.wave_type == Noise) {
                    for (double& value : noise_buffer) value = randNoise();
                }
            }
            const double fp = static_cast<double>(phase) / static_cast<double>(period);
            if (ps.wave_type == Square) sub_sample = fp < square_duty ? 0.5 : -0.5;
            else if (ps.wave_type == Sawtooth) sub_sample = 1.0 - fp * 2.0;
            else if (ps.wave_type == Sine) sub_sample = std::sin(fp * 2.0 * kPi);
            else if (ps.wave_type == Noise) sub_sample = noise_buffer[static_cast<size_t>(std::clamp(static_cast<int>(std::floor(phase * 32.0 / period)), 0, 31))];
            else if (ps.wave_type == Triangle) sub_sample = std::abs(1.0 - fp * 2.0) - 1.0;
            else if (ps.wave_type == Breaker) sub_sample = std::abs(1.0 - fp * fp * 2.0) - 1.0;

            const double pp = fltp;
            fltw *= fltw_d;
            fltw = std::clamp(fltw, 0.0, 0.1);
            if (ps.p_lpf_freq != 1.0) {
                fltdp += (sub_sample - fltp) * fltw;
                fltdp -= fltdp * fltdmp;
            } else {
                fltp = sub_sample;
                fltdp = 0.0;
            }
            fltp += fltdp;
            fltphp += fltp - pp;
            fltphp -= fltphp * flthp;
            sub_sample = fltphp;
            phaser_buffer[static_cast<size_t>(ipp & 1023)] = sub_sample;
            sub_sample += phaser_buffer[static_cast<size_t>((ipp - iphase + 1024) & 1023)];
            ipp = (ipp + 1) & 1023;
            sample += sub_sample * env_vol;
        }

        sample_sum += sample;
        if (++num_summed < summands) continue;
        num_summed = 0;
        sample = sample_sum / static_cast<double>(summands);
        sample_sum = 0.0;
        sample = sample / 8.0 * gain;
        for (int repeatIndex = 0; repeatIndex < upsampleFactor; ++repeatIndex) {
            buffer.push_back(static_cast<float>(sample));
        }
    }

    if (summands > 0) {
        double sample = sample_sum / static_cast<double>(summands);
        sample = sample / 8.0 * gain;
        for (int repeatIndex = 0; repeatIndex < upsampleFactor; ++repeatIndex) {
            buffer.push_back(static_cast<float>(sample));
        }
    }
    return buffer;
}

void applyBrowserLowpassFilters(std::vector<float>& samples) {
    if (samples.empty()) {
        return;
    }
    const double omega = 2.0 * kPi * 1600.0 / static_cast<double>(kBrowserMinSampleRate);
    const double sn = std::sin(omega);
    const double cs = std::cos(omega);
    const double alpha = sn / 2.0;
    double b0 = (1.0 - cs) / 2.0;
    double b1 = 1.0 - cs;
    double b2 = (1.0 - cs) / 2.0;
    const double a0 = 1.0 + alpha;
    double a1 = -2.0 * cs;
    double a2 = 1.0 - alpha;
    b0 /= a0;
    b1 /= a0;
    b2 /= a0;
    a1 /= a0;
    a2 /= a0;

    for (int pass = 0; pass < 3; ++pass) {
        double x1 = 0.0;
        double x2 = 0.0;
        double y1 = 0.0;
        double y2 = 0.0;
        for (float& value : samples) {
            const double x0 = value;
            const double y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            value = static_cast<float>(std::clamp(y0, -1.0, 1.0));
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
        }
    }
}

std::vector<float> resample(const std::vector<float>& input, int inputRate, int outputRate) {
    if (input.empty() || inputRate <= 0 || outputRate <= 0 || inputRate == outputRate) {
        return input;
    }
    const size_t outputCount = std::max<size_t>(1, static_cast<size_t>(std::ceil(static_cast<double>(input.size()) * outputRate / inputRate)));
    std::vector<float> output(outputCount);
    const double ratio = static_cast<double>(inputRate) / static_cast<double>(outputRate);
    for (size_t index = 0; index < output.size(); ++index) {
        const double source = static_cast<double>(index) * ratio;
        const size_t left = std::min(input.size() - 1, static_cast<size_t>(std::floor(source)));
        const size_t right = std::min(input.size() - 1, left + 1);
        const double t = source - std::floor(source);
        output[index] = static_cast<float>(input[left] * (1.0 - t) + input[right] * t);
    }
    return output;
}

} // namespace

std::vector<float> generateSfxrFromSeed(int32_t seed, int outputSampleRate) {
    const Params params = generateParamsFromSeed(seed);
    auto samples = generateSamples(params, seed);
    applyBrowserLowpassFilters(samples);
    return resample(samples, kBrowserMinSampleRate, outputSampleRate);
}

} // namespace puzzlescript::player
