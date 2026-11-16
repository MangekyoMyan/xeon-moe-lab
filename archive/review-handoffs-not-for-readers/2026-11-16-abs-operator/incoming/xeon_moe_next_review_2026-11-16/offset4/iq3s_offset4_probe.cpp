// Reuse the pinned synthetic generators and baseline; never run its benchmark main.
#define main previous_abs_probe_main
#include "iq_abs_rhs_probe.cpp"
#undef main
#include "iq3s_offset4.inc"

int main() try {
    for (int h = 0; h < 65536; ++h) ggml_table_f32_f16[h] = GGML_COMPUTE_FP16_TO_FP32(h);
    std::mt19937 rng(2026111617);
    const unsigned original = _mm_getcsr();
    size_t fast_checks = 0, compatible_checks = 0, unsafe_checks = 0, recode_checks = 0;
    size_t scalar_pairs = 0;
    static const int signed_values[16] = {1,3,5,7,9,11,13,15,-1,-3,-5,-7,-9,-11,-13,-15};
    for (int byte = 0; byte < 256; ++byte) {
        iq_compact_block w{}; w.qs[0] = byte;
        offset4_iq3s_block packed{}; offset4_pack(w, packed);
        need(2*int(packed.qs[0] & 15)-15 == signed_values[byte & 15], "low nibble recode");
        need(2*int(packed.qs[0] >> 4)-15 == signed_values[byte >> 4], "high nibble recode");
        recode_checks += 2;
    }
    // Exhaustive two-value integer identity, also bounds every int16 intermediate.
    for (int a = 0; a < 16; ++a) for (int b = 0; b < 16; ++b)
    for (int y = -127; y <= 127; ++y) for (int z = -127; z <= 127; ++z) {
        const int p = a*y + b*z, c = 15*(y+z), got = 2*p-c;
        need(p >= -32768 && p <= 32767 && 2*p >= -32768 && 2*p <= 32767 &&
             c >= -32768 && c <= 32767 && got >= -32768 && got <= 32767, "int16 bound");
        need(got == (2*a-15)*y + (2*b-15)*z, "integer pair identity");
        ++scalar_pairs;
    }
    for (unsigned flush : {0u, 0x8040u}) for (unsigned rounding : {0u, 0x2000u, 0x4000u, 0x6000u}) {
        _mm_setcsr((original & ~0xe040u) | flush | rounding);
        for (int n : {256,512,768,2304,5120}) for (int trial = 0; trial < 256; ++trial) {
            const int blocks = n/256;
            std::vector<iq_compact_block> w(blocks);
            std::vector<offset4_iq3s_block> packed(blocks);
            std::vector<block_q8_K> x(blocks);
            std::vector<offset4_rhs_block> correction(blocks);
            make<3>(rng, w.data(), blocks);
            for (int j = 0; j < blocks; ++j) {
                offset4_pack(w[j], packed[j]);
                static const int exponents[] = {-149,-145,-130,-126,-125,-24,-15,0,15};
                x[j].d = std::ldexp(float(int(rng()%8191)-4095)/4096, exponents[rng()%9]);
                for (auto & value : x[j].qs) value = int(rng()%255)-127;
            }
            need(offset4_prepare(n, x.data(), correction.data()), "safe input rejected");
            const float reference = current_dot<3>(n, w.data(), x.data());
            need(fb(reference) == fb(offset4_fast_dot(n, packed.data(), x.data(), correction.data())), "fast bit mismatch");
            need(fb(reference) == fb(offset4_compat_dot(n, packed.data(), x.data())), "compat bit mismatch");
            ++fast_checks; ++compatible_checks;
            if (trial < 16) for (int mode = 0; mode < 5; ++mode) {
                const block_q8_K saved = x.back();
                if (mode == 0) x.back().qs[255] = -128;
                else {
                    static const uint32_t bad[] = {0,0x7f800000u,0xff800000u,0x7fc00001u,0xffc00001u};
                    std::memcpy(&x.back().d, &bad[mode], 4);
                }
                need(!offset4_prepare(n, x.data(), correction.data()), "unsafe accepted");
                need(fb(current_dot<3>(n, w.data(), x.data())) == fb(offset4_compat_dot(n, packed.data(), x.data())), "unsafe compatibility mismatch");
                ++unsafe_checks; ++compatible_checks;
                x.back() = saved;
            }
        }
    }
    _mm_setcsr(original);
    std::printf("{\"schema\":\"%s\",\"weight_block_bytes\":%zu,\"rhs_block_bytes\":%zu,\"recode_nibbles_checked\":%zu,\"scalar_pairs_checked\":%zu,\"fast_bit_exact_outputs\":%zu,\"compat_bit_exact_outputs\":%zu,\"unsafe_rejections_and_compat_checks\":%zu,\"performance_measured\":false,\"passed\":true}\n",
        offset4_schema, sizeof(offset4_iq3s_block), sizeof(offset4_rhs_block), recode_checks, scalar_pairs,
        fast_checks, compatible_checks, unsafe_checks);
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
