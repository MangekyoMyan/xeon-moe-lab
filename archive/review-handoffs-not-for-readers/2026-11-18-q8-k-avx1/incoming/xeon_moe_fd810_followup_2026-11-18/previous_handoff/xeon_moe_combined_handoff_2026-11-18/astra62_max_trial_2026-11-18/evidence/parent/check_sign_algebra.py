"""Independent integer-only check of the frozen packed-sign idea.

This checks the proposed algebra, not the child's implementation, integration,
or floating-point execution. No timing is collected.
"""
import json
from pathlib import Path

LUTS = {
    "IQ3_XXS": [4, 12, 20, 28, 36, 44, 52, 62, -4, -12, -20, -28, -36, -44, -52, -62],
    "IQ2_S": [8, 25, 43, 0, 0, 0, 0, 0, -8, -25, -43, 0, 0, 0, 0, 0],
    "IQ3_S": [1, 3, 5, 7, 9, 11, 13, 15, -1, -3, -5, -7, -9, -11, -13, -15],
}


def int8(value):
    return (value + 128) % 256 - 128


def psign(value, sign):
    return 0 if sign == 0 else (value if sign > 0 else int8(-value))


def main():
    results = []
    for name, lut in LUTS.items():
        checked = 0
        for code in range(16):
            for x in range(-127, 128):
                legacy = abs(lut[code]) * psign(x, lut[code])
                folded = abs(x) * lut[code ^ (8 if x < 0 else 0)]
                assert legacy == folded, (name, code, x)
                checked += 1
        boundary = []
        for code in range(16):
            x = -128
            legacy = abs(lut[code]) * psign(x, lut[code])
            folded = abs(x) * lut[code ^ 8]
            if legacy != folded:
                boundary.append({"code": code, "weight": lut[code], "old": legacy, "folded": folded})
        for packed in range(256):
            for x0 in [-127, -64, -1, 0, 1, 64, 127]:
                for x1 in [-127, -64, -1, 0, 1, 64, 127]:
                    mask = (8 if x0 < 0 else 0) | (128 if x1 < 0 else 0)
                    new = packed ^ mask
                    old0 = abs(lut[packed & 15]) * psign(x0, lut[packed & 15])
                    old1 = abs(lut[(packed >> 4) & 15]) * psign(x1, lut[(packed >> 4) & 15])
                    assert old0 == abs(x0) * lut[new & 15]
                    assert old1 == abs(x1) * lut[(new >> 4) & 15]
        max_mag = max(map(abs, lut))
        results.append({
            "type": name,
            "all_int8_except_minus128_single_products_checked": checked,
            "packed_pair_samples_checked": 256 * 7 * 7,
            "minus128_requires_fallback_for_legacy_equivalence": boundary,
            "maximum_pair_product_magnitude_at_127": 2 * 127 * max_mag,
            "maximum_pair_product_magnitude_at_128": 2 * 128 * max_mag,
            "signed_int16_saturation_possible_for_two_terms": 2 * 128 * max_mag > 32767,
        })
    record = {
        "scope": "integer algebra only; independent of candidate code; no benchmark",
        "sent_to_candidate_before_initial_implementation": False,
        "source": "Published compact-kernel.inc and iq-small-kernels.inc LUTs",
        "finding": "Packed activation-sign XOR is equivalent for all input int8 values except the legacy minus128 sign-negation edge. A legacy-equivalent path must handle that edge explicitly.",
        "results": results,
        "not_proved": ["candidate implementation", "alignment and planner", "floating-point accumulation order", "non-finite handling", "performance", "actual-machine applicability"],
    }
    output = Path(__file__).with_name("SIGN_ALGEBRA_CHECK.json")
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"types": len(results), "regular_products_checked": sum(r["all_int8_except_minus128_single_products_checked"] for r in results), "packed_pair_samples_checked": sum(r["packed_pair_samples_checked"] for r in results), "result": "passed within stated scope"}))


if __name__ == "__main__":
    main()
