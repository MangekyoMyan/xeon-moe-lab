"""Independent proof by enumeration of the corrected packed sign operation."""
import json
from pathlib import Path
from check_sign_algebra import LUTS, psign

results = []
boundary_samples = [-128, -127, -64, -1, 0, 1, 64, 127]
for name, lut in LUTS.items():
    singles = pairs = 0
    for code in range(16):
        for x in range(-128, 128):
            new_code = (code ^ (8 if x < 0 else 0)) | (8 if x == -128 else 0)
            expected = abs(lut[code]) * psign(x, lut[code])
            assert abs(x) * lut[new_code] == expected
            singles += 1
    for packed in range(256):
        for x0 in boundary_samples:
            for x1 in boundary_samples:
                signs = (8 if x0 < 0 else 0) | (128 if x1 < 0 else 0)
                force = (8 if x0 == -128 else 0) | (128 if x1 == -128 else 0)
                result = (packed ^ signs) | force
                assert abs(x0) * lut[result & 15] == abs(lut[packed & 15]) * psign(x0, lut[packed & 15])
                assert abs(x1) * lut[result >> 4] == abs(lut[packed >> 4]) * psign(x1, lut[packed >> 4])
                pairs += 1
    results.append({'type': name, 'all_signed_int8_single_products': singles, 'packed_pair_samples': pairs})

record = {
    'scope': 'Integer algebra of corrected packed code only; not floating point, dispatch, memory safety or speed.',
    'correction': 'Force the negative LUT sign for an activation equal to signed byte -128, reproducing legacy psignb wrap.',
    'results': results,
    'single_products_checked': sum(x['all_signed_int8_single_products'] for x in results),
    'packed_pair_samples_checked': sum(x['packed_pair_samples'] for x in results),
    'all_equal_to_legacy_integer_semantics': True,
}
out = Path(__file__).with_name('CORRECTED_SIGN_ALGEBRA_CHECK.json')
out.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
