from glassbox.sr.operations.meta_ops import get_constant_symbol

tests = [
    (1.0, "should be '1', was 'pi/3'"),
    (1.047, "should be pi/3 (genuinely close)"),
    (2.0, "should be '2'"),
    (3.0, "should be '3'"),
    (-1.0, "should be '-1'"),
    (0.5, "should be '1/2'"),
    (3.14, "should be pi"),
    (2.718, "should be e"),
    (1.414, "should be sqrt(2)"),
    (0.333, "should be '1/3'"),
    (0.667, "should be '2/3'"),
    (0.75, "should be '3/4'"),
    (6.0, "should be '6'"),
    (8.0, "should be '8'"),
]

for val, desc in tests:
    result = get_constant_symbol(val, 0.05)
    # ASCII-safe output
    result_safe = result.encode("ascii", errors="backslashreplace").decode("ascii")
    print(f"  {val:8.4f}  ->  {result_safe:15s}  ({desc})")
