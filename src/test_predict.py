from src.modeling.scratch.linear_regression.predict import predict as scratch_linear_predict
from src.modeling.scratch.poly_regression.predict import predict as scratch_poly_predict
from src.modeling.scratch.multiple_regression.predict import predict as scratch_multiple_predict

from src.modeling.lib.linear_regression.predict import predict as lib_linear_predict
from src.modeling.lib.poly_regression.predict import predict as lib_poly_predict
from src.modeling.lib.multiple_regression.predict import predict as lib_multiple_predict


def _compare(name, scratch_val, lib_val):
    print(f"\n  {name}")
    print(f"    Scratch : {scratch_val}")
    print(f"    Lib     : {lib_val}")


if __name__ == "__main__":
    raw_input = input("Enter features separated by comma: ")
    values = list(map(float, raw_input.split(",")))

    print("\n" + "=" * 60)
    print("  Prediction Comparison: Scratch vs Lib (sklearn)")
    print("=" * 60)

    _compare(
        "Linear Regression",
        scratch_linear_predict(values),
        lib_linear_predict(values),
    )
    _compare(
        "Polynomial Regression",
        scratch_poly_predict(values),
        lib_poly_predict(values),
    )
    _compare(
        "Multiple Regression",
        scratch_multiple_predict(values),
        lib_multiple_predict(values),
    )

    print("\n" + "=" * 60)