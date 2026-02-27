from fastapi import FastAPI, Query
from typing import Literal
from src.modeling.regression.scratch.linear_regression.predict import predict as scratch_linear_regression_predict
from src.modeling.regression.lib.linear_regression.predict import predict as sklearn_linear_regression_predict
from src.modeling.regression.scratch.poly_regression.predict import predict as scratch_poly_regression_predict
from src.modeling.regression.lib.poly_regression.predict import predict as sklearn_poly_regression_predict
from src.modeling.regression.scratch.multiple_regression.predict import predict as scratch_multiple_regression_predict
from src.modeling.regression.lib.multiple_regression.predict import predict as sklearn_multiple_regression_predict
from src.modeling.regression.scratch.mlp_regression.predict import predict as mlp_regression_predict

app = FastAPI()

@app.get("/predict_price")
def predict_price(
    gpu_tier: int = Query(..., ge=1, le=6),
    ram_gb: int = Query(..., ge=8, le=144),
    resolution: Literal['2560x1440', '1920x1080', '3440x1440', '2560x1600', '3840x2160', '2880x1800'] = Query(...),
    cpu_tier: int = Query(..., ge=1, le=6),
    os: Literal['Windows', 'macOS', 'Linux', 'ChromeOS'] = Query(...),
    cpu_threads: int = Query(..., ge=4, le=56),
    cpu_cores: int = Query(..., ge=4, le=28),
):
    input_data = {
        "gpu_tier": gpu_tier,
        "ram_gb": ram_gb,
        "resolution": resolution,
        "cpu_tier": cpu_tier,
        "os": os,
        "cpu_threads": cpu_threads,
        "cpu_cores": cpu_cores,
    }

    return {
        **scratch_linear_regression_predict(input_data),
        **sklearn_linear_regression_predict(input_data),
        **scratch_poly_regression_predict(input_data),
        **sklearn_poly_regression_predict(input_data),
        **scratch_multiple_regression_predict(input_data),
        **sklearn_multiple_regression_predict(input_data),
        **mlp_regression_predict(input_data),
    }