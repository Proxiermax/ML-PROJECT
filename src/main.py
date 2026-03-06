from fastapi import FastAPI, Query
from typing import Literal
from src.modeling.regression.scratch.linear_regression.predict import predict as scratch_linear_regression_predict
from src.modeling.regression.lib.linear_regression.predict import predict as sklearn_linear_regression_predict
from src.modeling.regression.scratch.poly_regression.predict import predict as scratch_poly_regression_predict
from src.modeling.regression.lib.poly_regression.predict import predict as sklearn_poly_regression_predict
from src.modeling.regression.scratch.multiple_regression.predict import predict as scratch_multiple_regression_predict
from src.modeling.regression.lib.multiple_regression.predict import predict as sklearn_multiple_regression_predict
from src.modeling.regression.scratch.gradient_boosting.predict import predict as gradient_boosting_regression_predict

app = FastAPI()

@app.get("/predict_price")
def predict_price(
    gpu_tier: int = Query(..., ge=1, le=6),
    cpu_tier: int = Query(..., ge=1, le=6),
    ram_gb: int = Query(..., ge=8, le=144),
    cpu_cores: int = Query(..., ge=4, le=28),
    cpu_threads: int = Query(..., ge=4, le=56),
    device_type: Literal['Laptop', 'Desktop'] = Query(...),
    resolution: Literal['2560x1440', '1920x1080', '3440x1440', '2560x1600', '3840x2160', '2880x1800'] = Query(...),
    os: Literal['Windows', 'macOS', 'Linux', 'ChromeOS'] = Query(...),
    
    
):
    input_data = {
        "gpu_tier": gpu_tier,
        "cpu_tier": cpu_tier,
        "ram_gb": ram_gb,
        "cpu_cores": cpu_cores,
        "cpu_threads": cpu_threads,
        "device_type": device_type,
        "resolution": resolution,
        "os": os,
    }

    return {
        **scratch_linear_regression_predict(input_data),
        **sklearn_linear_regression_predict(input_data),
        **scratch_poly_regression_predict(input_data),
        **sklearn_poly_regression_predict(input_data),
        **scratch_multiple_regression_predict(input_data),
        **sklearn_multiple_regression_predict(input_data),
        **gradient_boosting_regression_predict(input_data),
    }