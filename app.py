from src.pipeline.inference_pipeline import ModelPredict
from src.entity.config_entity import InferenceConfig
from pydantic import BaseModel
from fastapi import FastAPI,HTTPException

class PredictionRequest(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: float
    sqft_basement: float
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: float


app=FastAPI(title="House Price Prediction Service")

@app.get("/")
def read_root():
    return {"message": "House Price Prediction API is functional"}

@app.post("/predict")
def predict_price(request: PredictionRequest):
    try:
        config=InferenceConfig(**request.model_dump())
        
        predictor=ModelPredict(config=config)
        prediction=predictor.predict()
        
        return {
            "status": "success",
            "prediction": float(prediction)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))