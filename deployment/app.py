from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Iris Classifier")

model = joblib.load("artifacts/model.joblib")

class Schema(BaseModel):
    sepal_length: float 
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message":"mlops assignment"}

@app.post("/predict")
def prediction(data: Schema):
    input = pd.DataFrame([data.dict()])
    output = model.predict(input)[0]
    return {"predicted_class":output}