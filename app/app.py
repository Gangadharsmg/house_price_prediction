from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Load the pre-trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Initialize FastAPI app
app = FastAPI()


# Define the input data structure
class HouseData(BaseModel):
    area: float
    bedrooms: int
    stories: int

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API"}


# Prediction endpoint
@app.post("/predict")
def predict_price(data: HouseData):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Predict house price
    prediction = model.predict(input_data)[0]
    
    return {
        "input": data.dict(),
        "predicted_price": prediction
    }
