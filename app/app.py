from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Replace with the actual tracking URI

# Define model loading function
def load_model(run_id, artifact_path="model.pkl"):
    """
    Load the model.pkl file from MLflow artifacts.

    Args:
        run_id (str): The MLflow run ID where the model is stored.
        artifact_path (str): Path to the artifact file (default is 'model.pkl').

    Returns:
        model: The loaded model object.
    """
    # Construct the artifact URI
    artifact_uri = mlflow.get_artifact_uri(run_id)
    model_path = os.path.join(artifact_uri, artifact_path)

    # Download the artifact locally
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)

    # Load the model
    with open(local_path, "rb") as f:
        model = pickle.load(f)
    return model

# Initialize FastAPI app
app = FastAPI()

run_id = "0"  
model = load_model(run_id)

# Define the input data structure
class HouseData(BaseModel):
    area: int
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

