from fastapi import FastAPI
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.losses import MeanSquaredError
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the trained model with custom objects
model = load_model("model.h5", custom_objects={"mse": MeanSquaredError()})

# Define the input data structure
class InputData(BaseModel):
    data: list

# Define a route for prediction
@app.post("/predict")
async def predict(input_data: InputData):
    try:
        # Convert input data to numpy array
        input_array = np.array(input_data.data, dtype=np.float32).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)

        return {"prediction": prediction.tolist()}

    except Exception as e:
        return {"error": str(e)}

# Root endpoint
@app.get("/")
def home():
    return {"message": "ML Model API is running!"}