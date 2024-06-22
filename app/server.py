from fastapi import FastAPI, File, UploadFile
from tempfile import NamedTemporaryFile
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

app = FastAPI()

# Load the pre-trained model
model = load_model('/code/app/mymodel.h5')  # Adjust the path as necessary

# Class labels for prediction
class_names = np.array(['Fractured', 'Not Fractured'])

@app.get('/')
def read_root():
    return {'message': 'Fracture Detection API'}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=True, suffix='.png') as temp:
        temp.write(await file.read())
        temp.flush()

        # Open image using PIL and resize to 224x224
        img = Image.open(temp.name).convert("RGB")
        img = img.resize((224, 224))

        # Convert image to numpy array and preprocess for the model
        img_array = keras_image.img_to_array(img)
        img_array /= 255.0  # Normalize image pixels
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the loaded model
        prediction = model.predict(img_array)
        prediction = (prediction > 0.5).astype("int32")[0][0]
        predicted_class = class_names[prediction]

        return {'predicted_class': predicted_class}
