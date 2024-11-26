# app.py

import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify, render_template
import io

# Initialize Flask app
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(class_names))  # Adjust final layer
model = model.to(device)

# Load the saved model state dictionary
model_path = 'plant_disease_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set the model to evaluation mode
print(f"Model loaded from {model_path}")

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Function to predict a single image
def predict_image(image_bytes, model, transform, class_names):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return class_names[predicted.item()]

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read image bytes
        image_bytes = file.read()
        # Predict the class of the image
        predicted_class = predict_image(image_bytes, model, transform, class_names)
        # Return the prediction as JSON
        return jsonify({'predicted_class': predicted_class}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
