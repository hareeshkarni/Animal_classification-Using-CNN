import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pre-trained model
model = load_model("animal.keras", compile=False)

# Ensure the uploads directory exists
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', y="")

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        if not f:
            return render_template('index.html', y="No file uploaded.")

        # Save file
        filepath = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(filepath)

        # Preprocess Image
        img = image.load_img(filepath, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Predict
        prediction = model.predict(x)
        pred_index = np.argmax(prediction)  # Corrected indexing
        index = ['bears', 'crows', 'elephants', 'rats']
        result_text = "The classified image is: " + str(index[pred_index])

        return render_template('index.html', y=result_text)

if __name__ == '__main__':
    app.run(debug=True)
