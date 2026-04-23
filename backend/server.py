from flask import Flask
from flask import request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import kagglehub
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
app = Flask(__name__)
CORS(app)  

model = tf.keras.models.load_model('ann_model.keras')
model2 = tf.keras.models.load_model('cnn_model.keras')
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
@app.route('/predictann', methods=['POST'])
def ann_evaluate_img():
    my_file = request.files['file']
    img = Image.open(my_file).convert("RGB").resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    probabilities = [float(p) for p in prediction[0]]  

    return jsonify({
        'probabilities': probabilities,
        'classes': classes
    })
@app.route('/predictcnn', methods=['POST'])
def cnn_evaluate_img():
    my_file = request.files['file']
    img = Image.open(my_file).convert("RGB").resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model2.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    predicted_class = classes[predicted_index]
    probabilities = [float(p) for p in prediction[0]]  

    return jsonify({
        "prediction": predicted_class,
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)