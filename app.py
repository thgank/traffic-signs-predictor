from __future__ import division, print_function
import os
import numpy as np
import pandas as pd
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
IMAGE_SIZE = (32, 32)
LABELS_FILE = 'labels.csv'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# сгрузка названий знаков из датасета
def load_label_names(labels_file):
    """
    Load label names from a CSV file.
    """
    labels_df = pd.read_csv(labels_file)
    return {row['ClassId']: row['Name'] for _, row in labels_df.iterrows()}


LABEL_NAMES = load_label_names(LABELS_FILE)

# модели на выбор
# CNN 
try:
    from tensorflow.keras.models import load_model
    cnn_model = load_model('cnn_model.keras')
except Exception as e:
    print(f"Error loading CNN model: {e}")
    cnn_model = None

# MLP 
try:
    import joblib
    mlp_model = joblib.load('mlp_model.joblib')
except Exception as e:
    print(f"Error loading MLP model: {e}")
    mlp_model = None

# RFC 
try:
    import joblib
    rfc_model = joblib.load('rfc_model.joblib')
except Exception as e:
    print(f"Error loading RFC model: {e}")
    rfc_model = None

# импорт модуля HOG в случае если выбрана rfc
if rfc_model:
    from skimage.feature import hog
    # параметры HOG
    HOG_PARAMS = {
        'orientations': 9,
        'pixels_per_cell': (8, 8),
        'cells_per_block': (2, 2),
        'block_norm': 'L2-Hys',
        'transform_sqrt': True,
        'feature_vector': True
    }


def preprocess_for_cnn(img):
    """
    препроцесс для CNN: изменение гаммы, нормалайз и решэйп.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized_img = cv2.equalizeHist(gray_img)
    normalized_img = equalized_img / 255.0
    preprocessed_img = normalized_img.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    return preprocessed_img

def preprocess_for_mlp(img):
    """
    препроцесс для MLP, схожие запросы как для CNN
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized_img = cv2.equalizeHist(gray_img)
    normalized_img = equalized_img / 255.0
    flattened_img = normalized_img.flatten().reshape(1, -1)
    return flattened_img

def preprocess_for_rfc(img):
    """
    препроцесс для RFC, схожие запросы, но теперь добавляются HOG параметры
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, IMAGE_SIZE)
    hog_features = hog(
        gray_img,
        orientations=HOG_PARAMS['orientations'],
        pixels_per_cell=HOG_PARAMS['pixels_per_cell'],
        cells_per_block=HOG_PARAMS['cells_per_block'],
        block_norm=HOG_PARAMS['block_norm'],
        transform_sqrt=HOG_PARAMS['transform_sqrt'],
        feature_vector=HOG_PARAMS['feature_vector']
    ).reshape(1, -1)
    return hog_features

def get_label_name(class_index):
    """
    возврат названия знака исходя из индекса.
    """
    return LABEL_NAMES.get(class_index, "Unknown")

def predict_label_from_image(img_path, model_type):
    """
    обработка картинки, возврат знака с использованием выбранной модели..
    """
    img = cv2.imread(img_path)
    if img is None:
        return "Error: Unable to read image"

    img = cv2.resize(img, IMAGE_SIZE)

    if model_type == 'cnn' and cnn_model:
        preprocessed_img = preprocess_for_cnn(img)
        predictions = cnn_model.predict(preprocessed_img)
        class_index = np.argmax(predictions, axis=1)[0]
        return get_label_name(class_index)
    elif model_type == 'mlp' and mlp_model:
        preprocessed_img = preprocess_for_mlp(img)
        class_index = mlp_model.predict(preprocessed_img)[0]
        return get_label_name(class_index)
    elif model_type == 'rfc' and rfc_model:
        preprocessed_img = preprocess_for_rfc(img)
        class_index = rfc_model.predict(preprocessed_img)[0]
        return get_label_name(class_index)
    else:
        return "Error: Selected model is not available"

@app.route('/', methods=['GET', 'POST'])
def home():

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        model_type = request.form.get('model')
        if model_type not in ['cnn', 'mlp', 'rfc']:
            return render_template('index.html', error="Invalid model selected")

        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(file_path)

        result = predict_label_from_image(file_path, model_type)
        return render_template('result.html', prediction=result, model_name=model_type.upper())

    return render_template('index.html')

@app.route('/predict-label', methods=['POST'])
def predict_label():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    model_type = request.form.get('model')
    if model_type not in ['cnn', 'mlp', 'rfc']:
        return "Invalid model selected", 400

    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(file_path)

    result = predict_label_from_image(file_path, model_type)
    return result


if __name__ == '__main__':
    app.run(port=5001, debug=True)
