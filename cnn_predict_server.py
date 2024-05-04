from flask import Flask, request, jsonify
import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

def extract_and_predict_genre(audio_file):
    # Define the number of expected features per type
    column_counts = {
        'chroma_cens': 84,
        'chroma_cqt': 84,
        'chroma_stft': 84,
        'mfcc': 140,
        'rmse': 7,
        'spectral_bandwidth': 7,
        'spectral_centroid': 7,
        'spectral_contrast': 49,
        'spectral_rolloff': 7,
        'tonnetz': 42,
        'zcr': 7
    }

    scaler = joblib.load('scaler_cnn.pkl')
    model = load_model('cnn_genre_classification.h5')
    label_encoder = joblib.load('label_encoder_cnn.pkl')

    y, sr = librosa.load(audio_file, sr=None)
    features = {
        'chroma_cens': librosa.feature.chroma_cens(y=y, sr=sr),
        'chroma_cqt': librosa.feature.chroma_cqt(y=y, sr=sr),
        'chroma_stft': librosa.feature.chroma_stft(y=y, sr=sr),
        'mfcc': librosa.feature.mfcc(y=y, sr=sr),
        'rmse': librosa.feature.rms(y=y),
        'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr),
        'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr),
        'spectral_contrast': librosa.feature.spectral_contrast(y=y, sr=sr),
        'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr),
        'tonnetz': librosa.feature.tonnetz(y=y, sr=sr),
        'zcr': librosa.feature.zero_crossing_rate(y)
    }

    feature_vector = []
    column_names = []
    for feature_name, feature_array in features.items():
        feature_array = feature_array.flatten()[:column_counts[feature_name]]
        for i in range(column_counts[feature_name]):
            column_names.append(f'{feature_name}.{i}' if i > 0 else feature_name)
        feature_vector.extend(feature_array)

    features_df = pd.DataFrame([feature_vector], columns=column_names)
    scaled_features = scaler.transform(features_df)
    reshaped_features = scaled_features.reshape(1, 1, 518, 1)

    predictions = model.predict(reshaped_features)
    predicted_genre_index = np.argmax(predictions, axis=1)
    predicted_genre = label_encoder.inverse_transform(predicted_genre_index)

    return predicted_genre

@app.route('/predict_genre', methods=['POST'])
def predict_genre():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        # Save the file to a temporary file or process in memory
        filename = 'temp_audio_file.wav'
        file.save(filename)
        predicted_genre = extract_and_predict_genre(filename)
        return jsonify({'genre': predicted_genre[0]})

if __name__ == '__main__':
    app.run(debug=True, host= '0.0.0.0', port=5000)
