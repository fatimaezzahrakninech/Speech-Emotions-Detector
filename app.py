import os
import numpy as np
import librosa
import sounddevice as sd
import scipy.io.wavfile as wav
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

# Load the model
MODEL_PATH = "emotion_recognition_model.h5" 
model = tf.keras.models.load_model(MODEL_PATH)

# Emotion labels
emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

app = Flask(__name__)

# 📌 Extract audio features
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# 📌 Real-time audio recording
def record_audio(filename, duration=3, sr=22050):
    print("🎙️ Recording in progress...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("✅ Recording completed.")

# 📌 Emotion prediction
def predict_emotion(audio_path):
    features = extract_features(audio_path)
    features = np.expand_dims(features, axis=0) 
    prediction = model.predict(features)
    predicted_label = emotion_labels[np.argmax(prediction)]
    return predicted_label

# 📌 Flask Web Interface
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"})
        
        filepath = "temp.wav"
        file.save(filepath)
        emotion = predict_emotion(filepath)
        os.remove(filepath)  # Cleanup
        return jsonify({"emotion": emotion})

@app.route("/record", methods=["POST"])
def record():
    filename = "recorded.wav"
    record_audio(filename, duration=3)
    emotion = predict_emotion(filename)
    os.remove(filename)  # Cleanup
    return jsonify({"emotion": emotion})

if __name__ == "__main__":
    app.run(debug=True)