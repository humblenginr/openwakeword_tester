from flask import Flask, render_template
from flask_socketio import SocketIO
import pyaudio
import numpy as np
from openwakeword.model import Model
import openwakeword
import threading
import time
import eventlet
import argparse
import os

eventlet.monkey_patch()

# Parse command line arguments or use environment variables
parser = argparse.ArgumentParser(description='Wake Word Detection Server')
parser.add_argument('--model_path', type=str, help='Path to the custom wake word model')
parser.add_argument('--port', type=int, help='Port to run the server on')
args = parser.parse_args()

# Use environment variables as fallback
MODEL_PATH = args.model_path or os.environ.get('MODEL_PATH')
PORT = args.port or int(os.environ.get('PORT', 8333))

if not MODEL_PATH:
    raise ValueError("Model path must be provided either via --model_path argument or MODEL_PATH environment variable")

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

class WakeWordDetector:
    def __init__(self, model_path):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        self.model = Model(wakeword_models=[model_path], inference_framework='tflite')
        
    def start_stream(self):
        if self.stream is None:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
        self.is_running = True
        
    def stop_stream(self):
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
    def process_audio(self):
        while self.is_running:
            try:
                audio = np.frombuffer(self.stream.read(CHUNK), dtype=np.int16)
                prediction = self.model.predict(audio)
                
                results = {}
                for mdl in self.model.prediction_buffer.keys():
                    scores = list(self.model.prediction_buffer[mdl])
                    results[mdl] = {
                        'score': float(scores[-1]),
                        'detected': bool(scores[-1] > 0.5)
                    }
                
                socketio.emit('detection_update', results)
                eventlet.sleep(0)
            except Exception as e:
                print(f"Error processing audio: {e}")
                break

detector = WakeWordDetector(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_detection')
def handle_start_detection():
    if not detector.is_running:
        detector.start_stream()
        threading.Thread(target=detector.process_audio).start()

@socketio.on('stop_detection')
def handle_stop_detection():
    detector.stop_stream()

if __name__ == '__main__':
    print(f"Starting server with model: {MODEL_PATH}")
    socketio.run(app, debug=True, host='0.0.0.0', port=PORT)
