import eventlet
eventlet.monkey_patch()

import argparse
import os
import threading
import time
import base64
from queue import Queue
from collections import deque

from flask import Flask, render_template
from flask_socketio import SocketIO
import numpy as np
import openwakeword
from openwakeword.model import Model

# Download models before anything else
print("Downloading required models...")
try:
    with eventlet.Timeout(100):
        openwakeword.utils.download_models()
    print("Models downloaded successfully")
except Exception as e:
    print(f"Warning: Could not download models: {e}")
    print("Continuing with local model only...")

# Parse command line arguments
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
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Audio settings
RATE = 16000
CHUNK = 2048

class WakeWordDetector:
    def __init__(self, model_path):
        print(f"Initializing model with path: {model_path}")
        try:
            self.model = Model(wakeword_models=[model_path], inference_framework='tflite')
            print("Model initialized successfully")
            print(f"Available models: {self.model.prediction_buffer.keys()}")
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
        self.is_running = False
        self.audio_queue = Queue()
        self.last_detection_time = 0
        self.processing_thread = None
        
    def start_processing(self):
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_audio_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop_processing(self):
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1)
            
    def _process_audio_queue(self):
        while self.is_running:
            try:
                # Get all available audio chunks
                chunks = []
                while not self.audio_queue.empty() and len(chunks) < 5:  # Process up to 5 chunks at once
                    chunks.append(self.audio_queue.get_nowait())
                
                if not chunks:
                    eventlet.sleep(0.01)  # Short sleep if no data
                    continue
                
                # Concatenate chunks
                audio = np.frombuffer(b''.join(chunks), dtype=np.int16)
                
                # Make prediction
                prediction = self.model.predict(audio)
                
                # Process results
                results = {}
                for mdl in self.model.prediction_buffer.keys():
                    scores = list(self.model.prediction_buffer[mdl])
                    score = float(scores[-1]) if scores else 0.0
                    detected = bool(score > 0.5)
                    results[mdl] = {
                        'score': score,
                        'detected': detected
                    }
                    
                    # Only emit if detected or every 10th frame
                    current_time = time.time()
                    if detected or (current_time - self.last_detection_time) >= 0.1:
                        self.last_detection_time = current_time
                        socketio.emit('detection_update', results)
                
            except Exception as e:
                print(f"Error processing audio: {e}")
                import traceback
                print(traceback.format_exc())
                eventlet.sleep(0.1)

detector = WakeWordDetector(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print("Client connected!")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected!")

@socketio.on('audio_data')
def handle_audio_data(audio_data):
    if detector.is_running:
        try:
            detector.audio_queue.put_nowait(bytes(audio_data))
        except Exception as e:
            print(f"Error in handle_audio_data: {e}")

@socketio.on('start_detection')
def handle_start_detection():
    print("Starting detection...")
    detector.start_processing()

@socketio.on('stop_detection')
def handle_stop_detection():
    print("Stopping detection...")
    detector.stop_processing()

if __name__ == '__main__':
    print(f"Starting server with model: {MODEL_PATH}")
    socketio.run(app, debug=True, host='0.0.0.0', port=PORT)
