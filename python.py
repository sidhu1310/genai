from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import os
import librosa
import tempfile
import sqlite3
import json
from datetime import datetime

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  

DB_PATH = "transcriptions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transcriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        filename TEXT,
        transcription TEXT,
        file_info TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

init_db()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("Loading Whisper model...")
try:

    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3")
    MODEL_LOADED = True
    print("Whisper model loaded successfully")
except Exception as e:
    MODEL_LOADED = False
    print(f"Error loading Whisper model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if not MODEL_LOADED:
        return jsonify({"error": "Whisper model failed to load"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_path = temp_file.name
    temp_file.close()  
    
    try:
        file.save(temp_path)
        
        audio_input, sr = librosa.load(temp_path, sr=16000, mono=True)
        
        
        input_features = processor(audio_input, sampling_rate=sr, return_tensors="pt").input_features
        
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return jsonify({"transcription": transcription})
    
    except Exception as e:
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500
    
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_path}: {e}")
@app.route('/save_transcription', methods=['POST'])
def save_transcription():
    data = request.json
    
    if not data or 'transcription' not in data:
        return jsonify({"error": "No transcription data provided"}), 400
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        name = data.get('name', 'Unnamed Transcription')
        filename = data.get('fileName', '')
        transcription = data['transcription']
        file_info = json.dumps({
            'fileType': data.get('fileType', ''),
            'fileSize': data.get('fileSize', 0)
        })
        
        cursor.execute(
            "INSERT INTO transcriptions (name, filename, transcription, file_info) VALUES (?, ?, ?, ?)",
            (name, filename, transcription, file_info)
        )
        
        conn.commit()
        transcription_id = cursor.lastrowid
        conn.close()
        
        return jsonify({"success": True, "id": transcription_id})
    
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route('/get_transcriptions', methods=['GET'])
def get_transcriptions():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM transcriptions ORDER BY created_at DESC")
        rows = cursor.fetchall()
        
        transcriptions = []
        for row in rows:
            transcriptions.append({
                "id": row['id'],
                "name": row['name'],
                "filename": row['filename'],
                "transcription": row['transcription'],
                "file_info": json.loads(row['file_info']),
                "created_at": row['created_at']
            })
        
        conn.close()
        return jsonify({"transcriptions": transcriptions})
    
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route('/delete_transcription/<int:transcription_id>', methods=['DELETE'])
def delete_transcription(transcription_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM transcriptions WHERE id = ?", (transcription_id,))
        conn.commit()
        conn.close()
        
        return jsonify({"success": True})
    
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)