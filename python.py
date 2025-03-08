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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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
        source_type TEXT,
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
    
    model = model.to(device)
    
    if device == "cuda":
        model = model.half()  
        
    MODEL_LOADED = True
    print(f"Whisper model loaded successfully on {device}")
    
    if device == "cuda":
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
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
        
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        try:
            audio_input, sr = librosa.load(temp_path, sr=16000, mono=True)
        except Exception as e:
            print(f"Standard loading failed: {e}, trying alternative method for webm")
            import soundfile as sf
            import io
            import subprocess
            
            temp_wav = temp_path + ".wav"
            subprocess.run(['ffmpeg', '-i', temp_path, '-ar', '16000', '-ac', '1', temp_wav], 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            audio_input, sr = librosa.load(temp_wav, sr=16000, mono=True)
            
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
        
        input_features = processor(audio_input, sampling_rate=sr, return_tensors="pt").input_features
        input_features = input_features.to(device)
        
        generation_config = {
            "do_sample": False,  # Deterministic generation (no sampling)
            "num_beams": 5,  # Beam search
            "max_new_tokens": 256,  # Limit output length
            "return_dict_in_generate": False,
        }
        

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            with torch.no_grad():
                predicted_ids = model.generate(input_features, **generation_config)
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Clean up GPU memory if using CUDA
        if device == "cuda":
            torch.cuda.empty_cache()
        
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
        source_type = data.get('sourceType', 'Uploaded File')
        file_info = json.dumps({
            'fileType': data.get('fileType', ''),
            'fileSize': data.get('fileSize', 0)
        })
        
        cursor.execute(
            "INSERT INTO transcriptions (name, filename, transcription, file_info, source_type) VALUES (?, ?, ?, ?, ?)",
            (name, filename, transcription, file_info, source_type)
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
                "source_type": row.get('source_type', 'Uploaded File'),
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

# Add a route to get GPU status
@app.route('/gpu_status', methods=['GET'])
def gpu_status():
    if torch.cuda.is_available():
        return jsonify({
            "gpu_available": True,
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "memory_allocated_mb": f"{torch.cuda.memory_allocated(0) / 1024**2:.2f}",
            "memory_reserved_mb": f"{torch.cuda.memory_reserved(0) / 1024**2:.2f}"
        })
    else:
        return jsonify({
            "gpu_available": False,
            "device": "cpu"
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)