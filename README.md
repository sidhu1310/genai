# Speech Transcription API with Whisper

This project is a Flask-based web API that enables speech-to-text transcription using OpenAI's Whisper model. The application supports file uploads, processes audio files, and returns transcriptions. It also includes database storage for saving and retrieving transcriptions.

---

## Features
- **Supports Audio Transcription** using Whisper Large-v3 model
- **GPU Acceleration** for faster inference (if available)
- **File Upload Support** for different audio formats
- **Database Storage** for transcriptions
- **Web API Endpoints** for retrieving and managing transcriptions
- **FFmpeg Integration** for handling WebM files

---

## Environment Setup

### 1. Install Dependencies
First, ensure you have Python 3.8 or later installed. Then, install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg (for WebM support)
If you expect to process WebM audio files, install FFmpeg:

#### **Ubuntu/Linux**:
```bash
sudo apt update && sudo apt install ffmpeg
```

#### **Mac (using Homebrew)**:
```bash
brew install ffmpeg
```

#### **Windows**:
Download and install FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add it to your system path.

---

## Running the Application

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

2. Initialize the database:
```bash
python -c "from app import init_db; init_db()"
```

3. Run the Flask app:
```bash
python app.py
```

4. Access the API at:
```
http://127.0.0.1:8000/
```

---

## API Endpoints

### 1. **Transcribe Audio**
- **Endpoint:** `/transcribe`
- **Method:** `POST`
- **Payload:** `Multipart Form-Data` (Upload an audio file)
- **Response:** JSON with transcription

#### **Example Request (cURL)**:
```bash
curl -X POST -F "file=@audio.mp3" http://127.0.0.1:8000/transcribe
```

### 2. **Save Transcription**
- **Endpoint:** `/save_transcription`
- **Method:** `POST`
- **Payload:** JSON `{ "name": "Test", "fileName": "audio.mp3", "transcription": "Hello world" }`

### 3. **Get All Transcriptions**
- **Endpoint:** `/get_transcriptions`
- **Method:** `GET`
- **Response:** JSON array of transcriptions

### 4. **Delete Transcription**
- **Endpoint:** `/delete_transcription/<id>`
- **Method:** `DELETE`

### 5. **Check GPU Status**
- **Endpoint:** `/gpu_status`
- **Method:** `GET`

---

## Deploying the Application

### **Using Docker**
To run the project inside a Docker container:
```bash
docker build -t whisper-transcription .
docker run -p 8000:8000 whisper-transcription
```

### **Deploy on a Cloud Server**
For production, consider using **Gunicorn** with **NGINX**:
```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

---

## Troubleshooting

1. **Whisper Model Loading Issues**:
   - Ensure you have sufficient RAM and disk space.
   - Use a smaller Whisper model (`whisper-base`, `whisper-medium`) if needed.

2. **CUDA Issues**:
   - Run `torch.cuda.is_available()` in Python to check GPU compatibility.
   - If running on GPU but facing errors, reinstall PyTorch with CUDA support:  
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

3. **Database Not Found**:
   - Ensure the `transcriptions.db` file exists.
   - Reinitialize using: `python -c "from app import init_db; init_db()"`

---

## License
This project is open-source under the MIT License.

---

## Contributors
- **K.Sidhartha Rao** - Developer
- **S.Srinija** - Developer

For suggestions or issues, feel free to raise a GitHub issue!

