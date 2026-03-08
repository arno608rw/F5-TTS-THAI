# F5-TTS Thai API - Complete Documentation

## 📋 Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Audio Return Formats](#audio-return-formats)
4. [Multistyle Modes](#multistyle-modes)
5. [Profile Management](#profile-management)
6. [API Endpoints](#api-endpoints)
7. [Multi-line Text Support](#multi-line-text-support)
8. [Postman Collection](#postman-collection)
9. [Examples](#examples)
10. [Error Handling](#error-handling)

## 📖 Overview

F5-TTS Thai API เป็น REST API สำหรับการแปลงข้อความเป็นเสียงพูดภาษาไทย รองรับคุณสมบัติต่างๆ ดังนี้:

- **Text-to-Speech**: แปลงข้อความเป็นเสียงพูด
- **Multistyle TTS**: สร้างเสียงหลายสไตล์ในข้อความเดียว
- **Audio Transcription**: แปลงเสียงเป็นข้อความ
- **Profile Management**: จัดการโปรไฟล์เสียงพร้อมระบบ emotion
- **Multiple Input Methods**: รองรับ file path, file upload, และ base64
- **Flexible Output**: ส่งคืนเป็น base64 หรือไฟล์โดยตรง

## 🚀 Quick Start

### 1. Start the API Server
```bash
# Start the server
python src/f5_tts/api_new.py --server

# Custom host and port
python src/f5_tts/api_new.py --server --host 0.0.0.0 --port 5000

# Use FP16 model
python src/f5_tts/api_new.py --server --model FP16
```

### 2. Test Basic TTS
```bash
curl -X POST http://localhost:4000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "profile_name": "thai_speaker_1",
    "gen_text": "สวัสดีครับ วันนี้อากาศดีมาก",
    "return_format": "audio_base64"
  }'
```

## 🎵 Audio Return Formats

API รองรับ 2 รูปแบบการส่งคืนข้อมูลเสียง:

### 1. Base64 Audio (Default)
- **Parameter**: `"return_format": "audio_base64"` (หรือไม่ใส่)
- **Response**: JSON พร้อม `audio_base64` field
- **Use Case**: เหมาะสำหรับ web applications, JavaScript clients

### 2. Audio File
- **Parameter**: `"return_format": "audio_file"`
- **Response**: ดาวน์โหลดไฟล์ WAV โดยตรง
- **Use Case**: เหมาะสำหรับการดาวน์โหลดไฟล์ หรือ streaming

### Example Usage

#### Base64 Response:
```json
{
  "profile_name": "thai_speaker_1",
  "gen_text": "วันนี้อากาศดีมาก เหมาะสำหรับไปเดินเล่นที่สวนสาธารณะ",
  "return_format": "audio_base64"
}
```

**Response:**
```json
{
  "success": true,
  "audio_base64": "UklGRkQGAABXQVZFZm10IBAAAAABAAEA...",
  "sample_rate": 24000,
  "seed": 12345,
  "ref_text": "processed reference text"
}
```

#### File Download:
```bash
curl -X POST http://localhost:4000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "profile_name": "thai_speaker_1",
    "gen_text": "สวัสดีครับ",
    "return_format": "audio_file"
  }' \
  --output generated_audio.wav
```

## 🎭 Multistyle Modes

API รองรับการสร้างเสียงหลายสไตล์ด้วย 3 วิธีการ:

### 1. Direct Mode (วิธีโดยตรง)
ใช้ไฟล์เสียงที่มีอยู่แล้วหรือข้อมูล base64

```json
{
  "mode": "direct",
  "gen_text": "{normal} สวัสดีครับ {sad} ผมเศร้ามาก",
  "speech_types": {
    "normal": {
      "audio": "/path/to/normal/audio.wav",
      "ref_text": "ข้อความอ้างอิงแบบปกติ"
    },
    "sad": {
      "audio": "/path/to/sad/audio.wav",
      "ref_text": "ข้อความอ้างอิงแบบเศร้า"
    }
  },
  "return_format": "audio_base64"
}
```

#### Base64 Audio Support:
```json
{
  "mode": "direct",
  "gen_text": "{normal} สวัสดีครับ {sad} ผมเศร้ามาก",
  "speech_types": {
    "normal": {"ref_text": "ข้อความอ้างอิงแบบปกติ"},
    "sad": {"ref_text": "ข้อความอ้างอิงแบบเศร้า"}
  },
  "audio_base64_normal": "UklGRkQGAABXQVZFZm10...",
  "audio_base64_sad": "UklGRkQGAABXQVZFZm10...",
  "return_format": "audio_base64"
}
```

### 2. File Upload Mode (อัปโหลดไฟล์)
อัปโหลดไฟล์เสียงผ่าน multipart/form-data

```bash
curl -X POST http://localhost:4000/multistyle \
  -F "mode=file_upload" \
  -F "gen_text={normal} สวัสดีครับ {sad} ผมเศร้ามาก" \
  -F "speech_types={\"normal\":{\"ref_text\":\"ข้อความอ้างอิง\"},\"sad\":{\"ref_text\":\"ข้อความเศร้า\"}}" \
  -F "audio_normal=@/path/to/normal.wav" \
  -F "audio_sad=@/path/to/sad.wav" \
  -F "return_format=audio_base64"
```

### 3. Profile Mode (โหมดโปรไฟล์)
ใช้โปรไฟล์ที่บันทึกไว้แล้ว พร้อมระบบ emotion

```json
{
  "mode": "profile",
  "gen_text": "{normal} สวัสดีครับ {happy} วันนี้ผมมีความสุขมาก {sad} แต่บางทีก็เศร้า",
  "profile_emotions": {
    "normal": "normal",
    "happy": "happy", 
    "sad": "sad"
  },
  "return_format": "audio_base64",
  "return_spectrogram": true
}
```

#### Profile by Name:
```json
{
  "mode": "profile",
  "gen_text": "{speaker1} สวัสดีครับ {speaker2} สวัสดีค่ะ",
  "profile_emotions": {
    "speaker1": "thai_speaker_male",
    "speaker2": "thai_speaker_female"
  }
}
```

## 👤 Profile Management

### Profile Structure with Emotions
```json
{
  "thai_speaker_happy": {
    "audio_path": "./profiles/thai_speaker_happy.wav",
    "ref_text": "ข้อความอ้างอิงแบบมีความสุข",
    "emotion": "happy",
    "description": "Thai speaker with happy emotion",
    "created_at": "2025-07-13T15:07:38.984980"
  }
}
```

### Supported Emotions
- `normal` - เสียงปกติ
- `happy` - เสียงมีความสุข
- `sad` - เสียงเศร้า
- `angry` - เสียงโกรธ
- `calm` - เสียงสงบ
- `excited` - เสียงตื่นเต้น
- `whisper` - เสียงกระซิบ
- `loud` - เสียงดัง

### Create Profile with Emotion
```bash
curl -X POST http://localhost:4000/profiles \
  -H "Content-Type: application/json" \
  -d '{
    "profile_name": "thai_speaker_happy",
    "ref_audio_path": "/path/to/happy/audio.wav",
    "ref_text": "ข้อความอ้างอิงแบบมีความสุข",
    "description": "Thai speaker with happy emotion",
    "emotion": "happy"
  }'
```

### Upload Profile
```bash
curl -X POST http://localhost:4000/profiles \
  -F "profile_name=thai_speaker_calm" \
  -F "ref_audio=@/path/to/calm.wav" \
  -F "ref_text=ข้อความอ้างอิงแบบสงบ" \
  -F "description=Thai speaker with calm voice" \
  -F "emotion=calm"
```

## 🔗 API Endpoints

### Health & Info
- **GET /health** - Check API server status
- **GET /info** - Get model information

### Text-to-Speech
- **POST /tts** - Generate speech (supports profile, file upload, base64)
- **POST /multistyle** - Generate speech with multiple styles (3 modes)

### Audio Transcription
- **POST /transcribe** - Transcribe audio to text (file path or upload)

### Profile Management
- **GET /profiles** - List all profiles
- **GET /profiles/{name}** - Get specific profile info
- **GET /profiles/emotions** - Get profiles grouped by emotion
- **GET /profiles/emotions/{emotion}** - Get profiles filtered by emotion
- **POST /profiles** - Create new profile (file path or upload)
- **DELETE /profiles/{name}** - Delete profile

## 📝 Multi-line Text Support

API รองรับข้อความหลายบรรทัดด้วยวิธีต่างๆ:

### JSON API Call
```json
{
  "profile_name": "thai_speaker_1",
  "gen_text": "สวัสดีครับ\nวันนี้อากาศดีมาก\nเหมาะสำหรับไปเดินเล่นที่สวนสาธารณะ",
  "return_format": "audio_base64"
}
```

### Python
```python
# วิธีที่ 1: Triple quotes
text = """สวัสดีครับ
วันนี้อากาศดีมาก
เหมาะสำหรับไปเดินเล่นที่สวนสาธารณะ"""

# วิธีที่ 2: List join
text = "\n".join([
    "สวัสดีครับ",
    "วันนี้อากาศดีมาก", 
    "เหมาะสำหรับไปเดินเล่นที่สวนสาธารณะ"
])

# วิธีที่ 3: String concatenation
text = "สวัสดีครับ\n" + \
       "วันนี้อากาศดีมาก\n" + \
       "เหมาะสำหรับไปเดินเล่นที่สวนสาธารณะ"
```

### JavaScript
```javascript
// วิธีที่ 1: Template literals
const text = `สวัสดีครับ
วันนี้อากาศดีมาก
เหมาะสำหรับไปเดินเล่นที่สวนสาธารณะ`;

// วิธีที่ 2: Array join
const text = [
    'สวัสดีครับ',
    'วันนี้อากาศดีมาก',
    'เหมาะสำหรับไปเดินเล่นที่สวนสาธารณะ'
].join('\n');
```

### cURL
```bash
curl -X POST http://localhost:4000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "profile_name": "thai_speaker_1",
    "gen_text": "สวัสดีครับ\nวันนี้อากาศดีมาก\nเหมาะสำหรับไปเดินเล่นที่สวนสาธารณะ",
    "return_format": "audio_base64"
  }'
```

## 📮 Postman Collection

### Import Collection
1. Open Postman
2. Click "Import" button
3. Select `F5-TTS-Thai-API.postman_collection.json`
4. Import the environment file `F5-TTS-Thai-API.postman_environment.json`
5. Select the "F5-TTS Thai API Environment" in the top-right corner

### Available Requests

#### 🔍 Health & Info
- **Health Check** - `GET /health`
- **Model Info** - `GET /info`

#### 🎤 Text-to-Speech
- **TTS with Profile** - `POST /tts`
- **TTS with File Upload** - `POST /tts` (multipart/form-data)
- **TTS with Base64 Audio** - `POST /tts`

#### 🎭 Multistyle
- **Multistyle Direct** - `POST /multistyle`
- **Multistyle File Upload** - `POST /multistyle`
- **Multistyle Profile** - `POST /multistyle`

#### 📝 Transcription
- **Transcribe Audio** - `POST /transcribe`
- **Transcribe File Upload** - `POST /transcribe`

#### 👤 Profile Management
- **List Profiles** - `GET /profiles`
- **Get Profile Info** - `GET /profiles/{name}`
- **Get Profiles by Emotion** - `GET /profiles/emotions`
- **Get Specific Emotion** - `GET /profiles/emotions/{emotion}`
- **Create Profile** - `POST /profiles`
- **Create Profile with Upload** - `POST /profiles`
- **Delete Profile** - `DELETE /profiles/{name}`

### Audio Input Methods

#### 1. File Path (JSON)
```json
{
  "ref_audio_path": "./path/to/audio.wav",
  "ref_text": "reference text"
}
```

#### 2. File Upload (Form Data)
```
ref_audio: [file upload]
ref_text: reference text
```

#### 3. Base64 Audio (JSON)
```json
{
  "ref_audio": "UklGRkQGAABXQVZFZm10...",
  "ref_text": "reference text"
}
```

## 💡 Examples

### Python Examples

#### Basic TTS
```python
import requests
import base64

response = requests.post('http://localhost:4000/tts', json={
    'profile_name': 'thai_speaker_1',
    'gen_text': 'สวัสดีครับ วันนี้อากาศดีมาก',
    'return_format': 'audio_base64'
})

if response.status_code == 200:
    data = response.json()
    audio_data = base64.b64decode(data['audio_base64'])
    with open('output.wav', 'wb') as f:
        f.write(audio_data)
```

#### Multistyle with Profile
```python
import requests

response = requests.post('http://localhost:4000/multistyle', json={
    'mode': 'profile',
    'gen_text': '{normal} สวัสดีครับ {happy} วันนี้ผมมีความสุขมาก {sad} แต่บางทีก็เศร้า',
    'profile_emotions': {
        'normal': 'normal',
        'happy': 'happy',
        'sad': 'sad'
    },
    'return_format': 'audio_base64',
    'return_spectrogram': True
})

if response.status_code == 200:
    data = response.json()
    # Save audio
    audio_data = base64.b64decode(data['audio_base64'])
    with open('multistyle_output.wav', 'wb') as f:
        f.write(audio_data)
    
    # Save spectrograms
    if 'spectrograms_base64' in data:
        for i, spec_base64 in enumerate(data['spectrograms_base64']):
            spec_data = base64.b64decode(spec_base64)
            with open(f'spectrogram_{i+1}.png', 'wb') as f:
                f.write(spec_data)
```

#### File Upload
```python
import requests

files = {
    'ref_audio': open('/path/to/audio.wav', 'rb')
}

data = {
    'profile_name': 'new_speaker',
    'ref_text': 'ข้อความอ้างอิง',
    'emotion': 'happy',
    'description': 'Happy speaker'
}

response = requests.post('http://localhost:4000/profiles', files=files, data=data)
```

### JavaScript Examples

#### Web Audio Playback
```javascript
const response = await fetch('/tts', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    profile_name: 'thai_speaker_1',
    gen_text: 'สวัสดีครับ วันนี้อากาศดีมาก',
    return_format: 'audio_base64'
  })
});

const data = await response.json();
const audioData = `data:audio/wav;base64,${data.audio_base64}`;
const audio = new Audio(audioData);
audio.play();
```

#### File Download
```javascript
const response = await fetch('/tts', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    profile_name: 'thai_speaker_1',
    gen_text: 'สวัสดีครับ',
    return_format: 'audio_file'
  })
});

const blob = await response.blob();
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = 'generated_audio.wav';
a.click();
```

## ⚠️ Error Handling

### Common Error Responses
```json
{
  "success": false,
  "error": "Profile 'thai_speaker_1' not found"
}
```

### HTTP Status Codes
- `200` - Success
- `400` - Bad Request (invalid parameters)
- `404` - Not Found (profile not found)
- `413` - Payload Too Large (file too big)
- `500` - Internal Server Error

### Best Practices
1. **Always check response status**: `response.status_code == 200`
2. **Handle errors gracefully**: Check for `success` field in response
3. **File size limits**: Max file size is 16MB
4. **Audio format**: Use WAV files for best results
5. **Profile management**: Create profiles before using them
6. **Memory management**: Use `audio_file` format for large files

## 🛠️ Advanced Features

### Spectrogram Generation
```json
{
  "profile_name": "thai_speaker_1",
  "gen_text": "สวัสดีครับ",
  "return_spectrogram": true
}
```

### Long Text Handling
```json
{
  "profile_name": "thai_speaker_1",
  "gen_text": "ข้อความยาวมาก...",
  "max_chars": 150,
  "remove_silence": true
}
```

### Voice Cloning Parameters
```json
{
  "profile_name": "thai_speaker_1",
  "gen_text": "สวัสดีครับ",
  "speed": 1.2,
  "cfg_strength": 2.5,
  "nfe_step": 32
}
```

## 📚 Additional Resources

### Configuration Files
- `F5-TTS-Thai-API.postman_collection.json` - Postman collection
- `F5-TTS-Thai-API.postman_environment.json` - Environment variables
- `examples/multiline_text_example.py` - Multi-line text examples
- `examples/multistyle_modes_example.py` - Multistyle examples

### Server Options
```bash
# Debug mode
python src/f5_tts/api_new.py --server --debug

# Custom model
python src/f5_tts/api_new.py --server --model Custom --custom-model /path/to/model.pt

# Help
python src/f5_tts/api_new.py --help
```

---

**F5-TTS Thai API** - Complete text-to-speech solution with multistyle support, profile management, and flexible audio input/output options.
