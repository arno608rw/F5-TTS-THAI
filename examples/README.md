# F5-TTS Thai API Examples

โฟลเดอร์นี้ประกอบด้วยตัวอย่างการใช้งาน F5-TTS Thai API

## ไฟล์ตัวอย่าง

### 1. multiline_text_example.py
ตัวอย่างการใช้งาน API กับข้อความหลายบรรทัด

**คุณสมบัติ:**
- ทดสอบ 3 วิธีในการสร้างข้อความหลายบรรทัด
- ทดสอบทั้ง Base64 และ File response format
- ทดสอบข้อความยาวหลายพารากราฟ
- ตรวจสอบสถานะ API

### 2. multistyle_modes_example.py
ตัวอย่างการใช้งาน Multistyle API ทั้ง 3 โหมด

**คุณสมบัติ:**
- ทดสอบ Direct Mode (ไฟล์โดยตรง)
- ทดสอบ File Upload Mode (อัปโหลดไฟล์)
- ทดสอบ Profile Mode (ใช้โปรไฟล์)
- การจัดการ emotions ในโปรไฟล์
- ตัวอย่างการใช้งาน spectrogram

**การใช้งาน:**
```bash
# ติดตั้ง requirements
pip install requests

# รัน API server ก่อน
python src/f5_tts/api_new.py --server

# รันตัวอย่างข้อความหลายบรรทัด
python examples/multiline_text_example.py

# รันตัวอย่าง multistyle modes
python examples/multistyle_modes_example.py
```

**ผลลัพธ์:**
- ไฟล์เสียงจะถูกบันทึกใน `./output/`
- แต่ละวิธีจะสร้างไฟล์ทั้งแบบ Base64 และ File format
- Multistyle จะสร้างไฟล์เสียงและ spectrogram สำหรับแต่ละโหมด

## Multistyle Modes

### 1. Direct Mode
ใช้ไฟล์เสียงที่มีอยู่แล้วหรือ base64 data

```json
{
  "mode": "direct",
  "gen_text": "{normal} สวัสดีครับ {sad} ผมเศร้ามาก",
  "speech_types": {
    "normal": {
      "audio": "/path/to/normal.wav",
      "ref_text": "ข้อความอ้างอิง"
    },
    "sad": {
      "audio": "/path/to/sad.wav",
      "ref_text": "ข้อความเศร้า"
    }
  }
}
```

### 2. File Upload Mode
อัปโหลดไฟล์เสียงผ่าน multipart/form-data

```bash
curl -X POST http://localhost:4000/multistyle \
  -F "mode=file_upload" \
  -F "gen_text={normal} สวัสดี {sad} เศร้า" \
  -F "speech_types={\"normal\":{\"ref_text\":\"ข้อความ\"}}" \
  -F "audio_normal=@normal.wav" \
  -F "audio_sad=@sad.wav"
```

### 3. Profile Mode
ใช้โปรไฟล์ที่บันทึกไว้ พร้อมระบบ emotion

```json
{
  "mode": "profile",
  "gen_text": "{normal} สวัสดี {happy} มีความสุข",
  "profile_emotions": {
    "normal": "normal",
    "happy": "happy"
  }
}
```

## Profile Management with Emotions

### สร้าง Profile พร้อม Emotion
```python
response = requests.post('http://localhost:4000/profiles', json={
    'profile_name': 'thai_speaker_happy',
    'ref_audio_path': '/path/to/happy.wav',
    'ref_text': 'ข้อความมีความสุข',
    'emotion': 'happy',
    'description': 'Thai speaker with happy emotion'
})
```

### ดูโปรไฟล์ตาม Emotion
```python
# ดูโปรไฟล์ทั้งหมดแยกตาม emotion
response = requests.get('http://localhost:4000/profiles/emotions')

# ดูโปรไฟล์ emotion เฉพาะ
response = requests.get('http://localhost:4000/profiles/emotions/happy')
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

## วิธีการใช้งานข้อความหลายบรรทัด
```json
{
  "profile_name": "thai_speaker_1",
  "gen_text": "สวัสดีครับ\nวันนี้อากาศดีมาก\nเหมาะสำหรับไปเดินเล่น",
  "return_format": "audio_base64"
}
```

### 2. Python
```python
# วิธีที่ 1: Triple quotes
text = """สวัสดีครับ
วันนี้อากาศดีมาก
เหมาะสำหรับไปเดินเล่น"""

# วิธีที่ 2: List join
text = "\n".join([
    "สวัสดีครับ",
    "วันนี้อากาศดีมาก", 
    "เหมาะสำหรับไปเดินเล่น"
])

# วิธีที่ 3: String concatenation
text = "สวัสดีครับ\n" + \
       "วันนี้อากาศดีมาก\n" + \
       "เหมาะสำหรับไปเดินเล่น"
```

### 3. JavaScript
```javascript
// วิธีที่ 1: Template literals
const text = `สวัสดีครับ
วันนี้อากาศดีมาก
เหมาะสำหรับไปเดินเล่น`;

// วิธีที่ 2: Array join
const text = [
    'สวัสดีครับ',
    'วันนี้อากาศดีมาก',
    'เหมาะสำหรับไปเดินเล่น'
].join('\n');
```

### 4. cURL
```bash
curl -X POST http://localhost:4000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "profile_name": "thai_speaker_1",
    "gen_text": "สวัสดีครับ\nวันนี้อากาศดีมาก\nเหมาะสำหรับไปเดินเล่น",
    "return_format": "audio_base64"
  }'
```

## ข้อควรระวัง

1. **การจัดการขึ้นบรรทัด**: ใช้ `\n` เป็นตัวขึ้นบรรทัดใหม่
2. **ความยาวข้อความ**: ใช้ `max_chars` เพื่อแบ่งข้อความยาวเป็นส่วนๆ
3. **การลบเสียงเงียb**: ใช้ `remove_silence=true` เพื่อลดช่วงเงียบ
4. **Profile**: ต้องสร้าง profile ก่อนใช้งาน

## การรัน API Server

```bash
# รัน server
python src/f5_tts/api_new.py --server

# รัน server บน port อื่น
python src/f5_tts/api_new.py --server --port 5000

# รัน server กับ model FP16
python src/f5_tts/api_new.py --server --model FP16
```

## การสร้าง Profile

```python
import requests

# สร้าง profile ใหม่
response = requests.post('http://localhost:4000/profiles', json={
    'profile_name': 'thai_speaker_1',
    'ref_audio_path': './path/to/reference/audio.wav',
    'ref_text': 'ข้อความอ้างอิงที่ตรงกับไฟล์เสียง',
    'description': 'Thai speaker with clear voice'
})

print(response.json())
```
