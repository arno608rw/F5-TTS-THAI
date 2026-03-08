#!/usr/bin/env python3
"""
ตัวอย่างการใช้งาน F5-TTS Thai API - Multistyle Modes
Example usage of F5-TTS Thai API - Multistyle with Direct, File Upload, and Profile modes
"""

import requests
import json
import base64
import os
import tempfile
from typing import Dict, Any

# API endpoint
API_URL = "http://localhost:4000"

def setup_demo_profiles():
    """สร้างโปรไฟล์ตัวอย่างสำหรับการทดสอบ"""
    print("=== Setting up demo profiles ===")
    
    # สร้างไฟล์เสียงตัวอย่างจำลอง (ในการใช้งานจริงจะใช้ไฟล์เสียงจริง)
    demo_profiles = [
        {
            "profile_name": "thai_speaker_normal",
            "ref_text": "สวัสดีครับ ผมเป็นคนไทย",
            "emotion": "normal",
            "description": "Thai speaker with normal emotion"
        },
        {
            "profile_name": "thai_speaker_happy",
            "ref_text": "วันนี้ผมมีความสุขมาก",
            "emotion": "happy",
            "description": "Thai speaker with happy emotion"
        },
        {
            "profile_name": "thai_speaker_sad",
            "ref_text": "ผมรู้สึกเศร้าใจ",
            "emotion": "sad",
            "description": "Thai speaker with sad emotion"
        }
    ]
    
    for profile in demo_profiles:
        try:
            # ในการใช้งานจริง จะใช้ไฟล์เสียงจริง
            # สำหรับ demo นี้ จะสร้างโปรไฟล์โดยใช้ไฟล์ตัวอย่างที่มีอยู่
            response = requests.post(f"{API_URL}/profiles", json={
                "profile_name": profile["profile_name"],
                "ref_audio_path": "./src/f5_tts/infer/examples/thai_examples/ref_gen_1.wav",  # ไฟล์ตัวอย่าง
                "ref_text": profile["ref_text"],
                "emotion": profile["emotion"],
                "description": profile["description"],
                "overwrite": True
            })
            
            if response.status_code == 200:
                print(f"✓ Created profile: {profile['profile_name']} ({profile['emotion']})")
            else:
                print(f"✗ Failed to create profile: {profile['profile_name']} - {response.text}")
                
        except Exception as e:
            print(f"✗ Error creating profile {profile['profile_name']}: {e}")

def test_direct_mode():
    """ทดสอบ Direct Mode"""
    print("\n=== Testing Direct Mode ===")
    
    try:
        response = requests.post(f"{API_URL}/multistyle", json={
            "mode": "direct",
            "gen_text": "{normal} สวัสดีครับ {sad} ผมเศร้ามาก {happy} แต่ตอนนี้ดีขึ้นแล้ว",
            "speech_types": {
                "normal": {
                    "audio": "./src/f5_tts/infer/examples/thai_examples/ref_gen_1.wav",
                    "ref_text": "ข้อความอ้างอิงแบบปกติ"
                },
                "sad": {
                    "audio": "./src/f5_tts/infer/examples/thai_examples/ref_gen_1.wav",
                    "ref_text": "ข้อความอ้างอิงแบบเศร้า"
                },
                "happy": {
                    "audio": "./src/f5_tts/infer/examples/thai_examples/ref_gen_2.wav",
                    "ref_text": "ข้อความอ้างอิงแบบมีความสุข"
                }
            },
            "return_format": "audio_base64",
            "return_spectrogram": True
        })
        
        if response.status_code == 200:
            data = response.json()
            
            # บันทึกไฟล์เสียง
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            
            audio_data = base64.b64decode(data['audio_base64'])
            with open(os.path.join(output_dir, "multistyle_direct.wav"), 'wb') as f:
                f.write(audio_data)
            
            print(f"✓ Direct mode successful")
            print(f"  Sample rate: {data['sample_rate']}")
            print(f"  Seed: {data['seed']}")
            print(f"  Audio saved to: ./output/multistyle_direct.wav")
            
            # บันทึก spectrograms
            if 'spectrograms_base64' in data:
                for i, spec_base64 in enumerate(data['spectrograms_base64']):
                    spec_data = base64.b64decode(spec_base64)
                    with open(os.path.join(output_dir, f"spectrogram_direct_{i+1}.png"), 'wb') as f:
                        f.write(spec_data)
                print(f"  Spectrograms saved: {len(data['spectrograms_base64'])} files")
            
        else:
            print(f"✗ Direct mode failed: {response.status_code}")
            print(f"  Error: {response.text}")
            
    except Exception as e:
        print(f"✗ Direct mode error: {e}")

def test_file_upload_mode():
    """ทดสอบ File Upload Mode"""
    print("\n=== Testing File Upload Mode ===")
    
    try:
        # เตรียมไฟล์สำหรับอัปโหลด
        files = {}
        audio_files = [
            ("audio_normal", "./src/f5_tts/infer/examples/thai_examples/ref_gen_1.wav"),
            ("audio_sad", "./src/f5_tts/infer/examples/thai_examples/ref_gen_1.wav"),
            ("audio_happy", "./src/f5_tts/infer/examples/thai_examples/ref_gen_2.wav")
        ]
        
        for field_name, file_path in audio_files:
            if os.path.exists(file_path):
                files[field_name] = open(file_path, 'rb')
        
        if not files:
            print("✗ No audio files found for upload test")
            return
        
        data = {
            "mode": "file_upload",
            "gen_text": "{normal} สวัสดีครับ {sad} ผมเศร้ามาก {happy} แต่ตอนนี้ดีขึ้นแล้ว",
            "speech_types": json.dumps({
                "normal": {"ref_text": "ข้อความอ้างอิงแบบปกติ"},
                "sad": {"ref_text": "ข้อความอ้างอิงแบบเศร้า"},
                "happy": {"ref_text": "ข้อความอ้างอิงแบบมีความสุข"}
            }),
            "return_format": "audio_base64"
        }
        
        response = requests.post(f"{API_URL}/multistyle", files=files, data=data)
        
        # ปิดไฟล์
        for f in files.values():
            f.close()
        
        if response.status_code == 200:
            data = response.json()
            
            # บันทึกไฟล์เสียง
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            
            audio_data = base64.b64decode(data['audio_base64'])
            with open(os.path.join(output_dir, "multistyle_upload.wav"), 'wb') as f:
                f.write(audio_data)
            
            print(f"✓ File upload mode successful")
            print(f"  Sample rate: {data['sample_rate']}")
            print(f"  Seed: {data['seed']}")
            print(f"  Audio saved to: ./output/multistyle_upload.wav")
            
        else:
            print(f"✗ File upload mode failed: {response.status_code}")
            print(f"  Error: {response.text}")
            
    except Exception as e:
        print(f"✗ File upload mode error: {e}")

def test_profile_mode():
    """ทดสอบ Profile Mode"""
    print("\n=== Testing Profile Mode ===")
    
    try:
        # ทดสอบโดยใช้ emotion
        response = requests.post(f"{API_URL}/multistyle", json={
            "mode": "profile",
            "gen_text": "{normal} สวัสดีครับ {happy} วันนี้ผมมีความสุขมาก {sad} แต่บางทีก็เศร้า",
            "profile_emotions": {
                "normal": "normal",
                "happy": "happy",
                "sad": "sad"
            },
            "return_format": "audio_base64",
            "return_spectrogram": True
        })
        
        if response.status_code == 200:
            data = response.json()
            
            # บันทึกไฟล์เสียง
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            
            audio_data = base64.b64decode(data['audio_base64'])
            with open(os.path.join(output_dir, "multistyle_profile.wav"), 'wb') as f:
                f.write(audio_data)
            
            print(f"✓ Profile mode successful")
            print(f"  Sample rate: {data['sample_rate']}")
            print(f"  Seed: {data['seed']}")
            print(f"  Audio saved to: ./output/multistyle_profile.wav")
            
            # บันทึก spectrograms
            if 'spectrograms_base64' in data:
                for i, spec_base64 in enumerate(data['spectrograms_base64']):
                    spec_data = base64.b64decode(spec_base64)
                    with open(os.path.join(output_dir, f"spectrogram_profile_{i+1}.png"), 'wb') as f:
                        f.write(spec_data)
                print(f"  Spectrograms saved: {len(data['spectrograms_base64'])} files")
            
        else:
            print(f"✗ Profile mode failed: {response.status_code}")
            print(f"  Error: {response.text}")
            
    except Exception as e:
        print(f"✗ Profile mode error: {e}")

def test_profile_by_name():
    """ทดสอบ Profile Mode โดยใช้ชื่อโปรไฟล์"""
    print("\n=== Testing Profile Mode by Name ===")
    
    try:
        response = requests.post(f"{API_URL}/multistyle", json={
            "mode": "profile",
            "gen_text": "{speaker1} สวัสดีครับ {speaker2} วันนี้ผมมีความสุขมาก",
            "profile_emotions": {
                "speaker1": "thai_speaker_normal",
                "speaker2": "thai_speaker_happy"
            },
            "return_format": "audio_base64"
        })
        
        if response.status_code == 200:
            data = response.json()
            
            # บันทึกไฟล์เสียง
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            
            audio_data = base64.b64decode(data['audio_base64'])
            with open(os.path.join(output_dir, "multistyle_profile_by_name.wav"), 'wb') as f:
                f.write(audio_data)
            
            print(f"✓ Profile by name mode successful")
            print(f"  Sample rate: {data['sample_rate']}")
            print(f"  Seed: {data['seed']}")
            print(f"  Audio saved to: ./output/multistyle_profile_by_name.wav")
            
        else:
            print(f"✗ Profile by name mode failed: {response.status_code}")
            print(f"  Error: {response.text}")
            
    except Exception as e:
        print(f"✗ Profile by name mode error: {e}")

def test_emotion_endpoints():
    """ทดสอบ Emotion Management Endpoints"""
    print("\n=== Testing Emotion Management ===")
    
    try:
        # ดูโปรไฟล์ทั้งหมด
        response = requests.get(f"{API_URL}/profiles")
        if response.status_code == 200:
            profiles = response.json()['profiles']
            print(f"✓ Available profiles: {profiles}")
        
        # ดูโปรไฟล์ตาม emotion
        response = requests.get(f"{API_URL}/profiles/emotions")
        if response.status_code == 200:
            emotions = response.json()['emotions']
            print(f"✓ Profiles by emotion:")
            for emotion, profile_list in emotions.items():
                print(f"  {emotion}: {profile_list}")
        
        # ดูโปรไฟล์ emotion เฉพาะ
        response = requests.get(f"{API_URL}/profiles/emotions/happy")
        if response.status_code == 200:
            happy_profiles = response.json()['profiles']
            print(f"✓ Happy profiles: {happy_profiles}")
        
    except Exception as e:
        print(f"✗ Emotion management error: {e}")

def check_api_health():
    """ตรวจสอบสถานะ API"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("API Status:", data)
            return True
        else:
            print(f"API Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Cannot connect to API: {str(e)}")
        return False

def main():
    """ฟังก์ชันหลัก"""
    print("F5-TTS Thai API - Multistyle Modes Example")
    print("=" * 60)
    
    # ตรวจสอบสถานะ API
    if not check_api_health():
        print("Please make sure the API server is running on http://localhost:4000")
        return
    
    # ทดสอบการใช้งาน
    setup_demo_profiles()
    test_direct_mode()
    test_file_upload_mode()
    test_profile_mode()
    test_profile_by_name()
    test_emotion_endpoints()
    
    print("\n" + "=" * 60)
    print("All tests completed! Check the './output' directory for generated files.")
    print("\nGenerated files:")
    print("- multistyle_direct.wav")
    print("- multistyle_upload.wav")
    print("- multistyle_profile.wav")
    print("- multistyle_profile_by_name.wav")
    print("- spectrogram_*.png")

if __name__ == "__main__":
    main()
