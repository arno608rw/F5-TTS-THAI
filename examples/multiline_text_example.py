#!/usr/bin/env python3
"""
ตัวอย่างการใช้งาน F5-TTS Thai API กับข้อความหลายบรรทัด
Example usage of F5-TTS Thai API with multi-line text
"""

import requests
import json
import base64
import os
from typing import Dict, Any

# API endpoint
API_URL = "http://localhost:4000"

def test_multiline_tts():
    """ทดสอบ TTS กับข้อความหลายบรรทัด"""
    
    # วิธีที่ 1: ใช้ triple quotes
    text_method1 = """สวัสดีครับผม
วันนี้อากาศดีมาก
เหมาะสำหรับไปเดินเล่นที่สวนสาธารณะ
หวังว่าทุกคนจะมีความสุข
ขอบคุณมากครับ"""
    
    # วิธีที่ 2: ใช้ list และ join
    text_lines = [
        "สวัสดีครับผม",
        "วันนี้อากาศดีมาก", 
        "เหมาะสำหรับไปเดินเล่นที่สวนสาธารณะ",
        "หวังว่าทุกคนจะมีความสุข",
        "ขอบคุณมากครับ"
    ]
    text_method2 = "\n".join(text_lines)
    
    # วิธีที่ 3: ใช้ string concatenation
    text_method3 = "สวัสดีครับผม\n" + \
                   "วันนี้อากาศดีมาก\n" + \
                   "เหมาะสำหรับไปเดินเล่นที่สวนสาธารณะ\n" + \
                   "หวังว่าทุกคนจะมีความสุข\n" + \
                   "ขอบคุณมากครับ"
    
    # ทดสอบทั้ง 3 วิธี
    methods = [
        ("Triple Quotes", text_method1),
        ("List Join", text_method2), 
        ("String Concatenation", text_method3)
    ]
    
    for method_name, text in methods:
        print(f"\n=== ทดสอบ {method_name} ===")
        print(f"Text: {repr(text)}")
        
        # ทดสอบ Base64 response
        test_base64_response(text, method_name)
        
        # ทดสอบ File response
        test_file_response(text, method_name)

def test_base64_response(text: str, method_name: str):
    """ทดสอบ Base64 audio response"""
    try:
        response = requests.post(f"{API_URL}/tts", json={
            "profile_name": "thai_speaker_1",  # ใช้ profile ที่มี
            "gen_text": text,
            "return_format": "audio_base64",
            "remove_silence": True,
            "speed": 1.0,
            "cfg_strength": 2.0
        })
        
        if response.status_code == 200:
            data = response.json()
            audio_base64 = data['audio_base64']
            
            # บันทึกไฟล์เสียง
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"{method_name.lower().replace(' ', '_')}_base64.wav"
            filepath = os.path.join(output_dir, filename)
            
            # แปลง base64 เป็นไฟล์
            audio_data = base64.b64decode(audio_base64)
            with open(filepath, 'wb') as f:
                f.write(audio_data)
            
            print(f"✓ Base64 audio saved: {filepath}")
            print(f"  Sample rate: {data['sample_rate']}")
            print(f"  Seed: {data['seed']}")
            
        else:
            print(f"✗ Base64 request failed: {response.status_code}")
            print(f"  Error: {response.text}")
            
    except Exception as e:
        print(f"✗ Base64 request error: {str(e)}")

def test_file_response(text: str, method_name: str):
    """ทดสอบ File audio response"""
    try:
        response = requests.post(f"{API_URL}/tts", json={
            "profile_name": "thai_speaker_1",  # ใช้ profile ที่มี
            "gen_text": text,
            "return_format": "audio_file",
            "remove_silence": True,
            "speed": 1.0,
            "cfg_strength": 2.0
        })
        
        if response.status_code == 200:
            # บันทึกไฟล์เสียง
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"{method_name.lower().replace(' ', '_')}_file.wav"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ File audio saved: {filepath}")
            print(f"  Content-Type: {response.headers.get('Content-Type')}")
            print(f"  File size: {len(response.content)} bytes")
            
        else:
            print(f"✗ File request failed: {response.status_code}")
            print(f"  Error: {response.text}")
            
    except Exception as e:
        print(f"✗ File request error: {str(e)}")

def test_long_text():
    """ทดสอบข้อความยาวหลายพารากราฟ"""
    print("\n=== ทดสอบข้อความยาวหลายพารากราฟ ===")
    
    long_text = """สวัสดีครับผม ผมชื่อ AI Assistant
วันนี้ผมจะมาแนะนำเกี่ยวกับ F5-TTS Thai API

ระบบนี้สามารถแปลงข้อความเป็นเสียงพูดได้
รองรับภาษาไทยได้อย่างดีเยี่ยม
และมีคุณภาพเสียงที่ใสชัด

สำหรับการใช้งาน สามารถส่งข้อความหลายบรรทัดได้
ระบบจะประมวลผลแต่ละบรรทัดอย่างเป็นธรรมชาติ
ทำให้เสียงมีความต่อเนื่องและฟังง่าย

ขอบคุณที่ใช้บริการครับ
หวังว่าจะเป็นประโยชน์สำหรับทุกคน"""
    
    try:
        response = requests.post(f"{API_URL}/tts", json={
            "profile_name": "thai_speaker_1",
            "gen_text": long_text,
            "return_format": "audio_base64",
            "remove_silence": True,
            "speed": 1.0,
            "cfg_strength": 2.0,
            "max_chars": 150  # แบ่งข้อความเป็นช่วงๆ
        })
        
        if response.status_code == 200:
            data = response.json()
            
            # บันทึกไฟล์เสียง
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, "long_text_example.wav")
            
            audio_data = base64.b64decode(data['audio_base64'])
            with open(filepath, 'wb') as f:
                f.write(audio_data)
            
            print(f"✓ Long text audio saved: {filepath}")
            print(f"  Sample rate: {data['sample_rate']}")
            print(f"  Seed: {data['seed']}")
            print(f"  Text length: {len(long_text)} characters")
            
        else:
            print(f"✗ Long text request failed: {response.status_code}")
            print(f"  Error: {response.text}")
            
    except Exception as e:
        print(f"✗ Long text request error: {str(e)}")

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
    print("F5-TTS Thai API - Multi-line Text Example")
    print("=" * 50)
    
    # ตรวจสอบสถานะ API
    if not check_api_health():
        print("Please make sure the API server is running on http://localhost:4000")
        return
    
    # ทดสอบการใช้งาน
    test_multiline_tts()
    test_long_text()
    
    print("\n" + "=" * 50)
    print("All tests completed! Check the './output' directory for generated audio files.")

if __name__ == "__main__":
    main()
