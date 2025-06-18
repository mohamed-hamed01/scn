from flask import Flask, request, jsonify
from transformers import pipeline
import pytesseract
import cv2
import os
import tempfile

app = Flask(__name__)

# تحميل نموذج التوليد من Hugging Face
text_generator = pipeline("text-generation", model="distilgpt2")

# توصيات بسيطة
all_rooms = ["1011", "Virtual Lab", "CCE Office", "محاضرة 1", "CCE Timetable"]

@app.route('/describe_path', methods=['POST'])
def describe_path():
    room_name = request.form.get('room_name', '')

    # توليد وصف باستخدام النص المُستخرج أو مجرد الاسم (للتجربة فقط)
    prompt = f"Describe the path to {room_name} inside a university building."
    description = text_generator(prompt, max_length=50, do_sample=True)[0]['generated_text']

    return jsonify({"description": description.strip()})


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    history = data.get("history", [])

    # فلترة الأماكن الموصى بها
    recommendations = [room for room in all_rooms if room not in history][:5]

    return jsonify({"recommendations": recommendations})


if __name__ == '__main__':
    app.run(debug=True)
