import os
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from google import genai
from google.genai import types
from chatbot_config import SYSTEM_PROMPT

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured in the .env file.")

client = genai.Client(api_key=API_KEY)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"reply": "Please enter a message."}), 400

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=600,
            ),
        )

        reply = response.text.strip() if response.text else (
            "I couldn't generate a response right now."
        )
        return jsonify({"reply": reply})

    except Exception:
        return jsonify({
            "reply": "Sorry, I’m unable to process your request right now."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
