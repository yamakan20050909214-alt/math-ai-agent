from google import genai

import os
import streamlit as st

API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

client = genai.Client(api_key=API_KEY)

response = client.models.generate_content(
    model='gemini-3.6-flash',
    contents='「接続成功です」と返答してください。'
)

print("応答結果:", response.text)