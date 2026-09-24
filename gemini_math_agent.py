import re
import time
import warnings
from google import genai
from google.genai import errors
import sympy as sp

# 警告メッセージを非表示
warnings.filterwarnings("ignore")

import os
import streamlit as st

API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=API_KEY)

def generate_with_retry(prompt: str, max_retries: int = 3):
    """503エラー（混雑）発生時に自動で再試行する関数"""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt
            )
        except errors.ServerError:
            if attempt < max_retries - 1:
                print(f"  ⚠️ サーバー混雑を検知。3秒後に自動再試行します... ({attempt + 1}/{max_retries})")
                time.sleep(3)
            else:
                raise

def process_math_question(user_question: str):
    print("1. 問題文から方程式を抽出中...")
    
    extract_prompt = f"""
以下の文章題を解析し、SymPyで解くための「方程式」と「変数」を抽出してください。
出力は必ず以下のフォーマットのみで行ってください。

EQUATION: <方程式 (例: 3*x - 10 = 20)>
VARIABLE: <変数名 (例: x)>

問題: {user_question}
"""

    response = generate_with_retry(extract_prompt)
    
    res_text = response.text
    eq_match = re.search(r'EQUATION:\s*(.+)', res_text)
    var_match = re.search(r'VARIABLE:\s*(.+)', res_text)
    
    if not eq_match or not var_match:
        print("❌ 方程式の抽出に失敗しました。")
        print("LLM応答:", res_text)
        return

    eq_str = eq_match.group(1).strip()
    var_str = var_match.group(1).strip()
    
    print(f"2. SymPyで厳密計算中... (式: {eq_str}, 変数: {var_str})")
    var = sp.Symbol(var_str)
    lhs_str, rhs_str = eq_str.split('=')
    eq = sp.Eq(sp.sympify(lhs_str), sp.sympify(rhs_str))
    solutions = sp.solve(eq, var)
    
    print("3. 解説文を生成中...")
    explain_prompt = f"""
以下の問題と計算結果を基に、分かりやすい解説を作成してください。

問題: {user_question}
方程式: {eq_str}
計算結果: {var_str} = {solutions}
"""
    
    final_response = generate_with_retry(explain_prompt)
    
    print("\n================ 数学AIの解答 ================")
    print(final_response.text)

if __name__ == "__main__":
    question = "ある数を3倍して10を引いた値が、その数に6を足した値と等しくなります。ある数はいくつですか？"
    process_math_question(question)