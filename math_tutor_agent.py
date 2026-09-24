import time
import json
import warnings
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import sympy as sp

# 警告メッセージを非表示
warnings.filterwarnings("ignore")

# APIキーの設定
import os
import streamlit as st

API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=API_KEY)

# ==========================================
# 1. Pydanticによる問題データの構造定義
# ==========================================
class MathProblem(BaseModel):
    title: str = Field(description="問題のタイトル（例: 一次方程式の計算）")
    question: str = Field(description="問題文（LaTeX形式の数式を含む）")
    final_answer: str = Field(description="最終的な正解（例: x = 5 または 5）")
    solution_steps: list[str] = Field(description="模範解答の途中式ステップのリスト")
    hint: str = Field(description="難易度に応じた解法のヒントやポイント")

# ==========================================
# 2. 問題生成エンジン
# ==========================================
def generate_problem(grade: str, unit: str, level: int) -> MathProblem:
    print(f"\n🎲 【問題作成中】{grade} / {unit} / 難易度 Level {level} ...")
    
    prompt = f"""
以下の条件に従って、数学の問題を1問作成してください。

- 学年: {grade}
- 単元: {unit}
- 難易度: Level {level} (1:基礎〜10:最高難易度の入試・大学受験レベル)

難易度レベルの基準:
Level 1〜3: 基本概念・計算の基礎（教科書例題レベル）
Level 4〜7: 標準・応用（定期テスト〜公立高校・標準大学入試レベル）
Level 8〜10: 発展・難問（難関高校・偏差値60以上の大学入試レベル）

※ 数式や記号は、表示が崩れないよう可能な限りLaTeX形式（例: $3x - 5 = 10$）で記述してください。
"""

    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MathProblem,
            temperature=0.7,
        ),
    )
    
    # JSON文字列をMathProblemオブジェクトに変換
    data = json.loads(response.text)
    return MathProblem(**data)

# ==========================================
# 3. SymPy による厳密な等価判定（最終解のチェック）
# ==========================================
def check_final_answer_sympy(user_ans: str, correct_ans: str) -> bool:
    try:
        # "x = 5" などの形式から右辺や数式のみを抽出する試み
        def clean_expr(s):
            s = s.replace('$', '').strip()
            if '=' in s:
                s = s.split('=')[-1].strip()
            return s

        user_clean = clean_expr(user_ans)
        correct_clean = clean_expr(correct_ans)

        user_eq = sp.sympify(user_clean)
        correct_eq = sp.sympify(correct_clean)

        # 差が 0 であれば同一と判定
        return sp.simplify(user_eq - correct_eq) == 0
    except Exception:
        # SymPyで変換できない文字列の場合は文字列の一致で判定
        return user_ans.strip() == correct_ans.strip()

# ==========================================
# 4. 採点・途中式＆誤答分析エンジン
# ==========================================
def analyze_answer(problem: MathProblem, user_final: str, user_steps: str) -> str:
    print("\n🔍 【AI採点・誤答分析中】...")
    
    prompt = f"""
以下の数学問題に対するユーザーの解答を検証し、採点および誤答分析を行ってください。

【問題情報】
問題: {problem.question}
正解: {problem.final_answer}
模範途中式:
{json.dumps(problem.solution_steps, ensure_ascii=False, indent=2)}

【ユーザーの解答】
ユーザーの最終解答: {user_final}
ユーザーの途中式メモ:
{user_steps if user_steps else "（記載なし）"}

【指示】
1. 最終解答および途中式メモを評価してください。
2. 正誤判定（正解 / 部分点 / 不正解）を明確に示してください。
3. 途中式で計算ミスや論理の飛躍があった場合、どのステップで間違えたかを具体的に指摘してください。
4. 次回に活かせるアドバイスや解説を提示してください。
"""

    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt
    )
    return response.text

# ==========================================
# 5. メイン実行ループ（1サイクルのフロー）
# ==========================================
def run_math_cycle():
    print("==================================================")
    print("      数学AI学習エージェント (Step 1 Prototype)")
    print("==================================================")
    
    # 1. 学年・単元・難易度の入力
    print("\n--- [設定フェーズ] ---")
    print("学年の指定例: 中1, 中2, 中3, 数I, 数II, 数III, 数A, 数B, 数C")
    grade = input("学年を入力してください: ").strip() or "中1"
    
    unit = input("単元を入力してください (例: 一次方程式, 二次関数, 微分法): ").strip() or "一次方程式"
    
    while True:
        try:
            level_in = input("難易度レベルを入力してください (1〜10): ").strip() or "3"
            level = int(level_in)
            if 1 <= level <= 10:
                break
            print("1から10の数値で入力してください。")
        except ValueError:
            print("数値で入力してください。")
            
    limit_time_min = input("制限時間(分)を入力してください (標準: 5分): ").strip() or "5"

    # 2. 問題作成
    problem = generate_problem(grade, unit, level)
    
    print("\n==================================================")
    print(f"【問題】 {problem.title}")
    print(f"難易度: Level {level} | 想定時間: {limit_time_min}分")
    print("--------------------------------------------------")
    print(f"{problem.question}")
    print("--------------------------------------------------")
    print(f"💡 ヒント: {problem.hint}")
    print("==================================================")
    
    # 計測開始
    start_time = time.time()
    
    # 3. 利用者による解答入力
    print("\n--- [解答入力フェーズ] ---")
    user_steps = input("途中式（考え方や計算過程）を入力してください (省略可/Enterキー): ").strip()
    user_final = input("最終解答を入力してください (例: x = 5 または 5): ").strip()
    
    elapsed_time = round((time.time() - start_time) / 60, 1)
    
    # 4. 数理判定 (SymPyによる一次チェック)
    is_exact_correct = check_final_answer_sympy(user_final, problem.final_answer)
    
    # 5 & 6. AIによる自動採点と誤答分析
    analysis_result = analyze_answer(problem, user_final, user_steps)
    
    # 結果出力
    print("\n==================================================")
    print("                 【採点＆分析結果】")
    print("==================================================")
    print(f"⏱️ 所要時間: {elapsed_time}分 / 設定制限時間: {limit_time_min}分")
    print(f"⚙️ SymPy数理検証 (最終解の一致): {'成功 (一致)' if is_exact_correct else '不一致 / 要AI確認'}")
    print("--------------------------------------------------")
    print(analysis_result)
    print("==================================================")
    print("\n【模範解答の途中式】")
    for idx, step in enumerate(problem.solution_steps, 1):
        print(f"  Step {idx}: {step}")
    print(f"  最終正解: {problem.final_answer}")
    print("==================================================\n")

if __name__ == "__main__":
    run_math_cycle()