import time
import json
import os
import warnings
import re
from datetime import datetime
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types, errors
import sympy as sp
import numpy as np
import plotly.graph_objects as go

# 警告非表示
warnings.filterwarnings("ignore")

# ページ設定
st.set_page_config(
    page_title="数学AI学習エージェント",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🎨 サイバーネオングリーン テーマ CSS
# ==========================================
st.markdown("""
<style>
    /* アプリ全体の背景を漆黒（サイバーブラック）に */
    .stApp {
        background-color: #0b0f12 !important;
        color: #00ff66 !important;
    }
    
    /* サイドバーの背景 */
    section[data-testid="stSidebar"] {
        background-color: #05080a !important;
        border-right: 1px solid #00ff6633 !important;
    }
    
    /* ヘッダー・タイトルテキスト */
    h1, h2, h3, h4, h5, h6, label, .stMarkdown {
        color: #ffffff !important;
        text-shadow: 0 0 5px rgba(0, 255, 102, 0.3);
    }
    
    /* メインボタン（蛍光ネオングリーン） */
    .stButton > button {
        background: linear-gradient(135deg, #00ff66 0%, #00cc52 100%) !important;
        color: #000000 !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 0 12px rgba(0, 255, 102, 0.5) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(0, 255, 102, 0.9) !important;
        transform: translateY(-2px);
    }
    
    /* 入力フォーム・テキストエリア・選択ボックス */
    .stTextInput input, .stTextArea textarea, div[data-baseweb="select"] {
        background-color: #12181d !important;
        color: #00ff66 !important;
        border: 1px solid #00ff66 !important;
        border-radius: 6px !important;
        box-shadow: inset 0 0 5px rgba(0, 255, 102, 0.2) !important;
    }
    
    /* アコーディオン・カード枠 */
    .streamlit-expanderHeader {
        background-color: #12181d !important;
        color: #00ff66 !important;
        border: 1px solid #00ff6644 !important;
        border-radius: 6px !important;
    }
    .streamlit-expanderContent {
        background-color: #0e1317 !important;
        border: 1px solid #00ff6622 !important;
    }

    /* キャンバス領域の外枠 */
    canvas {
        border: 2px solid #00ff66 !important;
        box-shadow: 0 0 15px rgba(0, 255, 102, 0.3) !important;
        touch-action: none !important;
    }
</style>
""", unsafe_allow_html=True)

# APIキー設定
import os
import streamlit as st

API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

@st.cache_resource
def get_client():
    return genai.Client(api_key=API_KEY)

client = get_client()

# ==========================================
# 👤 ユーザー別 JSON ファイル保存・読み込み管理
# ==========================================
def get_user_filename(username: str) -> str:
    safe_name = re.sub(r'[\\/*?:"<>|]', '_', username.strip())
    return f"user_data_{safe_name}.json" if safe_name else "user_data_default.json"

def load_user_data(username: str):
    filename = get_user_filename(username)
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "history_list": [],
        "review_list": [],
        "multi_expr_list_2d": ["x^2 - 4x", "2x + 1"],
        "multi_expr_list_3d": ["(3x - 4y)^3", "x^2 + y^2"]
    }

def save_user_data():
    username = st.session_state.get("current_user", "guest")
    filename = get_user_filename(username)
    data = {
        "history_list": st.session_state.get("history_list", []),
        "review_list": st.session_state.get("review_list", []),
        "multi_expr_list_2d": st.session_state.get("multi_expr_list_2d", ["x^2 - 4x", "2x + 1"]),
        "multi_expr_list_3d": st.session_state.get("multi_expr_list_3d", ["(3x - 4y)^3", "x^2 + y^2"])
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if "current_user" not in st.session_state:
    st.session_state.current_user = "guest"

init_data = load_user_data(st.session_state.current_user)
if "review_list" not in st.session_state:
    st.session_state.review_list = init_data.get("review_list", [])
if "history_list" not in st.session_state:
    st.session_state.history_list = init_data.get("history_list", [])
if "multi_expr_list_2d" not in st.session_state:
    st.session_state.multi_expr_list_2d = init_data.get("multi_expr_list_2d", ["x^2 - 4x", "2x + 1"])
if "multi_expr_list_3d" not in st.session_state:
    st.session_state.multi_expr_list_3d = init_data.get("multi_expr_list_3d", ["(3x - 4y)^3", "x^2 + y^2"])

# ==========================================
# 0. データベース＆定数
# ==========================================
UNIT_DATABASE = {
    "中1": ["正の数・負の数", "文字と式", "一次方程式", "比例と反比例", "平面図形", "空間図形", "データの活用"],
    "中2": ["式の計算", "連立方程式", "一次関数", "平行と合同（図形の性質）", "三角形と四角形", "確率", "データの比較"],
    "中3": ["多項式（展開・因数分解）", "平方根", "二次方程式", "関数 y = ax^2", "相似な図形", "円の性質（円周角の定理）", "三平方の定理", "標本調査"],
    "数I": ["数と式", "集合と命題", "二次関数", "図形と計量（三角比）", "データの分析"],
    "数A": ["場合の数と確率", "図形の性質", "数学と人間活動（整数の性質等）"],
    "数II": ["式と証明", "複素数と方程式", "図形と方程式", "三角関数", "指数関数・対数関数", "微分法と積分法（多項式関数）"],
    "数B": ["数列", "統計的な推測"],
    "数III": ["複素数平面", "式と曲線", "極限", "微分法（分数・三角・対数等）", "積分法（置換・部分積分・体積等）"],
    "数C": ["ベクトル", "平面上の曲線と複素数平面", "数学的帰納法・アルゴリズム"]
}

TIMER_OPTIONS = ["1分", "2分", "3分", "5分", "10分", "20分", "30分", "50分", "60分", "なし"]

# ==========================================
# 数式文字列の自動補正・正規化関数
# ==========================================
def normalize_math_expr(expr_str: str) -> str:
    s = expr_str.strip()
    s = s.replace('$', '').replace('y=', '').replace('y =', '').replace('z=', '').replace('z =', '')
    s = s.replace('^', '**')
    s = re.sub(r'\bPI\b|\bPi\b', 'pi', s)
    s = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', s)
    s = re.sub(r'(\))([a-zA-Z\d\(])', r'\1*\2', s)
    s = re.sub(r'([a-zA-Z\d])(\()', r'\1*\2', s)
    s = re.sub(r'(\*\*\d+)([a-zA-Z\(])', r'\1*\2', s)
    s = re.sub(r'(?<!p)(?<!i)([a-zA-Z])([a-zA-Z])(?!i)', r'\1*\2', s)
    return s

def extract_function_from_text(text: str) -> str:
    match = re.search(r'y\s*=\s*([^\s,$\\\}\)\\]+)', text)
    if match:
        return match.group(1).strip()
    match_frac = re.search(r'\\frac\{([^\}]+)\}\{x\}', text)
    if match_frac:
        return f"({match_frac.group(1)})/x"
    return ""

def render_proportional_graph(expr_str: str, title: str = "比例・反比例のグラフ"):
    try:
        parsed_expr = normalize_math_expr(expr_str)
        x_sym = sp.Symbol('x')
        expr = sp.sympify(parsed_expr)
        f_np = sp.lambdify(x_sym, expr, modules=['numpy'])

        fig = go.Figure()
        is_inverse = 'x' in str(sp.denom(expr))

        if is_inverse:
            x_neg = np.linspace(-10, -0.1, 250)
            x_pos = np.linspace(0.1, 10, 250)
            y_neg = f_np(x_neg)
            y_pos = f_np(x_pos)

            fig.add_trace(go.Scatter(x=x_neg, y=y_neg, mode="lines", name=f"y = {expr_str}", line=dict(color="#00ff66", width=2.5)))
            fig.add_trace(go.Scatter(x=x_pos, y=y_pos, mode="lines", name=f"y = {expr_str}", line=dict(color="#00ff66", width=2.5), showlegend=False))
        else:
            x_vals = np.linspace(-10, 10, 500)
            y_vals = f_np(x_vals)
            if isinstance(y_vals, (int, float)):
                y_vals = np.full_like(x_vals, y_vals)
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name=f"y = {expr_str}", line=dict(color="#00ff66", width=2.5)))

        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#00ff6633")
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#00ff6633")

        fig.update_layout(
            title=f"📈 {title}: y = {expr_str}",
            xaxis_title="X軸 (x)",
            yaxis_title="Y軸 (y)",
            paper_bgcolor="#0b0f12",
            plot_bgcolor="#0b0f12",
            font=dict(color="#00ff66"),
            xaxis=dict(range=[-10, 10], zeroline=True, gridcolor="#12181d"),
            yaxis=dict(range=[-10, 10], zeroline=True, gridcolor="#12181d"),
            margin=dict(l=20, r=20, t=40, b=20),
            height=350,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

# ==========================================
# 1. Pydantic 構造定義
# ==========================================
class MathProblem(BaseModel):
    id: int = Field(description="問題番号")
    title: str = Field(description="問題タイトル・単元名")
    question: str = Field(description="問題文（LaTeX形式含む）")
    final_answer: str = Field(description="最終正解")
    solution_steps: list[str] = Field(description="途中式ステップ")
    hint: str = Field(description="ヒント")

class ProblemSet(BaseModel):
    problems: list[MathProblem] = Field(description="問題リスト")

class SpeedProblem(BaseModel):
    id: int = Field(description="問題番号")
    unit: str = Field(description="該当する単元名")
    question: str = Field(description="難易度レベルに応じた問題")
    final_answer: str = Field(description="数値または簡単な数式（例: -6, x=3, 12）")

class SpeedProblemSet(BaseModel):
    problems: list[SpeedProblem] = Field(description="スピード問題リスト")

class KyotsuBlankItem(BaseModel):
    label: str = Field(description="マーク枠記号（例: ア, イウ, エオカ）")
    answer: str = Field(description="マーク枠の正解文字列（例: 3, -5, 12, √3）")

class KyotsuProblem(BaseModel):
    id: int = Field(description="大問番号")
    title: str = Field(description="大問タイトル")
    context_text: str = Field(description="問題文全体（LaTeX記号と【ア】【イウ】などのマーク枠を含む）")
    blanks: list[KyotsuBlankItem] = Field(description="全マーク枠の記号と正解リスト")
    solution_steps: list[str] = Field(description="詳細な誘導解説・途中式")

# ==========================================
# 2. API 呼び出し関数
# ==========================================
def generate_with_retry(prompt: str, config=None, max_retries: int = 3):
    models_to_try = ['gemini-3.6-flash', 'gemini-2.0-flash']
    last_error = None
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config
                )
            except errors.APIError as e:
                last_error = e
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    time.sleep(3 * (attempt + 1))
                else:
                    break
            except Exception as e:
                last_error = e
                break
                
    raise RuntimeError("API利用制限に達しました。しばらく待ってから再度お試しください。") from last_error

def generate_problem_set(grade: str, unit: str, level: int, count: int) -> list[MathProblem]:
    prompt = f"""
以下の条件に従って、数学の問題を【{count}問】まとめて作成してください。
- 学年: {grade} | 単元: {unit} | 難易度: Level {level}
※ 数式や記号は、LaTeX形式（例: $3x - 5 = 10$ や $y = \frac{{6}}{{x}}$）で記述してください。
"""
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ProblemSet,
        temperature=0.7,
    )
    response = generate_with_retry(prompt, config=config)
    data = json.loads(response.text)
    return ProblemSet(**data).problems

def generate_speed_problem_set(grade: str, level: int, count: int) -> list[SpeedProblem]:
    units_str = ", ".join(UNIT_DATABASE.get(grade, []))
    prompt = f"""
【学年別 スピード連答一問一答モード】
学年「{grade}」の全単元（{units_str}）から、指定された難易度レベルに合わせた連答問題を【{count}問】まとめて作成してください。

【難易度レベル設定: Level {level}】
- Level 1〜2: 1問あたり5秒〜10秒で暗算・即答できる超基礎問題。
- Level 3〜5: 1問あたり30秒〜1分程度で簡単な計算を行う基礎・標準問題。
- Level 6〜10: 1問あたり1分〜2分程度を要するやや複雑な計算問題。
"""
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SpeedProblemSet,
        temperature=0.5,
    )
    response = generate_with_retry(prompt, config=config)
    data = json.loads(response.text)
    return SpeedProblemSet(**data).problems

def generate_kyotsu_problem(grade: str, unit: str, level: int = 5) -> KyotsuProblem:
    prompt = f"""
大学入学共通テスト（数学）の形式に従って、誘導形式の穴埋め大問を作成してください。

【条件】
- 対象学年・科目: {grade}
- 該当単元: {unit}
- 難易度: Level {level}（共通テスト本試験レベル）
- マーク枠は 【ア】【イ】【ウエ】 のように角かっこで明記すること。
- 負の符号や分数、ルートが含まれる場合は、【ア】に「-」、【イ】に数字が入るような形式にする。
- 数式や変数、分数は LaTeX表記（例: $y = 2x^2 - 4x + 1$, $\\frac{{\\sqrt{{3}}}}{{2}}$）を用いること。
"""
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=KyotsuProblem,
        temperature=0.7,
    )
    response = generate_with_retry(prompt, config=config)
    data = json.loads(response.text)
    return KyotsuProblem(**data)

def check_final_answer_sympy(user_ans: str, correct_ans: str) -> bool:
    try:
        def clean_expr(s):
            s = s.replace('$', '').strip()
            if '=' in s:
                s = s.split('=')[-1].strip()
            return s

        user_eq = sp.sympify(clean_expr(user_ans))
        correct_eq = sp.sympify(clean_expr(correct_ans))
        return sp.simplify(user_eq - correct_eq) == 0
    except Exception:
        return user_ans.strip() == correct_ans.strip()

def analyze_all_answers(problems: list[MathProblem], user_answers: dict) -> str:
    analysis_input = []
    for p in problems:
        ans_data = user_answers.get(p.id, {"steps": "", "final": ""})
        analysis_input.append({
            "problem_id": p.id,
            "title": p.title,
            "question": p.question,
            "correct_answer": p.final_answer,
            "user_final": ans_data["final"],
            "user_steps": ans_data["steps"]
        })

    prompt = f"""
以下の数学問題セットに対するユーザーの解答を一括して評価・採点し、総括的な分析を出力してください。
【データ】{json.dumps(analysis_input, ensure_ascii=False, indent=2)}
"""
    response = generate_with_retry(prompt)
    return response.text

def analyze_speed_session(grade: str, level: int, speed_problems: list[SpeedProblem], speed_user_answers: dict, correct_cnt: int, total_cnt: int, accuracy: float) -> str:
    analysis_input = []
    for sp_p in speed_problems:
        u_ans = speed_user_answers.get(sp_p.id, "")
        is_ok = check_final_answer_sympy(u_ans, sp_p.final_answer) if u_ans else False
        analysis_input.append({
            "id": sp_p.id,
            "unit": sp_p.unit,
            "question": sp_p.question,
            "correct_answer": sp_p.final_answer,
            "user_answer": u_ans,
            "is_correct": is_ok
        })

    prompt = f"""
学年「{grade}」（難易度 Level {level}）のスピード連答テスト（全{total_cnt}問）が終了しました。
正解数: {correct_cnt} / {total_cnt} 問 (正解率: {accuracy}%)
"""
    response = generate_with_retry(prompt)
    return response.text

# ==========================================
# 3. サイドバー
# ==========================================
with st.sidebar:
    st.title("🧮 数学AI学習")
    
    st.subheader("👤 ユーザー切り替え")
    u_input = st.text_input("ユーザー名 / ID", value=st.session_state.current_user, key="user_id_input")
    
    if u_input.strip() and u_input.strip() != st.session_state.current_user:
        new_user = u_input.strip()
        st.session_state.current_user = new_user
        
        user_data = load_user_data(new_user)
        st.session_state.history_list = user_data.get("history_list", [])
        st.session_state.review_list = user_data.get("review_list", [])
        st.session_state.multi_expr_list_2d = user_data.get("multi_expr_list_2d", ["x^2 - 4x", "2x + 1"])
        st.session_state.multi_expr_list_3d = user_data.get("multi_expr_list_3d", ["(3x - 4y)^3", "x^2 + y^2"])
        
        st.toast(f"ユーザー『{new_user}』のデータを読み込みました！", icon="👤")
        st.rerun()

    st.caption(f"現在のログイン: **{st.session_state.current_user}**")
    st.markdown("---")
    
    app_mode = st.radio(
        "📌 機能を切り替え",
        [
            "📝 モード1: じっくりテスト",
            "⚡ モード2: スピード連答",
            "🎯 共通テスト穴埋め演習",
            "📈 グラフ描画ツール",
            "🔄 復習チェックノート",
            "📈 学習履歴・レポート"
        ],
        index=0
    )
    st.markdown("---")

    if app_mode == "📝 モード1: じっくりテスト":
        st.subheader("⚙️ モード1 条件設定")
        grade_m1 = st.selectbox("学年", list(UNIT_DATABASE.keys()), index=0, key="sb_m1_grade")
        unit_m1 = st.selectbox("単元", UNIT_DATABASE.get(grade_m1, []), key="sb_m1_unit")
        level_m1 = st.slider("難易度レベル", 1, 10, 3, key="sb_m1_lvl")
        timer_m1 = st.selectbox("制限時間", TIMER_OPTIONS, index=3, key="sb_m1_time")
        count_m1 = st.number_input("問題数", min_value=1, max_value=10, value=3, step=1, key="sb_m1_cnt")
        
        if st.button("🚀 テストを開始する", type="primary", use_container_width=True):
            st.session_state.m1_active = True
            st.session_state.m1_selected_grade = grade_m1
            st.session_state.m1_selected_unit = unit_m1
            st.session_state.m1_selected_level = level_m1
            st.session_state.m1_selected_timer = timer_m1
            st.session_state.m1_analyzed = False
            
            with st.spinner(f"AIが全 {count_m1} 問を作成中..."):
                try:
                    st.session_state.m1_problems = generate_problem_set(grade_m1, unit_m1, level_m1, count_m1)
                    st.session_state.m1_start_time = time.time()
                    st.rerun()
                except Exception as e:
                    st.error(f"作成失敗: {e}")

    elif app_mode == "⚡ モード2: スピード連答":
        st.subheader("⚙️ モード2 条件設定")
        grade_m2 = st.selectbox("学年", list(UNIT_DATABASE.keys()), index=0, key="sb_m2_grade")
        level_m2 = st.slider("難易度レベル", 1, 10, 3, key="sb_m2_lvl")
        count_m2 = st.select_slider("問題数", options=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], value=10, key="sb_m2_cnt")
        
        if st.button("🔥 スピード連答スタート", type="primary", use_container_width=True):
            st.session_state.m2_active = True
            st.session_state.m2_target_grade = grade_m2
            st.session_state.m2_target_level = level_m2
            st.session_state.m2_target_count = count_m2
            st.session_state.m2_current_idx = 0
            st.session_state.m2_user_answers = {}
            st.session_state.m2_finished = False
            st.session_state.m2_analysis_txt = ""
            
            with st.spinner(f"AIが問題（全 {count_m2} 問）を作成中..."):
                try:
                    st.session_state.m2_problems = generate_speed_problem_set(grade_m2, level_m2, count_m2)
                    st.session_state.m2_start_time = time.time()
                    st.rerun()
                except Exception as e:
                    st.error(f"作成失敗: {e}")

# ==========================================
# 4. メイン画面コンテンツ
# ==========================================

# ------------------------------------------
# メイン 1: モード1（じっくりテスト）
# ------------------------------------------
if app_mode == "📝 モード1: じっくりテスト":
    st.header(f"📝 モード1: 単元別 じっくりテスト（ユーザー: {st.session_state.current_user}）")
    st.caption("左側のサイドバーで単元と条件を設定し、「🚀 テストを開始する」を押してください。")
    st.markdown("---")

    if st.session_state.get("m1_active", False) and "m1_problems" in st.session_state:
        problems_m1: list[MathProblem] = st.session_state.m1_problems
        st.subheader(f"【{st.session_state.m1_selected_grade} - {st.session_state.m1_selected_unit}】（全 {len(problems_m1)} 問）")
        st.info(f"難易度: Level {st.session_state.m1_selected_level} | 目標時間: {st.session_state.m1_selected_timer}")

        user_inputs_m1 = {}
        for idx, p in enumerate(problems_m1, 1):
            st.markdown(f"### 📌 問 {idx}: {p.title}")
            st.markdown(p.question)

            is_proportional_unit = "比例" in st.session_state.m1_selected_unit or "関数" in st.session_state.m1_selected_unit
            detected_func = extract_function_from_text(p.question) or extract_function_from_text(p.final_answer)

            if is_proportional_unit or detected_func:
                with st.expander("📊 関連グラフを表示・確認", expanded=True):
                    func_to_draw = detected_func if detected_func else "2x"
                    render_proportional_graph(func_to_draw, f"問 {idx} 関連グラフ")

            with st.expander(f"💡 ヒントを見る"):
                st.write(p.hint)

            col_s, col_f = st.columns([2, 1])
            with col_s:
                s_in = st.text_area(f"問 {idx} 途中式メモ", key=f"m1_steps_{p.id}", height=90)
            with col_f:
                f_in = st.text_input(f"問 {idx} 最終解答", key=f"m1_final_{p.id}")
            user_inputs_m1[p.id] = {"steps": s_in, "final": f_in}
            st.markdown("---")

        if not st.session_state.get("m1_analyzed", False):
            if st.button("📝 送信して一括採点する", type="primary", use_container_width=True):
                elapsed = round((time.time() - st.session_state.m1_start_time) / 60, 1)
                with st.spinner("採点・分析中..."):
                    try:
                        sp_res = {}
                        correct_c = 0
                        for p in problems_m1:
                            u_ans = user_inputs_m1[p.id]["final"]
                            is_match = check_final_answer_sympy(u_ans, p.final_answer) if u_ans else False
                            sp_res[p.id] = is_match
                            if is_match:
                                correct_c += 1

                        an_txt = analyze_all_answers(problems_m1, user_inputs_m1)
                        st.session_state.m1_analyzed = True
                        st.session_state.m1_res_data = {
                            "elapsed": elapsed, "inputs": user_inputs_m1, "sp_res": sp_res, "an_txt": an_txt
                        }
                        
                        st.session_state.history_list.append({
                            "id": f"h_m1_{int(time.time())}",
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "mode": "モード1 (単元別)",
                            "title": f"{st.session_state.m1_selected_grade} - {st.session_state.m1_selected_unit}",
                            "score": f"{correct_c} / {len(problems_m1)} 問正解",
                            "elapsed": f"{elapsed}分",
                            "analysis": an_txt
                        })
                        save_user_data()
                        st.rerun()
                    except Exception as e:
                        st.error(f"採点に失敗しました: {e}")

        if st.session_state.get("m1_analyzed", False):
            res_m1 = st.session_state.m1_res_data
            st.header("📊 採点 ＆ 総合誤答分析結果")
            st.metric("所要時間", f"{res_m1['elapsed']}分", delta=f"目標 {st.session_state.m1_selected_timer}")
            st.markdown("### 🔍 AI指導アドバイス")
            st.write(res_m1["an_txt"])
            st.markdown("---")
            for p in problems_m1:
                u_data = res_m1["inputs"].get(p.id, {"steps": "", "final": ""})
                is_ok = res_m1["sp_res"].get(p.id, False)
                with st.expander(f"📌 問 {p.id}: {p.title} （判定: {'🟢 正解' if is_ok else '🔴 不正解'}）", expanded=True):
                    st.write(f"**問題:** {p.question}")
                    st.write(f"**あなたの解答:** {u_data['final']} | **模範正解:** {p.final_answer}")
                    
                    ans_func = extract_function_from_text(p.final_answer) or extract_function_from_text(p.question)
                    if ans_func:
                        render_proportional_graph(ans_func, f"問 {p.id} 正解グラフ")

                    if st.button(f"⭐ 復習ノートに追加 (問 {p.id})", key=f"m1_add_rev_{p.id}"):
                        st.session_state.review_list.append({
                            "id": f"m1_{p.id}_{int(time.time())}",
                            "grade": st.session_state.m1_selected_grade, "unit": st.session_state.m1_selected_unit,
                            "title": p.title, "question": p.question, "final_answer": p.final_answer,
                            "solution_steps": p.solution_steps, "hint": p.hint, "user_last_ans": u_data['final']
                        })
                        save_user_data()
                        st.toast(f"問 {p.id} を復習ノートに追加しました！")

# ------------------------------------------
# メイン 2: モード2（スピード連答）
# ------------------------------------------
elif app_mode == "⚡ モード2: スピード連答":
    st.header(f"⚡ モード2: 学年別 スピード連答（ユーザー: {st.session_state.current_user}）")
    st.caption("左側のサイドバーで学年・難易度・問題数を設定し、「🔥 スピード連答スタート」を押してください。")
    st.markdown("---")

    if st.session_state.get("m2_active", False) and not st.session_state.get("m2_finished", False) and "m2_problems" in st.session_state:
        sp_problems: list[SpeedProblem] = st.session_state.m2_problems
        cur_i = st.session_state.m2_current_idx
        tot_c = st.session_state.m2_target_count
        cur_p = sp_problems[cur_i]

        st.progress((cur_i + 1) / tot_c)
        st.caption(f"問題 {cur_i + 1} / {tot_c}  |  該当単元: 【{cur_p.unit}】  |  難易度: Level {st.session_state.m2_target_level}")
        
        st.markdown(f"## **Q{cur_i + 1}. {cur_p.question}**")

        if "比例" in cur_p.unit or "関数" in cur_p.unit:
            sp_func = extract_function_from_text(cur_p.question) or extract_function_from_text(cur_p.final_answer)
            if sp_func:
                with st.expander("📊 グラフを表示", expanded=True):
                    render_proportional_graph(sp_func, f"Q{cur_i + 1} 関連グラフ")

        elapsed_sec = int(time.time() - st.session_state.m2_start_time)
        st.caption(f"⏱️ 全体経過時間: {elapsed_sec // 60}分 {elapsed_sec % 60}秒")
        
        user_ans_val = st.text_input("解答を入力（Enterキーで送信）", key=f"sp_ans_input_{cur_i}")
        
        if st.button("次の問題へ ➔" if cur_i < tot_c - 1 else "🏁 全問提出・総合分析を見る", type="primary", use_container_width=True):
            st.session_state.m2_user_answers[cur_p.id] = user_ans_val
            if cur_i < tot_c - 1:
                st.session_state.m2_current_idx += 1
                st.rerun()
            else:
                st.session_state.m2_total_elapsed_sec = int(time.time() - st.session_state.m2_start_time)
                st.session_state.m2_finished = True
                st.rerun()

    elif st.session_state.get("m2_finished", False):
        st.balloons()
        sp_problems = st.session_state.m2_problems
        answers = st.session_state.m2_user_answers
        target_grade = st.session_state.m2_target_grade
        target_lvl = st.session_state.m2_target_level
        tot_c = st.session_state.m2_target_count
        tot_sec = st.session_state.m2_total_elapsed_sec
        
        st.header(f"🎉 スピード連答終了！ ({target_grade} - Level {target_lvl} 全 {tot_c} 問)")
        
        correct_cnt = 0
        detail_list = []
        for p in sp_problems:
            u_ans = answers.get(p.id, "")
            is_ok = check_final_answer_sympy(u_ans, p.final_answer) if u_ans else False
            if is_ok:
                correct_cnt += 1
            detail_list.append({"p": p, "u_ans": u_ans, "is_ok": is_ok})
            
        accuracy = round((correct_cnt / tot_c) * 100, 1)
        avg_sec_per_q = round(tot_sec / tot_c, 1)
        
        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("正解率", f"{accuracy}%", f"{correct_cnt} / {tot_c} 問正解")
        c_m2.metric("合計時間", f"{tot_sec // 60}分 {tot_sec % 60}秒")
        c_m3.metric("1問あたりの平均速度", f"{avg_sec_per_q}秒/問")
        
        st.markdown("---")
        with st.spinner("AIが総合分析レポートを作成中..."):
            if not st.session_state.m2_analysis_txt:
                st.session_state.m2_analysis_txt = analyze_speed_session(
                    target_grade, target_lvl, sp_problems, answers, correct_cnt, tot_c, accuracy
                )
                
                st.session_state.history_list.append({
                    "id": f"h_m2_{int(time.time())}",
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "mode": "モード2 (スピード連答)",
                    "title": f"学年: {target_grade} (Level {target_lvl} / 全{tot_c}問)",
                    "score": f"正解率: {accuracy}% ({correct_cnt}/{tot_c})",
                    "elapsed": f"{tot_sec // 60}分 {tot_sec % 60}秒",
                    "analysis": st.session_state.m2_analysis_txt
                })
                save_user_data()
            
        st.markdown("### 📊 AIによるスピード連答 総合分析レポート")
        st.write(st.session_state.m2_analysis_txt)
        st.markdown("---")
        
        st.subheader("📝 全解答・正解一覧")
        for item in detail_list:
            p = item["p"]
            status_mark = "🟢 正解" if item["is_ok"] else "🔴 不正解"
            with st.expander(f"問 {p.id} 【{p.unit}】 {status_mark} | 問題: {p.question}"):
                st.write(f"あなたの解答: **{item['u_ans']}**")
                st.write(f"模範正解: **{p.final_answer}**")
                if not item["is_ok"]:
                    if st.button(f"⭐ 復習ノートに追加 (問 {p.id})", key=f"m2_add_rev_{p.id}"):
                        st.session_state.review_list.append({
                            "id": f"m2_{p.id}_{int(time.time())}",
                            "grade": target_grade, "unit": p.unit,
                            "title": f"スピード連答問題 (Lvl {target_lvl})", "question": p.question,
                            "final_answer": p.final_answer, "solution_steps": [f"正解: {p.final_answer}"],
                            "hint": "基礎問題です", "user_last_ans": item["u_ans"]
                        })
                        save_user_data()
                        st.toast("復習ノートに追加しました！")

# ------------------------------------------
# メイン 3: 🎯 共通テスト穴埋め演習モード
# ------------------------------------------
elif app_mode == "🎯 共通テスト穴埋め演習":
    st.header(f"🎯 共通テスト形式 穴埋め演習（ユーザー: {st.session_state.current_user}）")
    st.caption("【ア】【イウ】などのマーク枠に適切な数値・符号を入力して解答してください。")
    st.markdown("---")

    col_k1, col_k2, col_k3 = st.columns([2, 2, 1])
    with col_k1:
        grade_k = st.selectbox("科目選択", ["数I・A", "数II・B・C"], key="k_grade_sel")
    with col_k2:
        unit_k = st.selectbox("単元選択", UNIT_DATABASE.get("数I" if "数I" in grade_k else "数II", ["二次関数", "図形と計量", "三角関数", "数列", "ベクトル"]), key="k_unit_sel")
    with col_k3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 大問生成", type="primary", use_container_width=True):
            with st.spinner("共通テスト形式の大問を作成中..."):
                try:
                    st.session_state.k_prob = generate_kyotsu_problem(grade_k, unit_k)
                    st.session_state.k_user_inputs = {}
                    st.session_state.k_submitted = False
                    st.rerun()
                except Exception as e:
                    st.error(f"作成失敗: {e}")

    if "k_prob" in st.session_state:
        prob_k: KyotsuProblem = st.session_state.k_prob
        st.markdown("---")
        st.subheader(f"📌 {prob_k.title}")
        st.markdown(prob_k.context_text)
        st.markdown("---")

        st.subheader("✏️ マーク解答欄")
        user_k_answers = {}
        cols_b = st.columns(3)
        for b_idx, b_item in enumerate(prob_k.blanks):
            with cols_b[b_idx % 3]:
                user_k_answers[b_item.label] = st.text_input(
                    f"【{b_item.label}】",
                    key=f"blank_in_{b_item.label}",
                    placeholder="例: -3 や 12"
                )

        if not st.session_state.get("k_submitted", False):
            if st.button("📝 送信して一括採点", type="primary", use_container_width=True):
                st.session_state.k_submitted = True
                st.session_state.k_user_inputs = user_k_answers
                st.rerun()

        if st.session_state.get("k_submitted", False):
            st.markdown("---")
            st.header("📊 採点結果")
            correct_cnt = 0
            
            for b_item in prob_k.blanks:
                u_ans = st.session_state.k_user_inputs.get(b_item.label, "").strip()
                is_ok = (u_ans == b_item.answer.strip())
                if is_ok:
                    correct_cnt += 1
                    st.success(f"【{b_item.label}】 あなたの解答: `{u_ans}` ⭕ (正解: `{b_item.answer}`)")
                else:
                    st.error(f"【{b_item.label}】 あなたの解答: `{u_ans if u_ans else '未入力'}` ❌ (正解: `{b_item.answer}`)")

            st.metric("正解率", f"{round((correct_cnt / len(prob_k.blanks)) * 100, 1)}%", f"{correct_cnt} / {len(prob_k.blanks)} 枠正解")

            with st.expander("📖 詳細な誘導解説・ステップを見る", expanded=True):
                for s_idx, step in enumerate(prob_k.solution_steps, 1):
                    st.write(f"**Step {s_idx}:** {step}")

# ------------------------------------------
# メイン 4: 2D/3Dマルチ重ね描き＆範囲自由設定ツール
# ------------------------------------------
elif app_mode == "📈 グラフ描画ツール":
    st.header(f"📈 マルチ数式グラフ描画ツール (ユーザー: {st.session_state.current_user})")
    st.caption("複数の数式を1つのグラフに重ねて描画できます。負の領域を含む軸の表示範囲も自由に設定できます。")
    st.markdown("---")

    dimension_mode = st.radio("次元を選択", ("📊 2D 曲線 ($y = f(x)$)", "🧊 3D 曲面 ($z = f(x, y)$)"), horizontal=True)
    st.markdown("---")

    col_ctrl, col_graph = st.columns([1, 2], gap="large")

    if "2D" in dimension_mode:
        with col_ctrl:
            st.subheader("✏️ 2D数式の追加と管理")
            new_expr_2d = st.text_input("追加する2D数式（例: 2x - 3, sin(x), x^2 - 4）", value="", placeholder="右辺のみ入力")
            if st.button("➕ 2Dグラフに追加", type="primary", use_container_width=True):
                if new_expr_2d.strip():
                    st.session_state.multi_expr_list_2d.append(new_expr_2d.strip())
                    save_user_data()
                    st.rerun()

            st.markdown("---")
            st.subheader("📋 描画中の2D数式")
            if not st.session_state.multi_expr_list_2d:
                st.info("登録されている2D数式はありません。")
            else:
                for idx, expr_item in enumerate(list(st.session_state.multi_expr_list_2d)):
                    col_e1, col_e2 = st.columns([3, 1])
                    col_e1.write(f"**y = {expr_item}**")
                    if col_e2.button("🗑️ 削除", key=f"del_2d_{idx}"):
                        st.session_state.multi_expr_list_2d.pop(idx)
                        save_user_data()
                        st.rerun()

            if st.session_state.multi_expr_list_2d:
                if st.button("🧹 全て消去", key="clr_2d", use_container_width=True):
                    st.session_state.multi_expr_list_2d = []
                    save_user_data()
                    st.rerun()

            st.markdown("---")
            st.subheader("⚙️ 2D軸範囲の設定")
            col_x1, col_x2 = st.columns(2)
            x_min = col_x1.number_input("X軸 最小値", value=-10.0, step=1.0, key="2d_xmin")
            x_max = col_x2.number_input("X軸 最大値", value=10.0, step=1.0, key="2d_xmax")

            col_y1, col_y2 = st.columns(2)
            y_min = col_y1.number_input("Y軸 最小値", value=-10.0, step=1.0, key="2d_ymin")
            y_max = col_y2.number_input("Y軸 最大値", value=10.0, step=1.0, key="2d_ymax")

        with col_graph:
            st.subheader("📊 2D 重ね合わせグラフ")
            if x_min >= x_max or y_min >= y_max:
                st.error("最小値は最大値より小さい値を設定してください。")
            else:
                fig = go.Figure()
                x_vals = np.linspace(x_min, x_max, 500)
                x_sym = sp.Symbol('x')

                for expr_item in st.session_state.multi_expr_list_2d:
                    try:
                        parsed_expr = normalize_math_expr(expr_item)
                        expr = sp.sympify(parsed_expr)
                        f_np = sp.lambdify(x_sym, expr, modules=['numpy'])
                        y_vals = f_np(x_vals)
                        if isinstance(y_vals, (int, float)):
                            y_vals = np.full_like(x_vals, y_vals)

                        fig.add_trace(go.Scatter(
                            x=x_vals, y=y_vals, mode="lines", name=f"y = {expr_item}"
                        ))
                    except Exception:
                        st.warning(f"『y = {expr_item}』を解析できませんでした（スキップ）")

                fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#00ff6633")
                fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#00ff6633")
                fig.update_layout(
                    xaxis_title="X軸 (x)", yaxis_title="Y軸 (y)",
                    paper_bgcolor="#0b0f12", plot_bgcolor="#0b0f12", font=dict(color="#00ff66"),
                    xaxis=dict(range=[x_min, x_max], zeroline=True, gridcolor="#12181d"),
                    yaxis=dict(range=[y_min, y_max], zeroline=True, gridcolor="#12181d"),
                    margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified", height=600,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(11,15,18,0.8)")
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        with col_ctrl:
            st.subheader("✏️ 3D数式の追加と管理")
            new_expr_3d = st.text_input("追加する3D数式（例: (3x - 4y)^3, x^2 + y^2, sin(x)cos(y)）", value="", placeholder="右辺のみ入力")
            if st.button("➕ 3Dグラフに追加", type="primary", use_container_width=True):
                if new_expr_3d.strip():
                    st.session_state.multi_expr_list_3d.append(new_expr_3d.strip())
                    save_user_data()
                    st.rerun()

            st.markdown("---")
            st.subheader("📋 描画中の3D曲面")
            if not st.session_state.multi_expr_list_3d:
                st.info("登録されている3D数式はありません。")
            else:
                for idx, expr_item in enumerate(list(st.session_state.multi_expr_list_3d)):
                    col_e1, col_e2 = st.columns([3, 1])
                    col_e1.write(f"**z = {expr_item}**")
                    if col_e2.button("🗑️ 削除", key=f"del_3d_{idx}"):
                        st.session_state.multi_expr_list_3d.pop(idx)
                        save_user_data()
                        st.rerun()

            if st.session_state.multi_expr_list_3d:
                if st.button("🧹 全て消去", key="clr_3d", use_container_width=True):
                    st.session_state.multi_expr_list_3d = []
                    save_user_data()
                    st.rerun()

            st.markdown("---")
            st.subheader("⚙️ 3D各軸範囲の設定")
            col_x1, col_x2 = st.columns(2)
            x3_min = col_x1.number_input("X軸 最小値", value=-1.5, step=0.5, key="3d_xmin")
            x3_max = col_x2.number_input("X軸 最大値", value=1.5, step=0.5, key="3d_xmax")

            col_y1, col_y2 = st.columns(2)
            y3_min = col_y1.number_input("Y軸 最小値", value=-1.5, step=0.5, key="3d_ymin")
            y3_max = col_y2.number_input("Y軸 最大値", value=1.5, step=0.5, key="3d_ymax")

            col_z1, col_z2 = st.columns(2)
            z3_min = col_z1.number_input("Z軸 最小値", value=-10.0, step=1.0, key="3d_zmin")
            z3_max = col_z2.number_input("Z軸 最大値", value=10.0, step=1.0, key="3d_zmax")

        with col_graph:
            st.subheader("🧊 3D 重ね合わせ立体曲面")
            if x3_min >= x3_max or y3_min >= y3_max or z3_min >= z3_max:
                st.error("最小値は最大値より小さい値を設定してください。")
            else:
                fig_3d = go.Figure()
                x_vals = np.linspace(x3_min, x3_max, 50)
                y_vals = np.linspace(y3_min, y3_max, 50)
                X, Y = np.meshgrid(x_vals, y_vals)
                x_sym, y_sym = sp.symbols('x y')

                opacity_list = [0.85, 0.65, 0.5]
                colorscales = ["Viridis", "Plasma", "Cividis", "Blues", "Reds"]

                for idx, expr_item in enumerate(st.session_state.multi_expr_list_3d):
                    try:
                        parsed_expr = normalize_math_expr(expr_item)
                        expr = sp.sympify(parsed_expr)
                        f_np = sp.lambdify((x_sym, y_sym), expr, modules=['numpy'])
                        Z = f_np(X, Y)
                        if isinstance(Z, (int, float)):
                            Z = np.full_like(X, Z)

                        Z = np.clip(Z, z3_min, z3_max)

                        fig_3d.add_trace(go.Surface(
                            z=Z, x=X, y=Y,
                            name=f"z = {expr_item}",
                            showscale=False,
                            opacity=opacity_list[idx % len(opacity_list)],
                            colorscale=colorscales[idx % len(colorscales)]
                        ))
                    except Exception:
                        st.warning(f"『z = {expr_item}』を解析できませんでした（スキップ）")

                fig_3d.update_layout(
                    paper_bgcolor="#0b0f12", font=dict(color="#00ff66"),
                    scene=dict(
                        xaxis=dict(title="X軸", range=[x3_min, x3_max], backgroundcolor="#0b0f12", gridcolor="#12181d"),
                        yaxis=dict(title="Y軸", range=[y3_min, y3_max], backgroundcolor="#0b0f12", gridcolor="#12181d"),
                        zaxis=dict(title="Z軸", range=[z3_min, z3_max], backgroundcolor="#0b0f12", gridcolor="#12181d"),
                    ),
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=600,
                    showlegend=True
                )
                st.plotly_chart(fig_3d, use_container_width=True)

# ------------------------------------------
# メイン 5: 復習チェックノート
# ------------------------------------------
elif app_mode == "🔄 復習チェックノート":
    st.header(f"📚 復習チェック・解き直しノート（ユーザー: {st.session_state.current_user}）")
    if not st.session_state.review_list:
        st.info("復習ノートに保存された問題はありません。")
    else:
        st.write(f"現在 **{len(st.session_state.review_list)} 問** の復習問題が保存されています。")
        if st.button("🗑️ 復習ノートをクリア"):
            st.session_state.review_list = []
            save_user_data()
            st.rerun()
        st.markdown("---")
        for rev_idx, r_item in enumerate(list(st.session_state.review_list), 1):
            st.markdown(f"### 📌 復習問 {rev_idx}: [{r_item['grade']} / {r_item['unit']}] {r_item['title']}")
            st.markdown(r_item["question"])
            rev_final = st.text_input("再チャレンジ解答", key=f"rev_f_{r_item['id']}")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if st.button(f"判定 (問 {rev_idx})", key=f"btn_chk_{r_item['id']}"):
                    if check_final_answer_sympy(rev_final, r_item["final_answer"]):
                        st.success(f"🎉 正解！ {r_item['final_answer']}")
                    else:
                        st.error(f"❌ 不正解！ 正解: {r_item['final_answer']}")
            with col_r2:
                if st.button(f"✅ 克服（削除）", key=f"btn_del_{r_item['id']}"):
                    st.session_state.review_list.pop(rev_idx - 1)
                    save_user_data()
                    st.rerun()
            st.markdown("---")

# ------------------------------------------
# メイン 6: 学習履歴・過去レポート
# ------------------------------------------
elif app_mode == "📈 学習履歴・レポート":
    st.header(f"📈 学習履歴・過去レポート（ユーザー: {st.session_state.current_user}）")
    st.caption("テストの採点履歴およびAI分析レポートがユーザー専用のJSONファイルに自動保存されます。")
    if not st.session_state.history_list:
        st.info("まだ学習履歴がありません。テストを完了すると自動的に記録されます。")
    else:
        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            st.write(f"累計実施回数: **{len(st.session_state.history_list)} 回**")
        with col_h2:
            if st.button("🗑️ 履歴をすべて消去"):
                st.session_state.history_list = []
                save_user_data()
                st.rerun()
        st.markdown("---")
        for h_item in reversed(st.session_state.history_list):
            with st.expander(f"🗓️ [{h_item['date']}] {h_item['mode']} : {h_item['title']} | 成績: {h_item['score']}"):
                st.write(f"**実施日時:** {h_item['date']}")
                st.write(f"**形式:** {h_item['mode']}")
                st.write(f"**対象:** {h_item['title']}")
                st.write(f"**成績 / 正解率:** {h_item['score']}")
                st.write(f"**所要時間:** {h_item['elapsed']}")
                st.markdown("**【AI総合分析レポート】**")
                st.write(h_item["analysis"])