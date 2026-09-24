import time
import json
import warnings
from datetime import datetime
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types, errors
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import sympy as sp
import numpy as np
import plotly.graph_objects as go

# 警告非表示
warnings.filterwarnings("ignore")

# ページ設定（スマホ向けにサイドバー初期閉じ）
st.set_page_config(
    page_title="数学AI - スマホ版",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# スマホ画面（縦持ち）専用UIのCSS調整
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }
    .stButton > button {
        height: 3.5rem;
        font-size: 1.1rem !important;
        font-weight: bold;
        border-radius: 12px;
    }
    .stTextInput input, .stTextArea textarea {
        font-size: 1.1rem !important;
    }
    canvas {
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
# 0. データベース ＆ セッション初期化
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

if "review_list" not in st.session_state:
    st.session_state.review_list = []
if "history_list" not in st.session_state:
    st.session_state.history_list = []

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

# ==========================================
# 2. API 呼び出し関数 ＆ 数式解析関数
# ==========================================
def generate_with_retry(prompt: str, config=None, contents=None, max_retries: int = 3):
    models_to_try = ['gemini-3.6-flash', 'gemini-2.0-flash']
    input_contents = contents if contents is not None else prompt
    
    last_error = None
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=input_contents,
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
                
    raise RuntimeError("API利用制限に達しました。20秒ほど待ってから再試行してください。") from last_error

def generate_problem_set(grade: str, unit: str, level: int, count: int) -> list[MathProblem]:
    prompt = f"""
以下の条件に従って、数学の問題を【{count}問】まとめて作成してください。
- 学年: {grade} | 単元: {unit} | 難易度: Level {level}
※ 数式や記号は、LaTeX形式（例: $3x - 5 = 10$）で記述してください。
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

def analyze_handwritten_image(problem: MathProblem, image: Image.Image) -> str:
    prompt = f"""
添付された画像は、スマホから提出された手書き解答です。
問題: {problem.question} | 正解: {problem.final_answer}
1. 手書き文字（途中式・最終解答）を読み取ってください。
2. 採点・解説を行ってください。
"""
    response = generate_with_retry(prompt="", contents=[image, prompt])
    return response.text

def analyze_speed_session(grade: str, level: int, speed_problems: list[SpeedProblem], speed_answers: dict, correct_cnt: int, total_cnt: int) -> str:
    analysis_input = []
    for sp_p in speed_problems:
        u_ans = speed_answers.get(sp_p.id, "")
        is_ok = check_final_answer_sympy(u_ans, sp_p.final_answer) if u_ans else False
        analysis_input.append({
            "question": sp_p.question, "correct_answer": sp_p.final_answer, "user_answer": u_ans, "is_correct": is_ok
        })

    prompt = f"""
学年「{grade}」（Level {level}）のスマホスピード連答結果（{correct_cnt}/{total_cnt}問正解）を分析・アドバイスしてください。
【データ】{json.dumps(analysis_input, ensure_ascii=False, indent=2)}
"""
    response = generate_with_retry(prompt)
    return response.text

# ==========================================
# 3. ハンバーガーメニュー (サイドバー)
# ==========================================
with st.sidebar:
    st.title("📱 数学AI スマホ版")
    st.caption("左上の ☰ でメニューを閉じられます")
    st.markdown("---")
    
    app_mode = st.radio(
        "📌 メニュー",
        [
            "📝 モード1: じっくりテスト",
            "⚡ モード2: スピード連答",
            "📈 グラフ描画ツール",
            "🔄 復習チェックノート",
            "📈 学習履歴"
        ],
        index=0
    )
    st.markdown("---")
    
    if app_mode == "📝 モード1: じっくりテスト":
        st.subheader("⚙️ テスト設定")
        grade_m1 = st.selectbox("学年", list(UNIT_DATABASE.keys()), index=0, key="m_g1")
        unit_m1 = st.selectbox("単元", UNIT_DATABASE.get(grade_m1, []), key="m_u1")
        level_m1 = st.slider("難易度", 1, 10, 3, key="m_l1")
        count_m1 = st.number_input("問題数", min_value=1, max_value=10, value=3, step=1, key="m_c1")
        
        if st.button("🚀 出題スタート", type="primary", use_container_width=True):
            st.session_state.m_active = True
            st.session_state.m_grade = grade_m1
            st.session_state.m_unit = unit_m1
            st.session_state.m_level = level_m1
            st.session_state.m_count = count_m1
            st.session_state.m_idx = 0
            st.session_state.m_analyzed = False
            
            with st.spinner("AIが問題作成中..."):
                try:
                    st.session_state.m_problems = generate_problem_set(grade_m1, unit_m1, level_m1, count_m1)
                    st.session_state.m_start_time = time.time()
                    st.rerun()
                except Exception as err:
                    st.error(f"作成失敗: {err}")

    elif app_mode == "⚡ モード2: スピード連答":
        st.subheader("⚙️ 連答設定")
        grade_sp = st.selectbox("学年", list(UNIT_DATABASE.keys()), index=0, key="m_g2")
        level_sp = st.slider("難易度", 1, 10, 3, key="m_l2")
        count_sp = st.select_slider("問題数", options=[5, 10, 15, 20, 30], value=5, key="m_c2")
        
        if st.button("🔥 連答スタート", type="primary", use_container_width=True):
            st.session_state.sp_m_active = True
            st.session_state.sp_m_grade = grade_sp
            st.session_state.sp_m_level = level_sp
            st.session_state.sp_m_count = count_sp
            st.session_state.sp_m_idx = 0
            st.session_state.sp_m_answers = {}
            st.session_state.sp_m_finished = False
            
            with st.spinner("問題準備中..."):
                try:
                    st.session_state.sp_m_problems = generate_speed_problem_set(grade_sp, level_sp, count_sp)
                    st.session_state.sp_m_start_time = time.time()
                    st.rerun()
                except Exception as err:
                    st.error(f"作成失敗: {err}")

# ==========================================
# 4. メイン画面 (完全1カラム・スマホ表示)
# ==========================================

# ------------------------------------------
# メイン 1: じっくりテストモード
# ------------------------------------------
if app_mode == "📝 モード1: じっくりテスト":
    st.title("📱 じっくりテスト")
    
    if st.session_state.get("m_active", False) and "m_problems" in st.session_state:
        problems: list[MathProblem] = st.session_state.m_problems
        cur_i = st.session_state.m_idx
        tot_c = st.session_state.m_count
        prob = problems[cur_i]
        
        st.progress((cur_i + 1) / tot_c)
        st.caption(f"【{cur_i + 1} / {tot_c}問】 {st.session_state.m_grade} - {st.session_state.m_unit}")
        
        st.subheader(prob.title)
        st.markdown(prob.question)
        
        with st.expander("💡 ヒントを見る"):
            st.write(prob.hint)
            
        st.markdown("---")
        
        input_type = st.radio("入力方法を選択", ("⌨️ キーボード（テキスト）", "✏️ 手書きキャンバス"), horizontal=True)
        
        is_submitted = st.session_state.get("m_analyzed", False)
        
        if input_type == "⌨️ キーボード（テキスト）":
            u_steps = st.text_area("途中式メモ", height=100, key=f"text_steps_{cur_i}")
            u_final = st.text_input("最終解答 (例: x = 5)", key=f"text_final_{cur_i}")
            
            if not is_submitted:
                if st.button("📝 この問題の解答を判定", type="primary", use_container_width=True):
                    if not u_final:
                        st.warning("最終解答を入力してください。")
                    else:
                        is_ok = check_final_answer_sympy(u_final, prob.final_answer)
                        st.session_state.m_analyzed = True
                        st.session_state.m_last_res = {"is_ok": is_ok, "u_final": u_final}
                        
                        st.session_state.history_list.append({
                            "id": f"h_m1_{int(time.time())}",
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "mode": "モード1 (キーボード)",
                            "title": f"{st.session_state.m_grade} - {st.session_state.m_unit} (Q{cur_i+1})",
                            "score": "🟢 正解" if is_ok else "🔴 不正解",
                            "analysis": f"あなたの解答: {u_final} | 正解: {prob.final_answer}"
                        })
                        st.rerun()
        else:
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=3,
                stroke_color="#000000",
                background_color="#ffffff",
                height=220,
                width=320,
                drawing_mode="freedraw" if not is_submitted else "transform",
                key=f"m_canvas_{cur_i}_{prob.id}",
            )
            if not is_submitted:
                if st.button("🔍 手書き画像を判定", type="primary", use_container_width=True):
                    if canvas_result.image_data is not None:
                        with st.spinner("手書きをAI解析中..."):
                            img_data = canvas_result.image_data
                            img = Image.fromarray(img_data.astype('uint8'), 'RGBA').convert('RGB')
                            an_txt = analyze_handwritten_image(prob, img)
                            st.session_state.m_analyzed = True
                            st.session_state.m_last_res = {"is_ok": None, "analysis": an_txt}
                            
                            st.session_state.history_list.append({
                                "id": f"h_m1_hw_{int(time.time())}",
                                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                "mode": "モード1 (手書き)",
                                "title": f"{st.session_state.m_grade} - {st.session_state.m_unit} (Q{cur_i+1})",
                                "score": "AI画像添削済み",
                                "analysis": an_txt
                            })
                            st.rerun()

        if is_submitted:
            st.markdown("---")
            st.subheader("📊 判定結果")
            res = st.session_state.m_last_res
            if "is_ok" in res and res["is_ok"] is not None:
                if res["is_ok"]:
                    st.success(f"🎉 正解！ 正解: {prob.final_answer}")
                else:
                    st.error(f"❌ 残念！ 正解: {prob.final_answer}")
            if "analysis" in res:
                st.write(res["analysis"])
                
            with st.expander("📖 模範途中式"):
                for s in prob.solution_steps:
                    st.write(s)
                    
            if cur_i < tot_c - 1:
                if st.button("次へ進む ➔", type="primary", use_container_width=True):
                    st.session_state.m_idx += 1
                    st.session_state.m_analyzed = False
                    st.rerun()
            else:
                st.success("🎉 全問題終了！")
                if st.button("🔄 最初から解く", use_container_width=True):
                    st.session_state.m_active = False
                    st.rerun()
    else:
        st.info("👈 左上の ☰ メニューから条件を設定し、「出題スタート」を押してください。")

# ------------------------------------------
# メイン 2: スピード連答モード
# ------------------------------------------
elif app_mode == "⚡ モード2: スピード連答":
    st.title("⚡ スピード連答")
    
    if st.session_state.get("sp_m_active", False) and not st.session_state.get("sp_m_finished", False) and "sp_m_problems" in st.session_state:
        sp_problems: list[SpeedProblem] = st.session_state.sp_m_problems
        cur_i = st.session_state.sp_m_idx
        tot_c = st.session_state.sp_m_count
        cur_p = sp_problems[cur_i]

        st.progress((cur_i + 1) / tot_c)
        st.caption(f"Q{cur_i + 1} / {tot_c}  |  【{cur_p.unit}】")
        
        st.markdown(f"### **Q{cur_i + 1}. {cur_p.question}**")
        
        u_ans = st.text_input("答えを入力", key=f"sp_m_in_{cur_i}")
        
        if st.button("次へ ➔" if cur_i < tot_c - 1 else "🏁 提出・分析", type="primary", use_container_width=True):
            st.session_state.sp_m_answers[cur_p.id] = u_ans
            if cur_i < tot_c - 1:
                st.session_state.sp_m_idx += 1
                st.rerun()
            else:
                st.session_state.sp_m_finished = True
                st.rerun()

    elif st.session_state.get("sp_m_finished", False):
        st.balloons()
        sp_problems = st.session_state.sp_m_problems
        answers = st.session_state.sp_m_answers
        grade = st.session_state.sp_m_grade
        lvl = st.session_state.sp_m_level
        tot_c = st.session_state.sp_m_count
        
        correct_cnt = 0
        for p in sp_problems:
            if check_final_answer_sympy(answers.get(p.id, ""), p.final_answer):
                correct_cnt += 1
                
        st.header(f"🎉 終了！ ({correct_cnt}/{tot_c}問正解)")
        
        with st.spinner("分析レポート作成中..."):
            if "sp_m_analysis" not in st.session_state:
                an_txt = analyze_speed_session(grade, lvl, sp_problems, answers, correct_cnt, tot_c)
                st.session_state.sp_m_analysis = an_txt
                
                st.session_state.history_list.append({
                    "id": f"h_sp_m_{int(time.time())}",
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "mode": "モード2 (スピード連答)",
                    "title": f"学年: {grade} (Level {lvl} / 全{tot_c}問)",
                    "score": f"{correct_cnt} / {tot_c} 問正解",
                    "analysis": an_txt
                })
                
        st.write(st.session_state.sp_m_analysis)
        st.markdown("---")
        if st.button("🚀 もう一度挑戦", use_container_width=True):
            st.session_state.sp_m_active = False
            st.session_state.sp_m_finished = False
            if "sp_m_analysis" in st.session_state:
                del st.session_state["sp_m_analysis"]
            st.rerun()
    else:
        st.info("👈 左上の ☰ メニューから条件を設定し、「連答スタート」を押してください。")

# ------------------------------------------
# メイン 3: グラフ描画ツール (自由入力機能付き)
# ------------------------------------------
elif app_mode == "📈 グラフ描画ツール":
    st.title("📈 数学グラフ描画ツール")
    st.caption("プリセット関数およびユーザーが入力した数式を自動解析して描画します。")
    st.markdown("---")
    
    graph_type = st.radio("入力・次元モード選択", ("✏️ 自由数式入力 (2D/3D)", "📊 プリセット2D", "🧊 プリセット3D"), horizontal=True)
    
    if graph_type == "✏️ 自由数式入力 (2D/3D)":
        st.subheader("✏️ 自由数式描画")
        st.caption("例: `x**2 - 4`, `sin(x) * cos(x)`, `x**3`, `x**2 + y**2` など")
        
        user_expr_str = st.text_input("数式を入力 (右辺のみ)", value="x**2 - 2*x", help="Pythonの数式表現（*は掛け算、**は乗数）")
        
        dim_choice = st.radio("グラフ種類", ("2D 曲線 ($y = f(x)$)", "3D 曲面 ($z = f(x, y)$)"), horizontal=True)
        
        if dim_choice == "2D 曲線 ($y = f(x)$)":
            x_min, x_max = st.slider("X軸範囲", -10.0, 10.0, (-5.0, 5.0), step=0.5, key="usr_2d_x")
            try:
                x_sym = sp.Symbol('x')
                expr = sp.sympify(user_expr_str)
                f_np = sp.lambdify(x_sym, expr, modules=['numpy', 'sympy'])
                
                x_vals = np.linspace(x_min, x_max, 500)
                y_vals = f_np(x_vals)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name=f"y = {user_expr_str}"))
                fig.update_layout(
                    title=f"y = {user_expr_str}",
                    xaxis_title="x", yaxis_title="y",
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"数式を正しく解析できませんでした: {e}\n（例: `x**2 + 3*x - 5` のように演算子 `*` や `**` を正しく入力してください）")
        else:
            r_val = st.slider("X・Y軸範囲", 1.0, 10.0, 3.0, step=0.5, key="usr_3d_r")
            try:
                x_sym, y_sym = sp.symbols('x y')
                expr = sp.sympify(user_expr_str)
                f_np = sp.lambdify((x_sym, y_sym), expr, modules=['numpy', 'sympy'])
                
                x_vals = np.linspace(-r_val, r_val, 50)
                y_vals = np.linspace(-r_val, r_val, 50)
                X, Y = np.meshgrid(x_vals, y_vals)
                Z = f_np(X, Y)
                
                fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="Viridis")])
                fig_3d.update_layout(
                    title=f"z = {user_expr_str}",
                    scene=dict(xaxis_title="X軸", yaxis_title="Y軸", zaxis_title="Z軸"),
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            except Exception as e:
                st.error(f"数式を正しく解析できませんでした: {e}\n（3D描画には `x` と `y` の2変数を使った式を入力してください）")

    elif graph_type == "📊 プリセット2D":
        selected_funcs = st.multiselect(
            "表示する関数を選択（複数可）",
            ["y = sin(x)", "y = cos(x)", "y = tan(x)", "y = x³ (3次関数 / cm³)", "y = x² (2次関数)"],
            default=["y = sin(x)", "y = cos(x)"]
        )
        x_range = st.slider("X軸の範囲 ($x$)", -10.0, 10.0, (-5.0, 5.0), step=0.5)
        x = np.linspace(x_range[0], x_range[1], 500)
        fig = go.Figure()
        
        if "y = sin(x)" in selected_funcs:
            fig.add_trace(go.Scatter(x=x, y=np.sin(x), mode="lines", name="y = sin(x)"))
        if "y = cos(x)" in selected_funcs:
            fig.add_trace(go.Scatter(x=x, y=np.cos(x), mode="lines", name="y = cos(x)"))
        if "y = tan(x)" in selected_funcs:
            y_tan = np.tan(x)
            y_tan[np.abs(np.cos(x)) < 0.05] = np.nan
            fig.add_trace(go.Scatter(x=x, y=y_tan, mode="lines", name="y = tan(x)"))
        if "y = x³ (3次関数 / cm³)" in selected_funcs:
            fig.add_trace(go.Scatter(x=x, y=x**3, mode="lines", name="y = x³"))
        if "y = x² (2次関数)" in selected_funcs:
            fig.add_trace(go.Scatter(x=x, y=x**2, mode="lines", name="y = x²"))
            
        fig.update_layout(xaxis_title="x", yaxis_title="y", margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.caption("💡 指で画面上の立体グラフをドラッグすると、360度自由に回転・拡大できます。")
        func_3d = st.selectbox(
            "3D立体の関数を選択",
            [
                "z = x² + y² (放物面)",
                "z = sin(x) * cos(y) (波動・周期面)",
                "z = x * y (サドル・馬の鞍形状)",
                "z = x³ - 3x + y² (3次曲面)"
            ]
        )
        x_3d = np.linspace(-3, 3, 50)
        y_3d = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x_3d, y_3d)
        
        if func_3d == "z = x² + y² (放物面)":
            Z = X**2 + Y**2
        elif func_3d == "z = sin(x) * cos(y) (波動・周期面)":
            Z = np.sin(X) * np.cos(Y)
        elif func_3d == "z = x * y (サドル・馬の鞍形状)":
            Z = X * Y
        elif func_3d == "z = x³ - 3x + y² (3次曲面)":
            Z = X**3 - 3*X + Y**2
            
        fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="Viridis")])
        fig_3d.update_layout(scene=dict(xaxis_title="X軸", yaxis_title="Y軸", zaxis_title="Z軸"), margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_3d, use_container_width=True)

# ------------------------------------------
# メイン 4: 復習チェックノート
# ------------------------------------------
elif app_mode == "🔄 復習チェックノート":
    st.title("📚 復習ノート")
    if not st.session_state.review_list:
        st.info("保存された復習問題はありません。")
    else:
        if st.button("🗑️ ノートをクリア"):
            st.session_state.review_list = []
            st.rerun()
        st.markdown("---")
        for rev_idx, r_item in enumerate(st.session_state.review_list, 1):
            st.markdown(f"**問 {rev_idx}: [{r_item['grade']}] {r_item['title']}**")
            st.markdown(r_item["question"])
            rev_final = st.text_input("解答", key=f"m_rev_f_{r_item['id']}")
            if st.button(f"判定 (問 {rev_idx})", key=f"m_btn_chk_{r_item['id']}"):
                if check_final_answer_sympy(rev_final, r_item["final_answer"]):
                    st.success("🎉 正解！")
                else:
                    st.error(f"❌ 正解: {r_item['final_answer']}")
            st.markdown("---")

# ------------------------------------------
# メイン 5: 学習履歴
# ------------------------------------------
elif app_mode == "📈 学習履歴":
    st.title("📈 学習履歴")
    st.caption("過去に回答したテストのスコアとAI分析が自動で記録されます。")
    
    if not st.session_state.history_list:
        st.info("まだ学習履歴がありません。テストやスピード連答を解くと、ここに結果が記録されます。")
    else:
        st.write(f"累計実施数: **{len(st.session_state.history_list)} 回**")
        if st.button("🗑️ 履歴をすべて消去", use_container_width=True):
            st.session_state.history_list = []
            st.rerun()
            
        st.markdown("---")
        
        for h_item in reversed(st.session_state.history_list):
            with st.expander(f"🗓️ [{h_item['date']}] {h_item['mode']}\n{h_item['title']}"):
                st.write(f"**実施日時:** {h_item['date']}")
                st.write(f"**テスト形式:** {h_item['mode']}")
                st.write(f"**対象:** {h_item['title']}")
                st.write(f"**成績・判定:** {h_item['score']}")
                st.markdown("**【AI添削・分析アドバイス】**")
                st.write(h_item["analysis"])