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
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import sympy as sp
import numpy as np
import plotly.graph_objects as go

# 警告非表示
warnings.filterwarnings("ignore")

# ページ設定（ワイドレイアウト・タブレットUI）
st.set_page_config(
    page_title="数学AI - タブレット完全版",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# UI・タッチ挙動最適化CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    .stButton > button {
        height: 3.2rem;
        font-size: 1.1rem !important;
        font-weight: bold;
        border-radius: 10px;
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
# JSON ファイル保存・読み込み管理
# ==========================================
DATA_FILE = "user_data.json"

def load_user_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
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
    data = {
        "history_list": st.session_state.get("history_list", []),
        "review_list": st.session_state.get("review_list", []),
        "multi_expr_list_2d": st.session_state.get("multi_expr_list_2d", ["x^2 - 4x", "2x + 1"]),
        "multi_expr_list_3d": st.session_state.get("multi_expr_list_3d", ["(3x - 4y)^3", "x^2 + y^2"])
    }
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

saved_data = load_user_data()

if "review_list" not in st.session_state:
    st.session_state.review_list = saved_data.get("review_list", [])
if "history_list" not in st.session_state:
    st.session_state.history_list = saved_data.get("history_list", [])
if "multi_expr_list_2d" not in st.session_state:
    st.session_state.multi_expr_list_2d = saved_data.get("multi_expr_list_2d", ["x^2 - 4x", "2x + 1"])
if "multi_expr_list_3d" not in st.session_state:
    st.session_state.multi_expr_list_3d = saved_data.get("multi_expr_list_3d", ["(3x - 4y)^3", "x^2 + y^2"])

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

# ==========================================
# 数式補正 ＆ 関数の抽出・グラフ描画ヘルパー
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

            fig.add_trace(go.Scatter(x=x_neg, y=y_neg, mode="lines", name=f"y = {expr_str}", line=dict(color="blue", width=2.5)))
            fig.add_trace(go.Scatter(x=x_pos, y=y_pos, mode="lines", name=f"y = {expr_str}", line=dict(color="blue", width=2.5), showlegend=False))
        else:
            x_vals = np.linspace(-10, 10, 500)
            y_vals = f_np(x_vals)
            if isinstance(y_vals, (int, float)):
                y_vals = np.full_like(x_vals, y_vals)
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name=f"y = {expr_str}", line=dict(color="red", width=2.5)))

        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")

        fig.update_layout(
            title=f"📈 {title}: y = {expr_str}",
            xaxis_title="X軸 (x)",
            yaxis_title="Y軸 (y)",
            xaxis=dict(range=[-10, 10], zeroline=True),
            yaxis=dict(range=[-10, 10], zeroline=True),
            margin=dict(l=20, r=20, t=40, b=20),
            height=320,
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

# ==========================================
# 2. API 呼び出し関数
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
添付画像（手書き途中式・解答）を視覚的に解析・採点してください。
問題: {problem.question}
正解: {problem.final_answer}
模範途中式: {json.dumps(problem.solution_steps, ensure_ascii=False)}

【指示】
1. 手書き文字（途中式と最終解答）を読み取って文章化してください。
2. 採点結果（正解/部分点/不正解）を判定してください。
3. 計算ミスや考え方の間違いを指導・アドバイスしてください。
"""
    response = generate_with_retry(prompt="", contents=[image, prompt])
    return response.text

def analyze_speed_handwritten_session(grade: str, level: int, speed_problems: list[SpeedProblem], images: list[Image.Image], total_sec: int) -> str:
    contents = []
    analysis_input_info = []
    for idx, (sp_p, img) in enumerate(zip(speed_problems, images), 1):
        analysis_input_info.append({
            "question_num": idx, "unit": sp_p.unit, "question": sp_p.question, "correct_answer": sp_p.final_answer
        })
        contents.append(f"--- Q{idx} の手書き解答画像 ---")
        contents.append(img)

    prompt = f"""
学年「{grade}」（難易度 Level {level}）のスピード手書き連答テスト（全{len(speed_problems)}問）の画像を視覚解析・採点してください。
所要時間: {total_sec // 60}分 {total_sec % 60}秒
【問題データ】{json.dumps(analysis_input_info, ensure_ascii=False, indent=2)}

【出力項目】
1. 各問題の手書き読み取り結果と正誤判定（および総合正解率）
2. 解答スピードと精度のバランス評価・復習アドバイス
"""
    contents.append(prompt)
    response = generate_with_retry(prompt="", contents=contents)
    return response.text

# ==========================================
# 3. サイドバー
# ==========================================
with st.sidebar:
    st.title("📱 数学AI タブレット")
    st.caption("左上の ☰ ボタンでメニューを閉じると、キャンバス領域を最大化できます。")
    st.markdown("---")
    
    app_mode = st.radio(
        "📌 機能を切り替え",
        [
            "✏️ モード1: 手書きじっくりテスト",
            "⚡ モード2: 手書きスピード連答",
            "📈 グラフ描画ツール",
            "🔄 復習チェックノート",
            "📈 学習履歴・過去レポート"
        ],
        index=0
    )
    st.markdown("---")
    
    if app_mode == "✏️ モード1: 手書きじっくりテスト":
        st.subheader("⚙️ モード1 条件設定")
        grade_m1 = st.selectbox("学年 (M1)", list(UNIT_DATABASE.keys()), index=0, key="sb_t_grade")
        unit_m1 = st.selectbox("単元 (M1)", UNIT_DATABASE.get(grade_m1, []), key="sb_t_unit")
        level_m1 = st.slider("難易度", 1, 10, 3, key="sb_t_lvl")
        count_m1 = st.number_input("問題数", min_value=1, max_value=10, value=3, step=1, key="sb_t_cnt")
        
        if st.button("🚀 テストを開始する", type="primary", use_container_width=True):
            st.session_state.tab_active = True
            st.session_state.t_grade_val = grade_m1
            st.session_state.t_unit_val = unit_m1
            st.session_state.t_level_val = level_m1
            st.session_state.t_total_count = count_m1
            st.session_state.t_current_idx = 0
            st.session_state.t_analyzed = False
            st.session_state.t_submitting = False
            
            with st.spinner(f"AIが全 {count_m1} 問を準備中..."):
                try:
                    st.session_state.t_problems = generate_problem_set(grade_m1, unit_m1, level_m1, count_m1)
                    st.session_state.t_start_time = time.time()
                    st.rerun()
                except Exception as err:
                    st.session_state.tab_active = False
                    st.error(f"問題作成に失敗しました: {err}")

    elif app_mode == "⚡ モード2: 手書きスピード連答":
        st.subheader("⚙️ モード2 条件設定")
        grade_sp = st.selectbox("学年 (M2)", list(UNIT_DATABASE.keys()), index=0, key="sb_sp_grade")
        level_sp = st.slider("難易度レベル", min_value=1, max_value=10, value=3, key="sb_sp_lvl")
        count_sp = st.select_slider("問題数", options=[5, 10, 15, 20, 30, 50], value=5, key="sb_sp_cnt")
        
        if st.button("🔥 スピード連答スタート", type="primary", use_container_width=True):
            st.session_state.sp_active = True
            st.session_state.sp_target_grade = grade_sp
            st.session_state.sp_target_level = level_sp
            st.session_state.sp_target_count = count_sp
            st.session_state.sp_current_idx = 0
            st.session_state.sp_user_images = []
            st.session_state.sp_finished = False
            st.session_state.sp_submitting = False
            
            with st.spinner(f"AIが問題（全 {count_sp} 問）を準備中..."):
                try:
                    st.session_state.sp_problems = generate_speed_problem_set(grade_sp, level_sp, count_sp)
                    st.session_state.sp_start_time = time.time()
                    st.rerun()
                except Exception as err:
                    st.session_state.sp_active = False
                    st.error(f"問題作成に失敗しました: {err}")

# ==========================================
# 4. メイン画面コンテンツ
# ==========================================

# ------------------------------------------
# メイン 1: モード1（手書きじっくりテスト）
# ------------------------------------------
if app_mode == "✏️ モード1: 手書きじっくりテスト":
    st.header("✏️ モード1: 手書きじっくりテスト")
    st.caption("左側のサイドバーで単元・条件を設定して「🚀 テストを開始する」を押してください。")
    st.markdown("---")

    if st.session_state.get("tab_active", False) and "t_problems" in st.session_state:
        problems: list[MathProblem] = st.session_state.t_problems
        cur_idx = st.session_state.t_current_idx
        tot_cnt = st.session_state.t_total_count
        prob = problems[cur_idx]
        
        is_submitted = st.session_state.get("t_analyzed", False)
        is_submitting = st.session_state.get("t_submitting", False)
        
        st.progress((cur_idx + 1) / tot_cnt)
        
        col_left, col_right = st.columns([1, 1], gap="medium")
        
        with col_left:
            st.markdown(f"### 【第 {cur_idx + 1} 問 / 全 {tot_cnt} 問】📌 {st.session_state.t_grade_val} - {st.session_state.t_unit_val}")
            st.subheader(prob.title)
            st.markdown("---")
            st.markdown(prob.question)

            # 比例・反比例単元の自動グラフ表示
            if "比例" in st.session_state.t_unit_val or "関数" in st.session_state.t_unit_val:
                sp_func = extract_function_from_text(prob.question) or extract_function_from_text(prob.final_answer)
                if sp_func:
                    with st.expander("📊 関連グラフを表示", expanded=True):
                        render_proportional_graph(sp_func, f"問 {cur_idx + 1} グラフ")

            st.markdown("---")
            
            with st.expander("💡 ヒントを見る"):
                st.write(prob.hint)
                
            if is_submitted:
                st.markdown("---")
                st.header("📊 AI手書き採点・添削結果")
                res = st.session_state.t_res
                st.metric("所要時間", f"{res['elapsed']}分")
                st.markdown("#### 🔍 添削指導アドバイス")
                st.write(res["analysis"])
                
                with st.expander("📖 模範解答ステップ"):
                    for idx_s, step in enumerate(prob.solution_steps, 1):
                        st.write(f"**Step {idx_s}:** {step}")
                    st.write(f"**最終正解:** {prob.final_answer}")
                
                already_in = any(item["question"] == prob.question for item in st.session_state.review_list)
                if already_in:
                    st.success("✅ 復習ノートに追加済みです")
                else:
                    if st.button("⭐ この問題を復習ノートに追加", key=f"t_add_rev_{cur_idx}"):
                        st.session_state.review_list.append({
                            "id": f"t_{prob.id}_{int(time.time())}",
                            "grade": st.session_state.t_grade_val,
                            "unit": st.session_state.t_unit_val,
                            "title": prob.title,
                            "question": prob.question,
                            "final_answer": prob.final_answer,
                            "solution_steps": prob.solution_steps,
                            "hint": prob.hint
                        })
                        save_user_data()  # JSON自動保存
                        st.toast("復習ノートに追加しました！", icon="📌")
                        st.rerun()

        with col_right:
            st.markdown("### ✏️ 手書き解答ノート")
            st.caption("途中式と最終解答をキャンバス内に手書きしてください。")
            
            is_locked = is_submitted or is_submitting
            if not is_locked:
                col_tool1, col_tool2 = st.columns(2)
                with col_tool1:
                    tool_choice = st.radio("ツール切替", ("pen", "eraser"), format_func=lambda x: "🖊️ ペン" if x=="pen" else "🧹 消しゴム", horizontal=True)
                with col_tool2:
                    stroke_width = st.slider("太さ", 1, 30, 3 if tool_choice == "pen" else 15)

                stroke_color = "#000000" if tool_choice == "pen" else "#ffffff"
                drawing_mode = "freedraw"
            else:
                stroke_color = "#000000"
                stroke_width = 3
                drawing_mode = "transform"

            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color="#ffffff",
                height=380,
                width=500,
                drawing_mode=drawing_mode,
                key=f"canvas_q_{cur_idx}_{prob.id}",
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if not is_submitted and not is_submitting:
                if st.button("📝 この手書き解答を提出してAI採点", type="primary", use_container_width=True):
                    if canvas_result.image_data is None:
                        st.warning("キャンバスに手書きで解答を書いてください。")
                    else:
                        st.session_state.t_submitting = True
                        st.rerun()

            if is_submitting and not is_submitted:
                elapsed = round((time.time() - st.session_state.t_start_time) / 60, 1)
                with st.spinner("AIが手書きノートを画像解析中..."):
                    try:
                        img_data = canvas_result.image_data
                        img = Image.fromarray(img_data.astype('uint8'), 'RGBA').convert('RGB')
                        an_txt = analyze_handwritten_image(prob, img)
                        
                        st.session_state.t_analyzed = True
                        st.session_state.t_submitting = False
                        st.session_state.t_res = {"elapsed": elapsed, "analysis": an_txt}
                        
                        st.session_state.history_list.append({
                            "id": f"h_t_{int(time.time())}",
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "mode": "モード1 (じっくり手書き)",
                            "title": f"第{cur_idx+1}/{tot_cnt}問: {st.session_state.t_grade_val} - {st.session_state.t_unit_val}",
                            "elapsed": f"{elapsed}分",
                            "analysis": an_txt
                        })
                        save_user_data()  # JSON自動保存
                        st.rerun()
                    except Exception as e:
                        st.session_state.t_submitting = False
                        st.error(f"解析処理に失敗しました: {e}")

            if is_submitted:
                if cur_idx < tot_cnt - 1:
                    if st.button(f"次へ進む（第 {cur_idx + 2} 問 / 全 {tot_cnt} 問）", type="primary", use_container_width=True):
                        st.session_state.t_current_idx += 1
                        st.session_state.t_analyzed = False
                        st.session_state.t_submitting = False
                        st.session_state.t_start_time = time.time()
                        st.rerun()
                else:
                    st.success("🎉 全問題が終了しました！")
                    if st.button("🔄 もう一度設定して解く", use_container_width=True):
                        st.session_state.tab_active = False
                        st.rerun()

# ------------------------------------------
# メイン 2: モード2（手書きスピード連答）
# ------------------------------------------
elif app_mode == "⚡ モード2: 手書きスピード連答":
    st.header("⚡ モード2: 手書きスピード連答")
    st.caption("左側のサイドバーで学年・難易度・問題数を設定して「🔥 スピード連答スタート」を押してください。")
    st.markdown("---")

    if st.session_state.get("sp_active", False) and not st.session_state.get("sp_finished", False) and "sp_problems" in st.session_state:
        sp_problems: list[SpeedProblem] = st.session_state.sp_problems
        cur_i = st.session_state.sp_current_idx
        tot_c = st.session_state.sp_target_count
        cur_p = sp_problems[cur_i]
        
        is_submitting = st.session_state.get("sp_submitting", False)

        st.progress((cur_i + 1) / tot_c)
        st.caption(f"問題 {cur_i + 1} / {tot_c}  |  該当単元: 【{cur_p.unit}】  |  難易度: Level {st.session_state.sp_target_level}")
        
        elapsed_sec = int(time.time() - st.session_state.sp_start_time)
        
        col_q, col_can = st.columns([1, 1], gap="medium")
        
        with col_q:
            st.markdown(f"## **Q{cur_i + 1}. {cur_p.question}**")

            # 比例・反比例問題の自動グラフ表示
            if "比例" in cur_p.unit or "関数" in cur_p.unit:
                sp_func = extract_function_from_text(cur_p.question) or extract_function_from_text(cur_p.final_answer)
                if sp_func:
                    with st.expander("📊 グラフを表示", expanded=True):
                        render_proportional_graph(sp_func, f"Q{cur_i + 1} 関連グラフ")

            st.markdown("---")
            st.caption(f"⏱️ 全体経過時間: {elapsed_sec // 60}分 {elapsed_sec % 60}秒")
            st.info("💡 右のキャンバスに答えを手書きしてください。")

        with col_can:
            st.markdown("### ✏️ 手書き解答枠")
            
            if not is_submitting:
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    tool_choice = st.radio("ツール", ("pen", "eraser"), format_func=lambda x: "🖊️ ペン" if x=="pen" else "🧹 消しゴム", horizontal=True, key=f"sp_tool_{cur_i}")
                with col_t2:
                    stroke_width = st.slider("太さ", 1, 30, 3 if tool_choice == "pen" else 15, key=f"sp_width_{cur_i}")

                stroke_color = "#000000" if tool_choice == "pen" else "#ffffff"
                drawing_mode = "freedraw"
            else:
                stroke_color = "#000000"
                stroke_width = 3
                drawing_mode = "transform"

            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color="#ffffff",
                height=300,
                width=450,
                drawing_mode=drawing_mode,
                key=f"sp_canvas_{cur_i}_{cur_p.id}",
            )
            
            btn_label = "次の問題へ ➔" if cur_i < tot_c - 1 else "🏁 全問提出・全手書き画像を総合分析"
            if not is_submitting:
                if st.button(btn_label, type="primary", use_container_width=True):
                    if canvas_result.image_data is None:
                        st.warning("手書きで答えを記入してください。")
                    else:
                        st.session_state.sp_submitting = True
                        st.rerun()

            if is_submitting:
                img_data = canvas_result.image_data
                img = Image.fromarray(img_data.astype('uint8'), 'RGBA').convert('RGB')
                st.session_state.sp_user_images.append(img)
                
                st.session_state.sp_submitting = False
                if cur_i < tot_c - 1:
                    st.session_state.sp_current_idx += 1
                    st.rerun()
                else:
                    st.session_state.sp_total_elapsed_sec = int(time.time() - st.session_state.sp_start_time)
                    st.session_state.sp_finished = True
                    st.rerun()

    elif st.session_state.get("sp_finished", False):
        st.balloons()
        sp_problems = st.session_state.sp_problems
        images = st.session_state.sp_user_images
        target_grade = st.session_state.sp_target_grade
        target_lvl = st.session_state.sp_target_level
        tot_c = st.session_state.sp_target_count
        tot_sec = st.session_state.sp_total_elapsed_sec
        
        st.header(f"🎉 スピード手書き連答終了！ ({target_grade} - Level {target_lvl} 全 {tot_c} 問)")
        st.caption(f"合計時間: {tot_sec // 60}分 {tot_sec % 60}秒  |  平均速度: {round(tot_sec / tot_c, 1)}秒/問")
        st.markdown("---")
        
        with st.spinner("AIが全問の手書き画像を視覚解析・採点中..."):
            if "sp_analysis_txt" not in st.session_state:
                an_txt = analyze_speed_handwritten_session(target_grade, target_lvl, sp_problems, images, tot_sec)
                st.session_state.sp_analysis_txt = an_txt
                
                st.session_state.history_list.append({
                    "id": f"h_sp_{int(time.time())}",
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "mode": "モード2 (スピード手書き連答)",
                    "title": f"学年: {target_grade} (Level {target_lvl} / 全{tot_c}問)",
                    "elapsed": f"{tot_sec // 60}分 {tot_sec % 60}秒",
                    "analysis": an_txt
                })
                save_user_data()  # JSON自動保存
            
        st.markdown("### 📊 AI手書き一括解析・総合分析レポート")
        st.write(st.session_state.sp_analysis_txt)
        st.markdown("---")
        
        if st.button("🚀 もう一度スピード連答に挑戦する", type="primary", use_container_width=True):
            st.session_state.sp_active = False
            st.session_state.sp_finished = False
            if "sp_analysis_txt" in st.session_state:
                del st.session_state["sp_analysis_txt"]
            st.rerun()

# ------------------------------------------
# メイン 3: 2D/3Dマルチ重ね描き＆自由軸設定ツール
# ------------------------------------------
elif app_mode == "📈 グラフ描画ツール":
    st.header("📈 マルチ数式グラフ描画ツール (2D / 3D)")
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
            x_min = col_x1.number_input("X軸 最小値", value=-10.0, step=1.0, key="t_2d_xmin")
            x_max = col_x2.number_input("X軸 最大値", value=10.0, step=1.0, key="t_2d_xmax")

            col_y1, col_y2 = st.columns(2)
            y_min = col_y1.number_input("Y軸 最小値", value=-10.0, step=1.0, key="t_2d_ymin")
            y_max = col_y2.number_input("Y軸 最大値", value=10.0, step=1.0, key="t_2d_ymax")

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

                fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
                fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
                fig.update_layout(
                    xaxis_title="X軸 (x)", yaxis_title="Y軸 (y)",
                    xaxis=dict(range=[x_min, x_max], zeroline=True),
                    yaxis=dict(range=[y_min, y_max], zeroline=True),
                    margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified", height=550,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.7)")
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
            x3_min = col_x1.number_input("X軸 最小値", value=-1.5, step=0.5, key="t_3d_xmin")
            x3_max = col_x2.number_input("X軸 最大値", value=1.5, step=0.5, key="t_3d_xmax")

            col_y1, col_y2 = st.columns(2)
            y3_min = col_y1.number_input("Y軸 最小値", value=-1.5, step=0.5, key="t_3d_ymin")
            y3_max = col_y2.number_input("Y軸 最大値", value=1.5, step=0.5, key="t_3d_ymax")

            col_z1, col_z2 = st.columns(2)
            z3_min = col_z1.number_input("Z軸 最小値", value=-10.0, step=1.0, key="t_3d_zmin")
            z3_max = col_z2.number_input("Z軸 最大値", value=10.0, step=1.0, key="t_3d_zmax")

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
                    scene=dict(
                        xaxis=dict(title="X軸", range=[x3_min, x3_max]),
                        yaxis=dict(title="Y軸", range=[y3_min, y3_max]),
                        zaxis=dict(title="Z軸", range=[z3_min, z3_max]),
                    ),
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=600,
                    showlegend=True
                )
                st.plotly_chart(fig_3d, use_container_width=True)

# ------------------------------------------
# メイン 4: 復習チェックノート
# ------------------------------------------
elif app_mode == "🔄 復習チェックノート":
    st.header("📚 復習チェック・解き直しノート")
    st.caption("保存した弱点問題の復習ができます。")
    if not st.session_state.review_list:
        st.info("復習ノートに保存された問題はありません。")
    else:
        st.write(f"現在 **{len(st.session_state.review_list)} 問** の復習問題があります。")
        if st.button("🗑️ 復習ノートをクリア"):
            st.session_state.review_list = []
            save_user_data()  # JSON自動保存
            st.rerun()
        st.markdown("---")
        for rev_idx, r_item in enumerate(list(st.session_state.review_list), 1):
            st.markdown(f"### 📌 復習問 {rev_idx}: [{r_item['grade']} / {r_item['unit']}] {r_item['title']}")
            st.markdown(r_item["question"])
            with st.expander("💡 ヒントを見る"):
                st.write(r_item["hint"])
            rev_final = st.text_input("再チャレンジ解答", key=f"rev_f_{r_item['id']}")
            col_r1, col_r2, col_r3 = st.columns([1, 1, 1])
            with col_r1:
                if st.button(f"正解判定 (問 {rev_idx})", key=f"btn_chk_{r_item['id']}"):
                    if not rev_final:
                        st.warning("解答を入力してください。")
                    else:
                        is_ok = check_final_answer_sympy(rev_final, r_item["final_answer"])
                        if is_ok:
                            st.success(f"🎉 正解！ 正解: {r_item['final_answer']}")
                        else:
                            st.error(f"❌ 残念！ 正解: {r_item['final_answer']}")
            with col_r2:
                with st.expander("📖 模範途中式を見る"):
                    for s_idx, step in enumerate(r_item["solution_steps"], 1):
                        st.write(f"Step {s_idx}: {step}")
            with col_r3:
                if st.button(f"✅ 克服（削除）", key=f"btn_del_{r_item['id']}"):
                    st.session_state.review_list.pop(rev_idx - 1)
                    save_user_data()  # JSON自動保存
                    st.toast("削除しました！")
                    st.rerun()
            st.markdown("---")

# ------------------------------------------
# メイン 5: 学習履歴・過去レポート
# ------------------------------------------
elif app_mode == "📈 学習履歴・過去レポート":
    st.header("📈 学習履歴・過去レポート")
    st.caption("モード1・モード2の手書きテスト履歴がすべて JSON に自動記録されます。")
    if not st.session_state.history_list:
        st.info("まだ学習履歴がありません。")
    else:
        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            st.write(f"累計実施回数: **{len(st.session_state.history_list)} 回**")
        with col_h2:
            if st.button("🗑️ 履歴をすべて消去"):
                st.session_state.history_list = []
                save_user_data()  # JSON自動保存
                st.rerun()
        st.markdown("---")
        for h_item in reversed(st.session_state.history_list):
            with st.expander(f"🗓️ [{h_item['date']}] {h_item['mode']} : {h_item['title']} (所要時間: {h_item['elapsed']})"):
                st.write(f"**形式:** {h_item['mode']}")
                st.write(f"**対象:** {h_item['title']}")
                st.write(f"**所要時間:** {h_item['elapsed']}")
                st.markdown("**【AI添削分析レポート】**")
                st.write(h_item["analysis"])