import sympy as sp

def solve_word_problem(equation_str, variable_str):
    """
    文章題から抽出した方程式をSymPyで解き、ステップ解説を生成する関数
    """
    x = sp.Symbol(variable_str)
    
    # 渡された文字列（例: "2*x + 5 - 15"）をSymPy方程式（2*x + 5 = 15）に変換
    lhs_str, rhs_str = equation_str.split('=')
    eq = sp.Eq(sp.sympify(lhs_str), sp.sympify(rhs_str))
    
    # 厳密解を計算
    solutions = sp.solve(eq, x)
    
    # 出力フォーマットの構築
    print("=== 数学AI計算プロセス ===")
    print(f"【解析した方程式】: {eq}")
    print(f"【途中式・変形】  : {sp.simplify(eq.lhs - eq.rhs)} = 0")
    print(f"【算出された解】  : {variable_str} = {solutions}")
    return solutions

# テスト実行（例: 「ある数を2倍して5を足すと15になる。ある数は？」）
solve_word_problem("2*x + 5 = 15", "x")