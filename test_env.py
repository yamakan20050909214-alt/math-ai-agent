import torch
import sympy as sp
from z3 import Solver, Ints, sat

# 1. PyTorch & GPUチェック
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# 2. SymPy (代数計算): 方程式 x^2 - 5x + 6 = 0 を解く
x = sp.Symbol('x')
solutions = sp.solve(sp.Eq(x**2 - 5*x + 6, 0), x)
print(f"SymPy Solution (x^2 - 5x + 6 = 0): x = {solutions}")

# 3. Z3 (論理ソルバー): x + y = 10 且つ x - y = 2 の解
x_z3, y_z3 = Ints('x y')
s = Solver()
s.add(x_z3 + y_z3 == 10, x_z3 - y_z3 == 2)
if s.check() == sat:
    m = s.model()
    print(f"Z3 Solution: x = {m[x_z3]}, y = {m[y_z3]}")
