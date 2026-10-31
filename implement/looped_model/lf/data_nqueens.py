"""M1 — Bộ sinh dữ liệu N-Queens 8x8 (PLAN_TABLE2.md M1, SPEC_TABLE2.md §2).

Chỉ phụ thuộc numpy (chạy được không cần GPU/torch). Dựng:
  - mọi lời giải N-Queens đầy đủ,
  - mọi dòng (lời giải, tập quân clue được giữ) với số quân ẩn trong {5,6,7},
  - đăng ký input duy nhất và tập lời giải tương thích S(c),
  - chia 85:15 theo input duy nhất với một hạt giống cố định.

Từ vựng |V|=3 (D3): 0=pad, 1=empty, 2=queen. Bàn N x N phẳng row-major thành
chuỗi dài N*N. Ô cố định (clue) chỉ là ô có quân clue (D4). Mẫu số coverage
|S(c)| bằng số dòng của input c (PLAN M1 — đã kiểm bằng liệt kê).

Các con số chuẩn KHÔNG lấy từ bài; chúng do liệt kê sinh ra và được test đối
chiếu trong tests/test_data_nqueens.py.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np

PAD_ID = 0
EMPTY_ID = 1
QUEEN_ID = 2


# --------------------------------------------------------------------------- #
# Liệt kê và mã hoá
# --------------------------------------------------------------------------- #
def enumerate_solutions(n: int = 8) -> list[tuple[int, ...]]:
    """Mọi lời giải N-Queens, mỗi lời giải là tuple cột theo từng hàng."""
    sols: list[tuple[int, ...]] = []
    cols = [-1] * n

    def safe(row: int, col: int) -> bool:
        for pr in range(row):
            pc = cols[pr]
            if pc == col or abs(pc - col) == row - pr:
                return False
        return True

    def backtrack(row: int) -> None:
        if row == n:
            sols.append(tuple(cols))
            return
        for col in range(n):
            if safe(row, col):
                cols[row] = col
                backtrack(row + 1)
                cols[row] = -1

    backtrack(0)
    return sols


def board_to_tokens(cols: tuple[int, ...], n: int = 8) -> np.ndarray:
    """Lời giải (cột theo hàng) -> chuỗi token dài n*n, giá trị trong {1,2}."""
    tokens = np.full(n * n, EMPTY_ID, dtype=np.int8)
    for r, c in enumerate(cols):
        tokens[r * n + c] = QUEEN_ID
    return tokens


def tokens_to_cols(tokens: np.ndarray, n: int = 8) -> list[int]:
    """Giải mã chuỗi token thành danh sách cột quân theo hàng.

    Trả về cột quân của mỗi hàng; -1 nếu hàng không có đúng một quân. Dùng cho
    kiểm tra khôi phục và cho metrics. KHÔNG tự xác nhận tính hợp lệ ở đây.
    """
    grid = tokens.reshape(n, n)
    cols: list[int] = []
    for r in range(n):
        qpos = np.flatnonzero(grid[r] == QUEEN_ID)
        cols.append(int(qpos[0]) if qpos.size == 1 else -1)
    return cols


def clue_cells_to_mask(clue_cells: np.ndarray, seq_len: int) -> np.ndarray:
    """Chỉ số ô clue (phẳng) -> mặt nạ boolean dài seq_len (True = cố định)."""
    mask = np.zeros(seq_len, dtype=bool)
    mask[clue_cells] = True
    return mask


# --------------------------------------------------------------------------- #
# Cấu trúc dữ liệu
# --------------------------------------------------------------------------- #
@dataclass
class NQueensData:
    n: int
    seq_len: int
    solutions: np.ndarray            # (S, L) int8, token các lời giải
    x1: np.ndarray                   # (R, L) int8, token lời giải đích của từng dòng
    clue_mask: np.ndarray            # (R, L) bool, True tại ô clue (cố định)
    input_id: np.ndarray             # (R,) int32, id input duy nhất của từng dòng
    solution_idx: np.ndarray         # (R,) int32, chỉ số lời giải đích của từng dòng
    input_clue_cells: list[np.ndarray]   # theo input_id: chỉ số ô clue (phẳng)
    input_solutions: list[np.ndarray]    # theo input_id: chỉ số lời giải tương thích S(c)
    input_num_rows: np.ndarray       # (num_inputs,) số dòng của mỗi input
    train_rows: np.ndarray           # chỉ số dòng thuộc train
    test_rows: np.ndarray            # chỉ số dòng thuộc test
    train_inputs: np.ndarray         # id input thuộc train
    test_inputs: np.ndarray          # id input thuộc test

    @property
    def num_inputs(self) -> int:
        return len(self.input_clue_cells)


def build_dataset(
    n: int = 8,
    hide_counts: tuple[int, ...] = (5, 6, 7),
    test_frac: float = 0.15,
    split_seed: int = 0,
) -> NQueensData:
    """Dựng toàn bộ dữ liệu N-Queens và chia train/test theo input duy nhất."""
    seq_len = n * n
    solutions = enumerate_solutions(n)
    sol_tokens = np.stack([board_to_tokens(s, n) for s in solutions])  # (S, L)
    keep_counts = sorted(n - h for h in hide_counts)

    # Gom dòng theo clue (phẳng, đã sắp) để id input ổn định, không phụ thuộc
    # thứ tự liệt kê.
    clue_to_sols: dict[tuple[int, ...], list[int]] = defaultdict(list)
    clue_rows: dict[tuple[int, ...], list[int]] = defaultdict(list)
    row_sol: list[int] = []
    row_clue: list[tuple[int, ...]] = []
    for si, sol in enumerate(solutions):
        flat_queens = [r * n + sol[r] for r in range(n)]
        for keep in keep_counts:
            for sub in combinations(flat_queens, keep):
                clue = tuple(sorted(sub))
                ridx = len(row_sol)
                row_sol.append(si)
                row_clue.append(clue)
                clue_rows[clue].append(ridx)
                clue_to_sols[clue].append(si)

    unique_clues = sorted(clue_to_sols.keys())
    clue_to_id = {clue: i for i, clue in enumerate(unique_clues)}
    num_rows = len(row_sol)

    x1 = sol_tokens[np.asarray(row_sol, dtype=np.int64)]            # (R, L)
    input_id = np.array([clue_to_id[c] for c in row_clue], dtype=np.int32)
    solution_idx = np.asarray(row_sol, dtype=np.int32)

    clue_mask = np.zeros((num_rows, seq_len), dtype=bool)
    for ridx, clue in enumerate(row_clue):
        clue_mask[ridx, list(clue)] = True

    input_clue_cells = [np.asarray(c, dtype=np.int32) for c in unique_clues]
    input_solutions = [
        np.asarray(sorted(set(clue_to_sols[c])), dtype=np.int32) for c in unique_clues
    ]
    input_num_rows = np.array([len(clue_rows[c]) for c in unique_clues], dtype=np.int32)

    # Chia theo input duy nhất (D9, D10): xáo id input rồi cắt 15% làm test.
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(unique_clues))
    n_test = int(round(test_frac * len(unique_clues)))
    test_inputs = np.sort(perm[:n_test]).astype(np.int32)
    train_inputs = np.sort(perm[n_test:]).astype(np.int32)
    is_test_input = np.zeros(len(unique_clues), dtype=bool)
    is_test_input[test_inputs] = True
    row_is_test = is_test_input[input_id]
    test_rows = np.flatnonzero(row_is_test).astype(np.int32)
    train_rows = np.flatnonzero(~row_is_test).astype(np.int32)

    return NQueensData(
        n=n,
        seq_len=seq_len,
        solutions=sol_tokens,
        x1=x1,
        clue_mask=clue_mask,
        input_id=input_id,
        solution_idx=solution_idx,
        input_clue_cells=input_clue_cells,
        input_solutions=input_solutions,
        input_num_rows=input_num_rows,
        train_rows=train_rows,
        test_rows=test_rows,
        train_inputs=train_inputs,
        test_inputs=test_inputs,
    )


if __name__ == "__main__":
    data = build_dataset()
    print("solutions      :", data.solutions.shape[0])
    print("rows           :", data.x1.shape[0])
    print("unique inputs  :", data.num_inputs)
    avg = data.input_num_rows.mean()
    print("avg sols/input : %.4f" % avg)
    print("max sols/input :", int(data.input_num_rows.max()))
    frac1 = float((data.input_num_rows == 1).mean())
    print("frac single    : %.4f" % frac1)
    print("train/test rows:", data.train_rows.size, "/", data.test_rows.size)
    print("test row frac  : %.4f" % (data.test_rows.size / data.x1.shape[0]))
