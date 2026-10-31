"""Test M2 (PLAN_TABLE2.md M2). Chạy bằng pytest, hoặc `python3 tests/test_metrics.py`."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lf import data_nqueens as dn  # noqa: E402
from lf import metrics as mt  # noqa: E402


SOLS = dn.enumerate_solutions(8)
SOL_TOKENS = np.stack([dn.board_to_tokens(s, 8) for s in SOLS])


def test_all_real_solutions_valid():
    for tok in SOL_TOKENS:
        assert mt.is_valid_board(tok, clue_cells=None, n=8)


def test_board_with_seven_queens_invalid():
    tok = SOL_TOKENS[0].copy()
    first_queen = int(np.flatnonzero(tok == dn.QUEEN_ID)[0])
    tok[first_queen] = dn.EMPTY_ID  # còn 7 quân
    assert not mt.is_valid_board(tok, n=8)


def test_two_queens_same_diagonal_invalid():
    # Bàn 8 quân mỗi hàng một quân nhưng có xung đột chéo.
    tok = np.full(64, dn.EMPTY_ID, dtype=np.int8)
    cols = [0, 1, 2, 3, 4, 5, 6, 7]  # đường chéo chính: mọi cặp xung đột
    for r, c in enumerate(cols):
        tok[r * 8 + c] = dn.QUEEN_ID
    assert int((tok == dn.QUEEN_ID).sum()) == 8
    assert not mt.is_valid_board(tok, n=8)


def test_clue_violation_invalid():
    # Lời giải hợp lệ nhưng không khớp clue -> không hợp lệ (D1).
    sol = SOL_TOKENS[0]
    other = SOL_TOKENS[1]
    # Ô clue lấy từ 'other' mà sol không có quân ở đó.
    q_other = set(np.flatnonzero(other == dn.QUEEN_ID).tolist())
    q_sol = set(np.flatnonzero(sol == dn.QUEEN_ID).tolist())
    diff = sorted(q_other - q_sol)
    clue = np.array(diff[:1], dtype=np.int64)
    assert mt.is_valid_board(sol, clue_cells=None, n=8)
    assert not mt.is_valid_board(sol, clue_cells=clue, n=8)


def _input_with_k_solutions(k: int):
    data = dn.build_dataset()
    iid = int(np.flatnonzero(data.input_num_rows == k)[0])
    clue = data.input_clue_cells[iid]
    sol_idx = data.input_solutions[iid]
    sol_boards = data.solutions[sol_idx]
    return clue, len(sol_idx), sol_boards


def test_coverage_full_set():
    clue, ncomp, boards = _input_with_k_solutions(3)
    cov = mt.coverage_for_input(boards, clue, ncomp, n=8)
    assert abs(cov - 1.0) < 1e-12


def test_coverage_half_set():
    clue, ncomp, boards = _input_with_k_solutions(4)
    cov = mt.coverage_for_input(boards[:2], clue, ncomp, n=8)
    assert abs(cov - 0.5) < 1e-12


def test_coverage_duplicates_count_once():
    clue, ncomp, boards = _input_with_k_solutions(3)
    repeated = np.repeat(boards[:1], 20, axis=0)  # cùng một lời giải 20 lần
    cov = mt.coverage_for_input(repeated, clue, ncomp, n=8)
    assert abs(cov - 1.0 / ncomp) < 1e-12


def test_coverage_invalid_samples_ignored():
    clue, ncomp, boards = _input_with_k_solutions(3)
    bad = np.full((5, 64), dn.EMPTY_ID, dtype=np.int8)  # không có quân, không hợp lệ
    samples = np.concatenate([boards[:1], bad], axis=0)
    cov = mt.coverage_for_input(samples, clue, ncomp, n=8)
    assert abs(cov - 1.0 / ncomp) < 1e-12


def test_coverage_valid_wrong_clue_not_counted():
    # Mẫu là lời giải hợp lệ nhưng không chứa clue -> không tính (D1).
    data = dn.build_dataset()
    iid = int(np.flatnonzero(data.input_num_rows == 1)[0])
    clue = data.input_clue_cells[iid]
    compatible = set(data.input_solutions[iid].tolist())
    incompatible = [i for i in range(len(SOLS)) if i not in compatible]
    sample = SOL_TOKENS[incompatible[:3]]
    cov = mt.coverage_for_input(sample, clue, 1, n=8)
    assert cov == 0.0


def test_two_aggregations_equal_when_single_solution():
    # Khi mọi input chỉ có một lời giải, hai cách gộp trùng nhau (cổng M2).
    per_input = {10: 1.0, 11: 0.0, 12: 1.0}
    row_ids = np.array([10, 10, 11, 12, 12, 12])  # input lặp theo dòng
    agg = mt.aggregate_coverage(per_input, row_ids)
    # by_input = mean(1,0,1)=2/3; by_row = mean(1,1,0,1,1,1)=5/6 -> khác nhau khi lặp.
    assert abs(agg["by_unique_input"] - 2 / 3) < 1e-12
    assert abs(agg["by_row"] - 5 / 6) < 1e-12
    # Khi không có input lặp (mỗi input một dòng) hai cách bằng nhau:
    agg2 = mt.aggregate_coverage(per_input, np.array([10, 11, 12]))
    assert abs(agg2["by_unique_input"] - agg2["by_row"]) < 1e-12


def test_first_sample_accuracy():
    data = dn.build_dataset()
    iids = data.test_inputs[:50]
    clues = [data.input_clue_cells[int(i)] for i in iids]
    # Mẫu đầu = một lời giải tương thích -> hợp lệ; accuracy 1.0.
    firsts = np.stack([data.solutions[data.input_solutions[int(i)][0]] for i in iids])
    assert abs(mt.first_sample_accuracy(firsts, clues, n=8) - 1.0) < 1e-12
    # Thay một mẫu đầu bằng bàn rỗng -> accuracy giảm đúng 1/50.
    firsts[0] = dn.EMPTY_ID
    assert abs(mt.first_sample_accuracy(firsts, clues, n=8) - 49 / 50) < 1e-12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"\n{len(fns)} test đạt.")
