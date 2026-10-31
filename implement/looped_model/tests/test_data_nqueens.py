"""Test M1 (PLAN_TABLE2.md M1). Chạy bằng pytest, hoặc `python3 tests/test_data_nqueens.py`.

Đối chiếu lf/data_nqueens.py với bộ liệt kê tham chiếu độc lập
(tests/golden_nqueens.py) và các giá trị chuẩn trong plan.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lf import data_nqueens as dn  # noqa: E402
from tests import golden_nqueens as golden  # noqa: E402


DATA = dn.build_dataset(n=8, hide_counts=(5, 6, 7), test_frac=0.15, split_seed=0)


def test_num_solutions():
    assert DATA.solutions.shape[0] == 92


def test_num_rows():
    assert DATA.x1.shape[0] == 8464


def test_num_unique_inputs():
    assert DATA.num_inputs == 5148


def test_avg_solutions_per_input():
    assert abs(DATA.input_num_rows.mean() - 1.644133644133644) < 1e-9


def test_max_solutions_per_input():
    assert int(DATA.input_num_rows.max()) == 18


def test_frac_single_solution():
    assert abs(float((DATA.input_num_rows == 1).mean()) - 0.655011655011655) < 1e-9


def test_inputs_by_clue_size():
    by_size = {1: 0, 2: 0, 3: 0}
    for cells in DATA.input_clue_cells:
        by_size[len(cells)] += 1
    assert by_size == {1: 64, 2: 1052, 3: 4032}


def test_matches_independent_reference():
    stats = golden.golden_stats(n=8, hide_counts=(5, 6, 7))
    assert stats["num_solutions"] == DATA.solutions.shape[0]
    assert stats["num_rows"] == DATA.x1.shape[0]
    assert stats["num_unique_inputs"] == DATA.num_inputs
    assert stats["max_solutions_per_input"] == int(DATA.input_num_rows.max())


def test_compatible_set_equals_row_count():
    # |S(c)| = số dòng của input c (PLAN M1).
    for iid in range(DATA.num_inputs):
        assert len(DATA.input_solutions[iid]) == int(DATA.input_num_rows[iid])


def test_split_disjoint_by_input():
    assert np.intersect1d(DATA.train_inputs, DATA.test_inputs).size == 0
    train_iids = set(DATA.input_id[DATA.train_rows].tolist())
    test_iids = set(DATA.input_id[DATA.test_rows].tolist())
    assert train_iids.isdisjoint(test_iids)


def test_test_row_fraction_near_gram():
    frac = DATA.test_rows.size / DATA.x1.shape[0]
    assert 0.13 <= frac <= 0.16  # GRAM: 1240/8464 ≈ 0.1465 (DEV-02: lệch nhẹ)


def test_encoding_values_and_length():
    # Mọi lời giải: dài 64, giá trị trong {1,2} (0 chỉ dành cho pad).
    assert DATA.solutions.shape[1] == 64
    vals = set(np.unique(DATA.solutions).tolist())
    assert vals == {dn.EMPTY_ID, dn.QUEEN_ID}
    assert set(np.unique(DATA.x1).tolist()) == {dn.EMPTY_ID, dn.QUEEN_ID}


def test_encode_decode_roundtrip():
    sols = dn.enumerate_solutions(8)
    for cols in sols[:20]:
        tokens = dn.board_to_tokens(cols, 8)
        assert dn.tokens_to_cols(tokens, 8) == list(cols)


def test_each_row_label_valid_and_matches_clue():
    # Mỗi dòng: nhãn là bàn hợp lệ và quân clue nằm trên nhãn.
    rng = np.random.default_rng(0)
    sample = rng.choice(DATA.x1.shape[0], size=300, replace=False)
    for ridx in sample:
        tok = DATA.x1[ridx]
        assert int((tok == dn.QUEEN_ID).sum()) == 8
        clue = DATA.input_clue_cells[int(DATA.input_id[ridx])]
        assert np.all(tok[clue] == dn.QUEEN_ID)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"\n{len(fns)} test đạt.")
