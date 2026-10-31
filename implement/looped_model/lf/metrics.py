"""M2 — Metric cho N-Queens (PLAN_TABLE2.md M2, SPEC_TABLE2.md §5).

Chỉ phụ thuộc numpy. Gồm:
  - is_valid_board: đủ N quân, không tấn công, và khớp clue (D1).
  - coverage cho một input: số lời giải hợp lệ khác nhau / |S(c)| (§5.1).
  - hai cách gộp coverage (D2): theo input duy nhất và theo dòng.
  - accuracy: chỉ xét mẫu đầu tiên trong 20 mẫu (§5.2 N-Queens).

"Hợp lệ" ở đây đòi khớp clue (D1, đã duyệt). Mẫu số |S(c)| do bộ dữ liệu cung
cấp (= số dòng của input, PLAN M1). Khử trùng lặp bằng khớp token chính xác sau
khi làm tròn.
"""
from __future__ import annotations

import numpy as np

QUEEN_ID = 2


def is_valid_board(tokens: np.ndarray, clue_cells: np.ndarray | None = None, n: int = 8) -> bool:
    """True nếu bàn là lời giải N-Queens hợp lệ và khớp clue.

    tokens: chuỗi dài n*n đã làm tròn (giá trị nguyên; quân = QUEEN_ID).
    clue_cells: chỉ số (phẳng) các ô clue; nếu nêu thì mọi ô đó phải là quân.
    """
    tokens = np.asarray(tokens).reshape(-1)
    if tokens.size != n * n:
        return False
    queen_flat = np.flatnonzero(tokens == QUEEN_ID)
    if queen_flat.size != n:
        return False
    rows = queen_flat // n
    cols = queen_flat % n
    # Một quân mỗi hàng, một quân mỗi cột, không trùng hai đường chéo.
    if len(set(rows.tolist())) != n:
        return False
    if len(set(cols.tolist())) != n:
        return False
    if len(set((rows - cols).tolist())) != n:
        return False
    if len(set((rows + cols).tolist())) != n:
        return False
    if clue_cells is not None and len(clue_cells) > 0:
        if not np.all(tokens[np.asarray(clue_cells)] == QUEEN_ID):
            return False
    return True


def board_key(tokens: np.ndarray) -> bytes:
    """Khoá khử trùng lặp: byte của chuỗi token đã làm tròn."""
    return np.asarray(tokens).reshape(-1).astype(np.int16).tobytes()


def coverage_for_input(
    sample_tokens: np.ndarray,
    clue_cells: np.ndarray,
    num_compatible: int,
    n: int = 8,
) -> float:
    """Coverage của một input: |lời giải hợp lệ khác nhau| / |S(c)|.

    sample_tokens: (num_samples, n*n) các mẫu đã làm tròn cho cùng một input.
    num_compatible: |S(c)|, do bộ dữ liệu cung cấp.
    """
    if num_compatible <= 0:
        raise ValueError("num_compatible phải > 0")
    distinct_valid: set[bytes] = set()
    for tok in np.asarray(sample_tokens):
        if is_valid_board(tok, clue_cells, n):
            distinct_valid.add(board_key(tok))
    return len(distinct_valid) / num_compatible


def aggregate_coverage(
    per_input_coverage: dict[int, float],
    row_input_ids: np.ndarray,
) -> dict[str, float]:
    """Hai cách gộp coverage (D2).

    per_input_coverage: input_id -> coverage (đã tính trên input duy nhất).
    row_input_ids: input_id của từng dòng test (input trùng lặp xuất hiện lặp).
    """
    ids = list(per_input_coverage.keys())
    by_unique_input = float(np.mean([per_input_coverage[i] for i in ids]))
    row_covs = [per_input_coverage[int(i)] for i in np.asarray(row_input_ids)]
    by_row = float(np.mean(row_covs))
    return {"by_unique_input": by_unique_input, "by_row": by_row}


def first_sample_accuracy(
    first_samples: np.ndarray,
    clue_cells_per_input: list[np.ndarray],
    n: int = 8,
) -> float:
    """Tỉ lệ input mà mẫu đầu tiên là bàn hợp lệ khớp clue (§5.2).

    first_samples: (num_inputs, n*n) mẫu thứ nhất của từng input.
    clue_cells_per_input: ô clue tương ứng từng input.
    """
    first_samples = np.asarray(first_samples)
    valid = [
        is_valid_board(first_samples[i], clue_cells_per_input[i], n)
        for i in range(first_samples.shape[0])
    ]
    return float(np.mean(valid))
