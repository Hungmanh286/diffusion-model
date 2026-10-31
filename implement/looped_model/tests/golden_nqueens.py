"""Bộ liệt kê N-Queens tham chiếu độc lập (PLAN_TABLE2.md M1, Mục 9).

Mục đích: sinh các giá trị chuẩn bằng backtracking đơn giản, tách khỏi
`lf/data_nqueens.py`, để test đối chiếu. Không phụ thuộc torch; chỉ cần Python
chuẩn. Không lấy con số nào từ bài báo — mọi giá trị do liệt kê sinh ra.

Giá trị chuẩn cho N=8 (đã chạy):
    solutions=92, rows=8464, unique_inputs=5148,
    avg_solutions_per_input=1.6441, max=18, frac_single=0.6550,
    inputs_by_clue_size={1:64, 2:1052, 3:4032}
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations


def enumerate_solutions(n: int = 8) -> list[tuple[int, ...]]:
    """Mọi lời giải N-Queens đầy đủ, mỗi lời giải là tuple cột theo từng hàng."""
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


def enumerate_rows(n: int = 8, hide_counts=(5, 6, 7)):
    """Sinh mọi dòng (lời giải, tập quân clue được giữ lại).

    'Ẩn h quân' nghĩa là giữ lại n-h quân làm clue. Trả về:
        rows: list các (solution_index, clue) với clue = frozenset{(row, col)}
        input_to_sols: clue -> set chỉ số lời giải tương thích (chứa clue đó)
        solutions: danh sách lời giải
    """
    solutions = enumerate_solutions(n)
    keep_counts = sorted(n - h for h in hide_counts)  # 1,2,3 với mặc định
    rows: list[tuple[int, frozenset]] = []
    input_to_sols: dict[frozenset, set[int]] = defaultdict(set)
    for si, sol in enumerate(solutions):
        queens = [(r, sol[r]) for r in range(n)]
        for keep in keep_counts:
            for sub in combinations(queens, keep):
                clue = frozenset(sub)
                rows.append((si, clue))
                input_to_sols[clue].add(si)
    return rows, dict(input_to_sols), solutions


def golden_stats(n: int = 8, hide_counts=(5, 6, 7)) -> dict:
    rows, input_to_sols, solutions = enumerate_rows(n, hide_counts)
    counts = [len(s) for s in input_to_sols.values()]
    by_clue_size: dict[int, int] = defaultdict(int)
    for clue in input_to_sols:
        by_clue_size[len(clue)] += 1
    return {
        "num_solutions": len(solutions),
        "num_rows": len(rows),
        "num_unique_inputs": len(input_to_sols),
        "avg_solutions_per_input": sum(counts) / len(counts),
        "max_solutions_per_input": max(counts),
        "frac_single_solution": sum(1 for c in counts if c == 1) / len(counts),
        "inputs_by_clue_size": dict(sorted(by_clue_size.items())),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(golden_stats(), indent=2))
