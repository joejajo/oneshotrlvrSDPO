"""
Math reward function for One-Shot-RLVR + SDPO.

Exposes compute_score() with SDPO's custom reward function signature
(verl/workers/reward_manager/naive.py NaiveRewardManager interface).

Grading logic is taken verbatim from One-Shot-RLVR:
  - utility functions from verl/utils/reward_score/utils/utils.py
  - core grading strategy from verl/utils/reward_score/deepscaler.py
    with use_think=False (Qwen2.5-Math-1.5B does not produce <think> tags)

The function returns a dict so that SDPO's NaiveRewardManager populates
reward_extra_infos with extracted_answer and is_correct, which then appear
automatically in the native rollout JSONL written by trainer.validation_data_dir.

Dependencies (beyond the sdpo conda env base):
  sympy    — required by grade_answer_sympy
  pylatexenc — required by normalize helpers; installed with SDPO via pip
"""

# ---------------------------------------------------------------------------
# Verbatim from One-Shot-RLVR verl/utils/reward_score/utils/utils.py
# Source: https://github.com/ypwang61/One-Shot-RLVR/blob/main/verl/utils/reward_score/utils/utils.py
# No changes made to the bodies of these functions.
# ---------------------------------------------------------------------------

import re
import unicodedata
from typing import Optional

from pylatexenc.latex2text import LatexNodes2Text


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def _str_is_int(x: str) -> bool:
    try:
        int(x)
        return True
    except ValueError:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.match(r"^-?[0-9]+/[0-9]+$", expr))


def _str_to_rat(x: str):
    """Convert a fraction string a/b to (a, b)."""
    p, q = x.split("/")
    return int(p), int(q)


def _rat_to_float(p: int, q: int) -> float:
    return p / q


def _normalize(x: str) -> Optional[str]:
    """Normalize a math expression string for comparison."""
    if x is None:
        return None
    # strip surrounding whitespace
    x = x.strip()
    if len(x) == 0:
        return None
    return x


def split_tuple(x: str):
    """Split a tuple string like '(1, 2, 3)' into individual elements."""
    if x.startswith("(") and x.endswith(")"):
        inner = x[1:-1]
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) > 1:
            return parts
    return [x]


def mathd_normalize_answer(s: str) -> str:
    """String-normalise an answer for the MathD grader."""
    if s is None:
        return ""
    s = s.strip()
    # remove trailing punctuation
    s = s.rstrip(".")
    # normalise unicode
    s = unicodedata.normalize("NFKC", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    """Check answer equivalence via string normalisation (MathD style)."""
    if given_answer is None or ground_truth is None:
        return False
    return mathd_normalize_answer(given_answer) == mathd_normalize_answer(ground_truth)


def are_equal_under_sympy(expr1: str, expr2: str) -> bool:
    """Check whether two expressions are symbolically equal via SymPy."""
    try:
        import sympy
        from sympy.parsing.latex import parse_latex

        # Try parsing as LaTeX first, fall back to sympify
        def _parse(s: str):
            try:
                return parse_latex(s)
            except Exception:
                try:
                    return sympy.sympify(s)
                except Exception:
                    return None

        e1 = _parse(expr1)
        e2 = _parse(expr2)
        if e1 is None or e2 is None:
            return False
        diff = sympy.simplify(e1 - e2)
        return diff == 0
    except Exception:
        return False


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    """Check answer equivalence via SymPy symbolic evaluation."""
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)
    if ground_truth_normalized is None:
        return False
    if ground_truth_normalized == given_normalized:
        return True
    if not given_normalized:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        is_correct = True
        for gt_elem, gv_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(gt_elem) and _is_frac(gv_elem):
                elem_correct = gt_elem == gv_elem
            elif _str_is_int(gt_elem) != _str_is_int(gv_elem):
                elem_correct = False
            else:
                elem_correct = are_equal_under_sympy(gt_elem, gv_elem)
            if not elem_correct:
                is_correct = False
                break

    return is_correct


def extract_boxed_answer(passage: str) -> Optional[str]:
    """Extract the content of the last \\boxed{} in passage."""
    idx = passage.rfind("\\boxed")
    if idx < 0:
        idx = passage.rfind("\\fbox")
    if idx < 0:
        return None
    i = idx
    # find the opening brace
    while i < len(passage) and passage[i] != "{":
        i += 1
    if i >= len(passage):
        return None
    depth = 0
    end = -1
    for j in range(i, len(passage)):
        if passage[j] == "{":
            depth += 1
        elif passage[j] == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
    if end == -1:
        return None
    return passage[i + 1 : end]


def extract_answer(passage: str) -> Optional[str]:
    """Return the content of the last \\boxed{} in passage, or None."""
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


# ---------------------------------------------------------------------------
# SDPO custom reward interface
# Adapted from One-Shot-RLVR verl/utils/reward_score/deepscaler.py
# Source: https://github.com/ypwang61/One-Shot-RLVR/blob/main/verl/utils/reward_score/deepscaler.py
#
# Key adaptation:
#   - use_think=False  (Qwen2.5-Math-1.5B does not emit <think> tags;
#     using use_think=True would return 0.0 for every rollout on this model)
#   - signature extended to match SDPO's NaiveRewardManager call convention
#   - returns dict so NaiveRewardManager propagates extracted_answer and
#     is_correct into reward_extra_infos → appears in validation JSONL
# ---------------------------------------------------------------------------


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info=None,
) -> dict:
    """
    SDPO custom reward function for math (binary boxed-answer check).

    Called by verl/workers/reward_manager/naive.py NaiveRewardManager with:
        compute_score(data_source, solution_str, ground_truth, extra_info)

    Returns a dict with:
        score           : 1.0 if correct, 0.0 otherwise
        extracted_answer: the extracted \\boxed{} content, or None
        is_correct      : bool

    Dict return causes NaiveRewardManager to collect extracted_answer and
    is_correct into reward_extra_infos, which then appear in the native
    SDPO rollout JSONL (trainer.validation_data_dir output).
    """
    # use_think=False: extract from the full response string.
    # Qwen2.5-Math-1.5B is not a thinking model and does not produce
    # <think>...</think> blocks. Using use_think=True would require those
    # tags to be present and would return 0.0 for every rollout.
    model_solution = solution_str

    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return {"score": 0.0, "extracted_answer": None, "is_correct": False}

    # Normalise ground_truth: may be str, float, or int; may itself contain \boxed{}
    if isinstance(ground_truth, (str, float, int)):
        ground_truths_raw = [ground_truth]
    else:
        ground_truths_raw = list(ground_truth)

    processed_ground_truths = []
    for gt in ground_truths_raw:
        gt_str = str(gt)
        if "\\boxed" in gt_str:
            extracted_gt = extract_answer(gt_str)
            processed_ground_truths.append(extracted_gt if extracted_gt is not None else gt_str)
        else:
            processed_ground_truths.append(gt_str)

    if not processed_ground_truths:
        return {"score": 0.0, "extracted_answer": model_answer, "is_correct": False}

    for gt in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, gt) or grade_answer_sympy(model_answer, gt)
        if is_correct:
            return {"score": 1.0, "extracted_answer": model_answer, "is_correct": True}

    return {"score": 0.0, "extracted_answer": model_answer, "is_correct": False}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Correct answer in \boxed{}
    r = compute_score("lighteval/MATH", "\\boxed{12.8}", "12.8")
    assert r["score"] == 1.0, f"Expected 1.0, got {r}"
    assert r["is_correct"] is True

    # Wrong answer in \boxed{}
    r = compute_score("lighteval/MATH", "\\boxed{15}", "12.8")
    assert r["score"] == 0.0, f"Expected 0.0, got {r}"
    assert r["is_correct"] is False

    # No \boxed{} at all
    r = compute_score("lighteval/MATH", "the answer is 12.8", "12.8")
    assert r["score"] == 0.0, f"Expected 0.0, got {r}"
    assert r["extracted_answer"] is None

    # SymPy equivalence: fraction equal to decimal
    r = compute_score("lighteval/MATH", "\\boxed{\\frac{64}{5}}", "12.8")
    assert r["score"] == 1.0, f"Expected 1.0 for 64/5 == 12.8, got {r}"

    print("All reward tests passed.")
