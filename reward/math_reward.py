"""
Math reward function for One-Shot-RLVR + SDPO.

Exposes compute_score() with SDPO's custom reward function signature
(verl/workers/reward_manager/naive.py NaiveRewardManager interface).

Grading utilities are taken VERBATIM from One-Shot-RLVR:
  verl/utils/reward_score/utils/utils.py
  (commit fa0677487c8aec63f7a87a9568de4bf2c47205b4)

Core grading strategy is from One-Shot-RLVR:
  verl/utils/reward_score/deepscaler.py
  with use_think=False (Qwen2.5-Math-1.5B does not produce <think> tags)

The function returns a dict so that SDPO's NaiveRewardManager populates
reward_extra_infos with extracted_answer and is_correct, which then appear
automatically in the native rollout JSONL written by trainer.validation_data_dir.

Dependencies:
  sympy       — required by grade_answer_sympy
  pylatexenc  — required by _parse_latex; installed with SDPO via pip
"""

# ---------------------------------------------------------------------------
# VERBATIM from One-Shot-RLVR verl/utils/reward_score/utils/utils.py
# Source: https://github.com/ypwang61/One-Shot-RLVR/blob/main/verl/utils/reward_score/utils/utils.py
# Commit: fa0677487c8aec63f7a87a9568de4bf2c47205b4
# NO changes made to any function body below this line until the SDPO wrapper.
# ---------------------------------------------------------------------------

"""
Answer checker API that uses sympy to simplify expressions and check for equality.

Call grade_answer(given_answer: str, ground_truth: str).
"""
import re
from pylatexenc import latex2text
import sympy
from sympy.parsing import sympy_parser
from typing import Optional


# Dan Hendrycks' code
def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer

def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string


    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string


    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string


    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution

def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
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
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct

def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False

def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None

# ---------------------------------------------------------------------------
# END verbatim One-Shot-RLVR code
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# grader.py — math_equal() as third grading fallback
# Source: pipeline_files/grader.py (from One-Shot-RLVR evaluation toolkit)
# Handles cases grade_answer_mathd and grade_answer_sympy both miss:
#   - numeric closeness (rel_tol=1e-4), percentage equivalence
#   - matrix equality, equation forms, latex2sympy parsing
# Gracefully disabled if latex2sympy2 or regex are not installed.
# ---------------------------------------------------------------------------

# latex2sympy2 generated parser targets ANTLR 4.7.2 but we install 4.9.3.
# The generated code calls: RuntimeMetaData.checkVersion("4.7.2", RuntimeMetaData.VERSION)
# RuntimeMetaData lives in antlr4/RuntimeMetaData.py (not antlr4/Runtime.py).
# Patch the class method before importing grader.py (which triggers latex2sympy2 parser load).
try:
    from antlr4.RuntimeMetaData import RuntimeMetaData as _AntlrRMD
    _AntlrRMD.checkVersion = classmethod(lambda cls, *args: None)
except Exception:
    try:
        from antlr4 import RuntimeMetaData as _AntlrRMD
        _AntlrRMD.checkVersion = classmethod(lambda cls, *args: None)
    except Exception:
        pass

try:
    from reward.grader import math_equal as _math_equal
    _GRADER_AVAILABLE = True
except Exception:
    try:
        import importlib.util, os as _os
        _grader_path = _os.path.join(_os.path.dirname(__file__), "grader.py")
        _spec = importlib.util.spec_from_file_location("grader", _grader_path)
        _grader_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_grader_mod)
        _math_equal = _grader_mod.math_equal
        _GRADER_AVAILABLE = True
    except Exception:
        _GRADER_AVAILABLE = False


def grade_answer_grader(given_answer: str, ground_truth: str) -> bool:
    """Third fallback: math_equal from grader.py (numeric + symbolic + latex2sympy)."""
    if not _GRADER_AVAILABLE:
        return False
    try:
        return bool(_math_equal(given_answer, ground_truth, timeout=False))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# SDPO custom reward wrapper
# Adapted from One-Shot-RLVR verl/utils/reward_score/deepscaler.py
# with use_think=False (Qwen2.5-Math-1.5B does not emit <think> tags)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Feedback for failed rollouts — environment-natural verifier output.
#
# The feedback string is SDPO's environment_output f_i.  It is placed into
# the self-teacher prompt alongside the original question and (optionally) a
# correct sibling rollout, then the original failed response is appended so
# the teacher re-evaluates the old trajectory under feedback-augmented
# context — NOT to generate a fresh answer (SDPO §3).
#
# Layer 0 — Format error (no \boxed{}):
#   "Your previous response did not include a final answer in \boxed{} format."
#
# Layer 1 — Wrong answer (all problems):
#   "Your answer X is incorrect. The correct answer is Y."
#   Mirrors what a real verifier returns: actual vs expected, nothing more.
#   No problem-specific heuristics — works for any math problem.
# ---------------------------------------------------------------------------

_FEEDBACK_NO_BOXED = (
    "Your previous response did not include a final answer in \\boxed{} format. "
    "Please state your answer as \\boxed{your answer}."
)


def _make_feedback(no_boxed: bool, model_answer: str = "", ground_truth: str = "") -> str:
    """Return environment feedback for a failed rollout (SDPO environment_output f_i).

    Layer 0 — no \\boxed{}: format nudge.
    Layer 1 — wrong answer: verifier output stating actual vs expected.
    """
    if no_boxed:
        return _FEEDBACK_NO_BOXED
    if not model_answer:
        return "Your answer is incorrect."
    if ground_truth:
        return f"Your answer {model_answer} is incorrect. The correct answer is {ground_truth}."
    return f"Your answer {model_answer} is incorrect."


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info=None,
) -> dict:
    """
    SDPO custom reward function for math (binary boxed-answer check).

    Called by verl/workers/reward_manager/naive.py NaiveRewardManager:
        compute_score(data_source, solution_str, ground_truth, extra_info)

    Returns a dict:
        score           : 1.0 if correct, 0.0 otherwise
        extracted_answer: the extracted \\boxed{} content, or None
        feedback        : verifier output for failed π₁ rollouts ("Your answer X is
                          incorrect. The correct answer is Y."); empty string for
                          correct answers and validation sources.
                          Must be present in every return dict so agent_loop can
                          stack reward_extra_infos into uniform numpy arrays.
                          Consumed by SDPO ray_trainer when
                          include_environment_feedback=true.

    Note: is_correct is intentionally omitted. SDPO's reward aggregator
    stacks extra-info values into numpy arrays; Python bool becomes
    numpy.bool_ which is not JSON-serializable. score=1.0/0.0 is
    sufficient to determine correctness.

    Dict return causes NaiveRewardManager to collect all extra keys into
    reward_extra_infos, which appear in the native SDPO rollout JSONL.
    """
    # use_think=False: extract from the full response string.
    # Qwen2.5-Math-1.5B is not a thinking model — no <think>...</think> blocks.
    # Only generate feedback for training data (deepscaler / π₁).
    # MATH-500 and all other eval data sources get feedback="" always:
    # they are validation-only (single-turn, no reprompting) so the
    # feedback field would never be consumed by the trainer.
    is_training_source = (data_source == "deepscaler")

    model_answer = extract_answer(solution_str)
    if model_answer is None:
        return {
            "score": 0.0,
            "extracted_answer": "",
            "feedback": _make_feedback(no_boxed=True) if is_training_source else "",
        }

    # Normalise ground_truth: may be str, float, int, or list
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
        return {
            "score": 0.0,
            "extracted_answer": model_answer,
            "feedback": _make_feedback(no_boxed=False, model_answer=model_answer) if is_training_source else "",
        }

    primary_gt = processed_ground_truths[0]

    for gt in processed_ground_truths:
        is_correct = (grade_answer_mathd(model_answer, gt)
                      or grade_answer_sympy(model_answer, gt)
                      or grade_answer_grader(model_answer, gt))
        if is_correct:
            return {"score": 1.0, "extracted_answer": model_answer, "feedback": ""}

    return {
        "score": 0.0,
        "extracted_answer": model_answer,
        "feedback": _make_feedback(no_boxed=False, model_answer=model_answer,
                                   ground_truth=primary_gt) if is_training_source else "",
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Correct answer in \boxed{} — feedback="" (always present so agent_loop keys are uniform)
    r = compute_score("lighteval/MATH", "\\boxed{12.8}", "12.8")
    assert r["score"] == 1.0, f"Expected 1.0, got {r}"
    assert r.get("feedback") == "", f"feedback must be empty string for correct answers, got {r}"
    assert "is_correct" not in r, "is_correct must not appear (causes numpy.bool_ serialization error)"

    # Wrong answer — non-pi1 (MATH-500): feedback="" (present but empty; val never uses it)
    r = compute_score("lighteval/MATH", "\\boxed{15}", "12.8")
    assert r["score"] == 0.0, f"Expected 0.0, got {r}"
    assert r.get("feedback") == "", f"feedback must be empty string for MATH-500, got {r}"

    # π₁ wrong answer: verifier output includes correct answer
    r = compute_score("deepscaler", "P = k*A*V^3, so V = 10 \\boxed{10}", "12.8")
    assert r["score"] == 0.0, f"Expected 0.0, got {r}"
    assert r["feedback"] == "Your answer 10 is incorrect. The correct answer is 12.8.", \
        f"Feedback should state correct answer, got: {r['feedback']}"

    # No \boxed{} at all — π₁ training: format nudge
    r = compute_score("deepscaler", "the answer is 12.8", "12.8")
    assert r["score"] == 0.0, f"Expected 0.0, got {r}"
    assert r["extracted_answer"] == ""
    assert "boxed" in r["feedback"].lower(), f"No-boxed feedback should mention format, got: {r['feedback']}"

    # No \boxed{} at all — non-training: feedback empty
    r = compute_score("lighteval/MATH", "the answer is 12.8", "12.8")
    assert r["score"] == 0.0, f"Expected 0.0, got {r}"
    assert r["extracted_answer"] == ""
    assert r.get("feedback") == "", f"MATH-500 no-boxed should have empty feedback, got: {r['feedback']}"

    # SymPy equivalence: \frac{64}{5} == 12.8
    r = compute_score("lighteval/MATH", "\\boxed{\\frac{64}{5}}", "12.8")
    assert r["score"] == 1.0, f"Expected 1.0 for 64/5 == 12.8, got {r}"

    # mathd: \dfrac normalised to \frac, then string match
    r = compute_score("lighteval/MATH", "\\boxed{\\dfrac{64}{5}}", "\\frac{64}{5}")
    assert r["score"] == 1.0, f"Expected 1.0 for dfrac == frac, got {r}"

    # Tuple answer
    r = compute_score("lighteval/MATH", "\\boxed{(1, 2, 3)}", "(1, 2, 3)")
    assert r["score"] == 1.0, f"Expected 1.0 for tuple match, got {r}"

    print("All reward tests passed.")
