# implement/agents/symbolic_checker.py

from typing import Any, Dict, List, Tuple
import os
import sys

# 이 파일이 있는 폴더(agents)를 sys.path에 추가해서
# evaluation.hard_constraint 를 import 할 수 있게 함
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from evaluation.hard_constraint import evaluation as hard_eval

def run_symbolic_check(
    updated_constraints: Dict[str, Any],
    response_data: Any,
    ref_data: Any,
) -> Dict[str, Any]:
    """
    외부에서 사용하는 통합 제약 검사 함수.

    Args:
        updated_constraints: query_data 형식의 제약 정보 (budget, local_constraint 등)
        response_data: LLM이 생성한 plan (문자열 또는 리스트[dict])
        ref_data: reference_information (호텔/식당/비행기 정보)

    Returns:
        {
            "total_score": 각 제약별 검사 결과(dict),
            "pass_rate": float,
            "correct_count": int,
            "total_count": int,
            "violated": [깨진 제약 이름 리스트],
        }
    """
    # 1) 기존 hard_constraint 평가 함수 호출
    total_score = hard_eval(updated_constraints, response_data, ref_data)

    # 2) pass rate 계산 (query_value가 있는 제약만 대상으로)
    correct_count = 0
    total_count = 0
    violated: List[str] = []

    for ctype, result in total_score.items():
        query_val = result.get("query_value", None)
        is_correct = result.get("is_correct", None)

        if query_val is None or is_correct is None:
            continue  # 이번 샘플에서 해당 제약이 아예 없거나 검사 대상이 아님

        total_count += 1
        if is_correct:
            correct_count += 1
        else:
            violated.append(ctype)

    pass_rate = correct_count / total_count if total_count > 0 else 0.0

    return {
        "total_score": total_score,
        "pass_rate": pass_rate,
        "correct_count": correct_count,
        "total_count": total_count,
        "violated": violated,
    }


def get_current_constraint_result(
    turn_constraint: Any,
    total_score: Dict[str, Dict[str, Any]],
) -> Any:
    """
    Flow에서 '이번 턴 제약' 결과만 따로 보고 싶을 때 사용.

    Args:
        turn_constraint: "budget", "room type" 같은 문자열 또는 {"budget": ...} dict
        total_score: run_symbolic_check()["total_score"]

    Returns:
        해당 제약의 검사 결과(dict) 또는 "No specific constraint this turn"
    """
    # 예: {"budget": 11300} 형태로 들어오는 경우 키만 뽑기
    if isinstance(turn_constraint, dict):
        turn_constraint = next(iter(turn_constraint))

    if turn_constraint == "house rule":
        key = "room_rule"
    else:
        key = turn_constraint.replace(" ", "_")  # "room type" -> "room_type"

    return total_score.get(key, "No specific constraint this turn")


def format_violation_messages(result: Dict[str, Any], show_satisfied: bool = True) -> str:
    """
    run_symbolic_check 결과를 받아서,
    LLM 프롬프트에 그대로 붙일 수 있는 '자연어 피드백' 문자열로 바꿉니다.
    (인코딩 에러 방지 및 구체적 힌트 추가 버전)
    """
    lines: List[str] = []
    lines.append("[Constraint Checker Feedback]")
    lines.append(
        "The current plan violates the following constraints. Please modify the plan specifically to resolve these issues:\n")

    total_score: Dict[str, Dict[str, Any]] = result.get("total_score", {})

    # 1. 힌트 사전: 각 제약조건별로 LLM이 해야 할 행동을 미리 정의
    hints = {
        "budget": "Calculate the total cost again (Flights + Accommodation + Restaurants). Try to choose cheaper hotels or restaurants to fit the budget.",
        "people_number": "Ensure the cost calculation accounts for all members of the group.",
        "cuisine": "Check if the selected restaurants serve the required cuisine type and exist in the database.",
        "room_type": "Check the accommodation description to ensure it matches the required room type (e.g., 'entire room', 'shared room').",
        "room_rule": "Check the house rules of the accommodation (e.g., 'parties allowed', 'no smoking').",
        "transportation": "Verify that the transportation method matches the user's request.",
        "days": "Ensure the plan covers exactly the requested number of days."
    }

    # 아무 검사 결과가 없는 경우
    if not total_score:
        return "[Constraint Checker Feedback]\nNo constraints checked."

    violation_count = 0

    for constraint_type, info in total_score.items():
        query_val = info.get("query_value", None)
        is_correct = info.get("is_correct")

        # 에러 메시지 가져오기 (없으면 기본값)
        raw_error_message = info.get("error_message", "Constraint violated.")

        # [중요] 인코딩 에러를 유발하는 특수 문자(–, —)를 일반 하이픈(-)으로 교체
        if isinstance(raw_error_message, str):
            error_message = raw_error_message.replace("–", "-").replace("—", "-").replace("’", "'")
        else:
            error_message = str(raw_error_message)

        if query_val is None or is_correct is None:
            continue

        if is_correct:
            if show_satisfied:
                lines.append(f"[PASS] {constraint_type}: satisfied (Required: {query_val})")
        else:
            violation_count += 1
            # 위반 시: 상태 + 에러메시지 + 힌트
            lines.append(f"[VIOLATED] {constraint_type}")
            lines.append(f"  - Requirement: {query_val}")

            if "calculated_value" in info:
                lines.append(f"  - Your Plan's Actual Value: {info['calculated_value']}")

            lines.append(f"  - Error: {error_message}")

            # 힌트가 있으면 추가
            if constraint_type in hints:
                lines.append(f"  - Action: {hints[constraint_type]}")

            lines.append("")  # 가독성을 위해 빈 줄 추가

    # 모두 통과했을 때 메시지
    if not show_satisfied and violation_count == 0:
        lines.append("All constraints are currently satisfied.")

    feedback_text = "\n".join(lines)
    return feedback_text
