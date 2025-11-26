import os
import sys
import json 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#from evaluation.hard_constraint import evaluation as hard_eval
from symbolic_checker import run_symbolic_check, get_current_constraint_result

'''
def planning_validate_constraints(turn_constraint, updated_constraints, response_data, ref_data):
    """
    Validates travel constraints based on the current turn's constraint and all updated constraints.
    
    Args:
        turn_constraint (str): Current turn's constraint (e.g., 'budget')
        updated_constraints (dict): All constraints including updates
        response_data (list): List of daily itinerary data
        ref_data (dict): Reference data for the current turn
    
    Returns:
        tuple: (turn_validation_result, all_constraints_result)
            - turn_validation_result: Dictionary containing validation result for the current turn's constraint
            - all_constraints_result: Dictionary containing validation results for all constraints
    """
    # Run evaluation on the entire plan
    all_constraints_result = hard_eval(updated_constraints, response_data, ref_data)

    # Extract the result specific to the current turn's constraint
    if type(turn_constraint) is dict:
       turn_constraint = next(iter(turn_constraint))
    # if turn_constraint == 'people_number':
    #     turn_validation_result = all_constraints_result.get('budget')
    print ("Current Turn Constraint:", turn_constraint)
    if turn_constraint == 'house rule':
        turn_constraint = 'room_rule'
    turn_validation_result = all_constraints_result.get(turn_constraint.replace(" ", "_"), 'No specific constraint this turn')

    return turn_validation_result, all_constraints_result
'''

def planning_validate_constraints(turn_constraint, updated_constraints, response_data, ref_data):
    """
    현재 턴 제약 + 전체 제약을 심볼릭 체커로 검증합니다.

    Args:
        turn_constraint (str or dict): 이번 턴에서 새로 추가된 제약 (예: 'budget')
        updated_constraints (dict): 지금까지 누적된 전체 제약 (query_data 형식)
        response_data: LLM이 생성한 plan
        ref_data: reference_information

    Returns:
        turn_validation_result: 이번 턴 제약의 검사 결과 (또는 메시지 문자열)
        all_constraints_result: 전체 제약들에 대한 검사 결과 dict
    """
    # 1) 통합 심볼릭 체크 실행
    result = run_symbolic_check(
        updated_constraints=updated_constraints,
        response_data=response_data,
        ref_data=ref_data,
    )
    all_constraints_result = result["total_score"]

    # 2) 이번 턴 제약 결과만 추출
    print("Current Turn Constraint:", turn_constraint)
    turn_validation_result = get_current_constraint_result(
        turn_constraint=turn_constraint,
        total_score=all_constraints_result,
    )

    return turn_validation_result, all_constraints_result

