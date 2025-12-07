import json
import logging
import pandas as pd
import random
import os
import re
from datetime import datetime
from pathlib import Path

# import dataclass
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml
from openai import OpenAI
# from together import Together
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents.swarm import Agent, Result, Swarm
from agents.utils import get_json_prompt
from agents.constraints_checker import planning_validate_constraints
from agents.prompts_add_mpe import build_constraint_memory, get_priority_guideline, SELF_EVAL_PROMPT, CONSTRAINT_ADDING_W_HISTORY, CONSTRAINT_ADDING_WO_HISTORY, INITIAL_PROMPT 
from agents.constraints_generator import update_constraints_and_query
from agents.postprocess_plan import parse_plan

#추가
import ast
#추가) Priority MAP
ALWAYS_CRITICAL_KEYS = [
    'people_number', 
    'days', 
    'org', 
    'dest', 
    'date', 
    'visiting_city_number'
]
# Pass-rate grouping helpers
GLOBAL_KEYS = {'budget', 'people_number', 'days', 'org', 'dest', 'date', 'visiting_city_number'}
LOCAL_KEYS = {'house rule', 'room type', 'transportation', 'cuisine', 'ratings'}
PREFERENCE_KEYS = {'cuisine_pref', 'rating_pref'}
### NUMERICAL PRIORITY MAP
def calculate_priority_numerical(rank_map, mode: str = "linear"):
    """Numerical priority.
    - linear: 가중치를 개수 기반 선형 분포로 떨어뜨림 (최소 0.2 보장)
    - reciprocal: 역수 분포를 정규화해 너무 작아지지 않도록 보정
    """
    if not rank_map:
        return {}

    priority_map = {}
    # 1. Critical keys → 1.0 고정, 나머지는 경쟁 그룹에 담기
    competitors = []
    for key, rank in rank_map.items():
        if key in ALWAYS_CRITICAL_KEYS:
            priority_map[key] = 1.0
        else:
            competitors.append((key, rank))

    # 2. Competitor 재정렬 (Rank순)
    competitors.sort(key=lambda x: x[1])
    # 경쟁 그룹 크기 (ALWAYS_CRITICAL_KEYS 제외)
    n = len(competitors)

    def linear_weight(idx, total, min_w=0.2, max_w=1.0):
        if total <= 1:
            return max_w
        step = (max_w - min_w) / (total - 1)
        return max(max_w - step * idx, min_w)

    def reciprocal_weight(idx, total):
        if total <= 1:
            return 1.0
        weights = [1 / (i + 1) for i in range(total)]
        total_w = sum(weights)
        # 정규화하여 평균을 1.0 근처로 유지
        return (weights[idx] / total_w) * total

    for i, (key, _) in enumerate(competitors):
        if mode == "reciprocal":
            weight = reciprocal_weight(i, n)
        else:  # linear
            weight = linear_weight(i, n)
        priority_map[key] = round(max(0.1, weight), 2)

    return priority_map

### RANK PRIORITY MAP
def calculate_priority_rank_only(rank_map):
    if not rank_map: return {}
    return {k: v for k, v in rank_map.items()}

### LABEL PRIORITY MAP
def calculate_priority_label(rank_map):
    """
    HYBRID_RANK에서 상대평가를 통해 얻은 label 사용.
    """
    hybrid_map = calculate_priority_hybrid_rank(rank_map)
    return {k: v['label'] for k, v in hybrid_map.items()}
       
### HYBRID PRIORITY MAP (LABEL + RANK)
def calculate_priority_hybrid_rank(rank_map):
    """
    1. 필수 키(Always Critical)는 무조건 [CRITICAL] 부여.
    2. 나머지 키들끼리만 다시 줄을 세워 상대평가(HIGH/MED/LOW)하되,
       전체 개수에 따라 임계값을 가변적으로 설정.
    3. Rank는 받았던 그대로 사용.
    """
    if not rank_map: return {}
    
    priority_map = {}
    competitors = [] # (key, original_rank)

    # 1. 그룹 분리
    for key, rank in rank_map.items():
        if key in ALWAYS_CRITICAL_KEYS:
            priority_map[key] = {
                'label': "[CRITICAL]",
                'rank': rank
            }
        else:
            competitors.append((key, rank))

    # 2. 경쟁그룹 서열 재정리
    competitors.sort(key=lambda x: x[1])
    total_competitors = len(competitors)

    # 3. 상대 평가 (가변 임계값)
    if total_competitors > 0:
        # 개수 기반 임계값: 상위 1/3 → HIGH, 중간 1/3 → MED, 나머지 → LOW
        import math
        high_cut = max(1, math.ceil(total_competitors / 3))
        med_cut = max(high_cut + 1, math.ceil((2 * total_competitors) / 3))

        for i, (key, original_rank) in enumerate(competitors):
            relative_rank = i + 1
            
            if total_competitors == 1:
                label = "[HIGH]"
            elif total_competitors == 2:
                label = "[HIGH]" if relative_rank == 1 else "[MEDIUM]"
            else:
                if relative_rank <= high_cut:
                    label = "[HIGH]"
                elif relative_rank <= med_cut:
                    label = "[MEDIUM]"
                else:
                    label = "[LOW]"
            
            priority_map[key] = {
                'label': label,
                'rank': original_rank
            }
            
    return priority_map

### HYBRID PRIORITY MAP (LABEL + WEIGHT)
def calculate_priority_hybrid_weight(rank_map):
    if not rank_map: return {}
    
    # 1. 각각의 함수에서 정보 가져오기
    labels_map = calculate_priority_label(rank_map)      # {key: '[HIGH]'}
    weights_map = calculate_priority_numerical(rank_map) # {key: 0.9}
    
    # 2. 결과 합치기
    priority_map = {}
    for key in rank_map.keys():
        priority_map[key] = {
            'label': labels_map.get(key),
            'weight': weights_map.get(key)
        }

    return priority_map



@dataclass
class Response:
    messages: list
    agent: Agent
    context_variables: dict

class Evaluator(Agent):
    def __init__(
        self,
        config: dict,
        name: Optional[str] = None,
    ):

        evaluator_config = config["evaluator"]
        name = name or evaluator_config["name"]
        client_kwargs = evaluator_config.get("client", None)
        client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
        task = config['task']
        instructions = ""
        super().__init__(name=name, instructions=instructions, client=client)
        self.task = task
        self.evaluator_config = evaluator_config
        # self.constraints_dict = self.inital_constraints
        
    def check_constraint(self, constraints: str, response: str):
        if self.task == "test":
            current_constraint_score, total_score = planning_validate_constraints(
                turn_constraint=constraints,
                updated_constraints=self.constraints_dict, 
                response_data=response,
                ref_data=self.ref_data)
            # calculate the pass rate of the constraints (is_correct==True/total length of constraints)
            correct_count = sum(1 for item in total_score.values() if item.get('is_correct') is True)
            # total_count = len(current_constraint_score)
            # total count is where the constraints 'is_correct' is not None
            total_count = sum(1 for item in total_score.values() if item.get('query_value') is not None)
            if total_count == 0:
                print("No constraints to evaluate")
                return current_constraint_score, total_score, 0
            total_pass_rate = correct_count / total_count
        return current_constraint_score, total_score, total_pass_rate

    def check_quality(self, response) -> Result:
        # TODO: Implement quality check. Commonsense constraints, etc.
        final_quality_scores = "None"
        return final_quality_scores
    
    def conclude_evaluate(self, score: int, comments: str) -> Result:
        """End interview with final assessment.

        Called when max questions reached, understanding established, or unable to progress further.
        Also called when forced to conclude interview.

        Args:
            score (int): Final score (0-10) based on rubric
            comments (str): Overall evaluation including strengths,
                weaknesses, and areas for improvement

        Returns:
            Result: Final assessment with score and detailed feedback
        """
        return Result(
            value=f"Interview concluded. Score: {score}\nComments: {comments}",
            context_variables={
                "interview_complete": True,
                "score": score,
                "comments": comments,
            },
        )


class Evaluatee(Agent): #계획 생성 LLM
    def __init__(
        self,
        config: dict,
        name: Optional[str] = None,
    ):

        evaluatee_config = config["evaluatee"]
        name = name or evaluatee_config["name"]
        model = evaluatee_config["model"]
        api_type = evaluatee_config.get("api_type", "openai")
        client_kwargs = evaluatee_config.get("client", None)
        if api_type == 'together':
            # client = Together(api_key=evaluatee_config.get("client", {}).get("api_key"))
            pass
        else:
            # Upstage API 연동을 위한 base_url 설정
            if client_kwargs:
                client_kwargs['base_url'] = "https://api.upstage.ai/v1"
                client = OpenAI(**client_kwargs)
            else:
                client = OpenAI(base_url="https://api.upstage.ai/v1")

        instructions = evaluatee_config["instructions"]
        super().__init__(name=name, model=model,instructions=instructions, client=client, api_type=api_type)

class Runner:
    def __init__(
        self,
        evaluator: Agent,
        evaluatee: Agent,
        config: dict,
        logger: logging.Logger,
        log_file_path: str,
        console: Console,
    ):
        self.client = Swarm()
        self.evaluator = evaluator
        self.evaluatee = evaluatee
        self.config = config
        self.logger = logger
        self.log_file_path = log_file_path
        self.console = console
        self.questions_count = 0
        self.max_constraints = len(self.evaluator.new_constraints)
        self.constraints = self.evaluator.new_constraints
        self.evaluator_messages = []
        self.evaluatee_messages = []
        self.scores = {"constraints":[]}
        self.all_total_scores = []  # Add this line
        self.questions = []
        self.responses = []
        self.seed_question_used = False
        # self.constraints_dict = self.evaluator.inital_constraints
        self.task = self.evaluator.task
        


    def display_message(self, agent_name: str, content: str):
        """Display a message with proper formatting."""

        agent_name_to_style = {
            self.evaluator.name.lower(): "blue",
            self.evaluatee.name.lower(): "green",
            "constraint checker": "yellow",
            "overall quality checker": "magenta",
        }

        style = agent_name_to_style[agent_name.lower()]
        panel = Panel(
            content,
            title=f"[{style}]{agent_name}[/{style}]",
            border_style=style,
            padding=(1, 2),
        )
        # Only print to console if in verbose mode
        if self.logger.getEffectiveLevel() <= logging.INFO:
            self.console.print(panel)

        # Always log to file if file logging is enabled
        self.logger.info(f"{agent_name}: {content}")

    def display_results(self, results: Dict[str, Any]):
        """Display interview results with formatting."""
        score = results["score"]
        score_color = "success" if score >= 0.7 else "warning" if score >= 0.5 else "error"

        results_panel = Panel(
            f"\n[{score_color}]Final Score: {score}[/{score_color}]\n\n"
            f"[info]Constrained Added: {results['questions_asked']}[/info]\n\n"
            f"[white]Constraint Count:[/white]\n{results['constraint_counts']}",
            title="[success]Assessment Results[/success]",
            border_style="success",
            padding=(1, 2),
        )
        self.console.print("\n")
        self.console.print(results_panel)

    def _get_response(self, agent: Agent, messages: list,
                      context: dict) -> Result:
        """Helper method to get response with progress spinner."""
        
        if not self.config['history']:
            messages = [messages[-1]]
        messages = [{"role": message["role"], "content": str(message["content"])} for message in messages]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Processing response...", total=None)
            return self.client.run(agent=agent, messages=messages, context_variables=context)

    def _get_response_raw(self, agent: Agent, messages: list,
                          chat_params: dict, json: bool = False) -> Response:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Processing response...", total=None)
            full_params = {
                "model": agent.model,
                "messages": messages,
            }

            if json:
                full_params["response_format"] = {"type": "json_object"}

            full_params.update(chat_params)
            raw_response = agent.client.chat.completions.create(**full_params)
            content = raw_response.choices[0].message.content
            return Response(messages=[
                            {"role": "assistant", "content": content}], agent=agent, context_variables={})

    def add_message(self, speaker, content):
        """Add messages to both conversation tracks based on who's speaking.

        When evaluator speaks: they're the assistant, evaluatee is the user
        When evaluatee speaks: they're the assistant, evaluator is the user
        """
        if speaker == self.evaluator:
            # evaluator is speaking (as assistant) to evaluatee (as user)
            self.evaluator_messages.extend(
                [{"role": "assistant", "content": content}])
            self.evaluatee_messages.extend(
                [{"role": "user", "content": content}])
        else:
            # evaluatee is speaking (as assistant) to evaluator (as user)
            self.evaluator_messages.extend(
                [{"role": "user", "content": content}])
            self.evaluatee_messages.extend(
                [{"role": "assistant", "content": content}])

    def call_feedback_agent(self, response):
        # constraints = self.evaluator.new_constraints[:self.questions_count]
        # if self.questions_count == 0, constraints = '' else, constraints = self.evaluator.new_constraints[self.questions_count]
        constraints = self.evaluator.new_constraints[self.questions_count - 1] if self.questions_count > 0 else ''
        current_score, total_score, total_pass_rate = self.evaluator.check_constraint(constraints,response)
        return current_score, total_score, total_pass_rate

    # 수정)
    def call_question_agent(self):
        # Config 설정 읽기
        use_memory = self.config.get('use_memory', False)
        use_priority = self.config.get('use_priority', False)
        use_self_eval = self.config.get('use_self_eval', False)
        priority_type = self.config.get('priority_type', 'hybrid_rank')

        user_query_text = ""
        response_prompt = ""
        current_turn_keys = []

        # 질문 텍스트 준비
        target_constraints = {}

        if self.questions_count == 0:
            user_query_text = self.evaluator.seed_question
            target_constraints = self.evaluator.constraints_dict.copy()
        else:
            constraint_seed = self.evaluator.new_constraints[self.questions_count-1]
            self.evaluator.constraints_dict, constraint = update_constraints_and_query(self.evaluator.constraints_dict, constraint_seed)
            user_query_text = constraint

            # 새로 추가된 constraint의 키가 대상
            if isinstance(constraint_seed, dict):
                current_turn_keys = list(constraint_seed.keys())

        # Priority Info & Guideline 계산
        priority_info = None
        guideline_text = ""
        
        if use_priority:
            priority_list = []
            raw_constraints = target_constraints if target_constraints else self.evaluator.constraints_dict
            needed_constraints = []
            constraint_values = {}

            # preference용 랭크가 있는 경우 cuisine -> cuisine_pref로 정규화
            def canonical_key(k: str) -> str:
                if k == 'cuisine' and 'cuisine_pref' in self.evaluator.constraint_ranks.get('turn_1', {}):
                    return 'cuisine_pref'
                if k == 'cuisine' and 'cuisine_pref' in self.evaluator.constraint_ranks:
                    return 'cuisine_pref'
                return k

            for k, v in raw_constraints.items():
                if k == 'local_constraint':
                    # Normalize local constraint data to a dict; avoid unbound actual_data
                    actual_data = {}
                    if isinstance(v, str):
                        try:
                            # str -> dict
                            actual_data = ast.literal_eval(v)
                        except Exception as e:
                            print(f"[Warning] Failed to parse local_constraint: {v}")
                            actual_data = {}
                    elif isinstance(v, dict):
                        actual_data = v

                    if isinstance(actual_data, dict):
                        for sub_k, sub_v in actual_data.items():
                            canon_k = canonical_key(sub_k)
                            if canon_k not in needed_constraints:
                                needed_constraints.append(canon_k)
                            constraint_values[canon_k] = sub_v
                else:
                    canon_k = canonical_key(k)
                    if canon_k not in needed_constraints:
                        needed_constraints.append(canon_k)
                    constraint_values[canon_k] = v

            all_ranks = self.evaluator.constraint_ranks
            # 랭크 키 결정: preference 전용(turn_2_rating/turn_2_cuisine) 우선, 없으면 일반 turn_n 사용
            if self.questions_count == 0:
                rank_key = 'turn_1'
            else:
                if len(current_turn_keys) == 1 and current_turn_keys[0] in ('rating_pref', 'cuisine_pref'):
                    pref_key = current_turn_keys[0].split('_')[0]  # rating / cuisine
                    rank_key = f"turn_{self.questions_count+1}_{pref_key}"
                else:
                    rank_key = f"turn_{self.questions_count+1}"
            ranks = all_ranks.get(rank_key, all_ranks.get('turn_2', all_ranks.get('turn_1', {})))

            # config priority_type 설정에 따라 priority 계산
            numerical_mode = "linear" if self.questions_count == 0 else "reciprocal"
            if priority_type == 'hybrid_rank': priority_info = calculate_priority_hybrid_rank(ranks)
            elif priority_type == 'hybrid_weight': priority_info = calculate_priority_hybrid_weight(ranks)
            elif priority_type == 'label': priority_info = calculate_priority_label(ranks)
            elif priority_type == 'rank_only': priority_info = calculate_priority_rank_only(ranks)
            else: priority_info = calculate_priority_numerical(ranks, mode=numerical_mode)

            guideline_text = get_priority_guideline(priority_type)

        # 우선순위 상세 목록 생성 및 템플릿 주입
        priority_details_text = ""
        if use_priority and priority_info:
            priority_list = []
            
            # config RANK를 사용할 경우, RANK가 정해지지 않은 constraint에 기본 rank 값 설정
            current_max_rank = 4 
            
            if priority_info:
                extracted_ranks = []
                for v in priority_info.values():
                    # Hybrid Rank인 경우 딕셔너리에서 'rank' 추출
                    if isinstance(v, dict) and 'rank' in v:
                        extracted_ranks.append(int(v['rank']))
                    # Rank Only인 경우 값 자체가 랭크
                    elif isinstance(v, (int, float)):
                        extracted_ranks.append(int(v))
                
                if extracted_ranks:
                    current_max_rank = max(extracted_ranks)
            
            # 정보가 없는 항목에 부여할 순위 (= 꼴등 + 1)
            default_next_rank = current_max_rank + 1

            # 해당 priority 값 주입
            include_values = bool(use_memory)  # 값 노출은 use_memory가 켜진 경우에만

            for k in needed_constraints:
                val_repr = constraint_values.get(k, "")
                val_repr = val_repr if isinstance(val_repr, str) else str(val_repr)
                if k.lower() in [key.lower() for key in ALWAYS_CRITICAL_KEYS]:
                    if priority_type == 'numerical':
                        priority_list.append(f"- {k}: {val_repr} [Weight 1.00]" if include_values else f"- {k} : 1.00")
                    elif priority_type == 'rank_only':
                        priority_list.append(f"- {k}: {val_repr} [Rank 1]" if include_values else f"- {k} : Rank 1")
                    else:
                        priority_list.append(f"- {k}: {val_repr} [CRITICAL]" if include_values else f"- {k} : [CRITICAL]")
                    continue

                if k in priority_info and priority_info:
                    val = priority_info[k]
                    
                    # 모드별 포맷팅 분기 (Priority Type을 완벽히 고려)
                    if priority_type == 'hybrid_rank':
                        # 예: - Budget : [HIGH] (Rank 1)
                        priority_list.append(f"- {k}: {val_repr} {val['label']} [Rank {val['rank']}]" if include_values else f"- {k} : {val['label']} (Rank {val['rank']})")
                        
                    elif priority_type == 'hybrid_weight':
                        # 예: - Budget : [HIGH] (Weight 0.85) 
                        priority_list.append(f"- {k}: {val_repr} {val['label']} [Weight {val['weight']:.2f}]" if include_values else f"- {k} : {val['label']} (Weight {val['weight']:.2f})")
                        
                    elif priority_type == 'label':
                        # 예: - Budget : [HIGH]
                        label = val['label'] if isinstance(val, dict) else val
                        priority_list.append(f"- {k}: {val_repr} {label}" if include_values else f"- {k} : {label}")
                        
                    elif priority_type == 'rank_only':
                        # 예: - Budget : Rank 1
                        priority_list.append(f"- {k}: {val_repr} [Rank {val}]" if include_values else f"- {k} : Rank {val}")
                        
                    else:
                        # 기본 (Numerical)
                        priority_list.append(f"- {k}: {val_repr} [Weight {val}]" if include_values else f"- {k} : {val}")
                else:
                    # 정보 없음
                    if priority_type == 'numerical':
                        # Weight 모드인데 정보가 없으면 낮은 점수 부여
                        priority_list.append(f"- {k}: {val_repr} [Weight 0.10]" if include_values else f"- {k} : 0.10")
                    elif priority_type == 'rank_only':
                        # Rank 모드인데 정보가 없으면 낮은 순위 부여
                        priority_list.append(f"- {k}: {val_repr} [Rank {default_next_rank}]" if include_values else f"- {k} : Rank {default_next_rank}") 
                    else: #hybrid의 경우, RANK가 주어지지 않았다면 label만 나올것.
                        # Label / Hybrid Rank / Hybrid weight 모드는 [LOW] 사용 
                        priority_list.append(f"- {k}: {val_repr} [LOW]" if include_values else f"- {k} : [LOW]")
            
            if priority_list:
                priority_details_text = "\n".join(priority_list)

        # priority 템플릿 완성하기.
        if guideline_text:
            if "{priority_lines}" in guideline_text:
                # priority 정보가 없는 경우.
                replacement = priority_details_text if priority_details_text else "(No specific priority details)"
                guideline_text = guideline_text.replace("{priority_lines}", replacement)
            elif priority_details_text:
                # full prompt 뒤에 붙이기
                 guideline_text += "\n\n[Current Constraints Priority]:\n" + priority_details_text
            
        # Memory Block 생성 (use_memory=True, use_priority = False, first turn이 아닐 때)
        constraint_block_text = ""
        if use_memory and not use_priority and self.questions_count > 0:
            current_new_constraints = self.evaluator.new_constraints[:self.questions_count]
            constraint_block_text = build_constraint_memory(
                previous_constraints= self.evaluator.constraints_dict,
                new_constraints=current_new_constraints,
                priority_info=priority_info
            )

        # 프롬프트 조립
        prompt_components = []


        # memory + priority 둘 다 켜진 경우 추가 안내 문구
        if use_memory and use_priority:
            prompt_components.append(
                "The following list summarizes the constraints you must satisfy.\n"
                "If there is the priority system, you must satisfy the constraints based on the priority system below:"
            )

        # priority info 넣기
        if guideline_text: prompt_components.append(guideline_text)
        # 과거 메모리 넣기.
        if constraint_block_text and not guideline_text: prompt_components.append(constraint_block_text)
        
        combined_content = "\n\n".join(prompt_components)   
        if self.questions_count == 0:
            response_prompt = INITIAL_PROMPT.format(question=combined_content, ref_data=self.evaluator.ref_data)
        else:
            if self.config.get('history', True):
                response_prompt = CONSTRAINT_ADDING_W_HISTORY.format(constraint=constraint)
                response_prompt = f"{response_prompt}\n\n{combined_content}"
            else:
                prev_condition = "\n".join(f"{k}: {v}" for k, v in self.evaluator.constraints_dict.items())
                response_prompt = CONSTRAINT_ADDING_WO_HISTORY.format(
                    ref_data=self.evaluator.ref_data,
                    previous_condition=prev_condition,
                    additional_condition=constraint,
                    response=self.responses[-1] if self.responses else ""
                )
                response_prompt = f"{response_prompt}\n\n{combined_content}"

        # SELF_EVAL
        if use_self_eval:
            response_prompt += "\n\n" + SELF_EVAL_PROMPT

        return user_query_text, response_prompt


    # Note: Please follow the convention of adding the message to the
    # conversation first and then displaying it
    def run(self) -> Dict[str, Any]:
        """Run the evaluation and return results."""
        # Start the evaluation loop
        
        self.console.print(f"\n[info]Constraints {self.questions_count}[/info]")
        
        self.seed_question_used = True
        query_to_display, response = self.call_question_agent()
        
        self.questions.append(response)
        self.add_message(self.evaluator, response)
        self.display_message(self.evaluator.name,
                             query_to_display)

        while self.questions_count <= (self.max_constraints):
            # 1. Get response from evaluatee
            question = self.evaluator_messages[-1]["content"]

            response = self._get_response(
                self.evaluatee, self.evaluatee_messages, {}).messages[-1]["content"]
            parsed_plan = parse_plan(response)
            if len(parsed_plan) == 0:
                self.console.print(f"\n[warning]No plan found in the response. Please provide a valid plan.[/warning]")
            
            # self.responses.append(response)
            self.responses.append(parsed_plan)
            self.add_message(self.evaluatee, parsed_plan)
            self.display_message(self.evaluatee.name, str(parsed_plan))
            
            # 2. Get feedback from feedback agent. Note that the message from
            # feedback agent is not added to the conversation
            # current_score, total_score = self.call_feedback_agent(response)
            current_score, total_score, total_pass_rate = self.call_feedback_agent(parsed_plan)

            self.scores['constraints'].append(current_score)
            self.scores['total_constraints_score'] = total_score
            self.scores['total_pass_rate'] = total_pass_rate
            self.all_total_scores.append(total_score)  # Add this line
            self.scores['all_total_constraints_score'] = self.all_total_scores

            self.display_message(
                "Constraint Checker",
                "Current_constraints_score: {}\nTotal_score: {}".format(
                    current_score, total_score
                ),
            )

            # Constraints
            self.questions_count += 1
            if self.questions_count <= self.max_constraints:
                self.console.print(f"\n[info]Constraints {self.questions_count}")
                query_to_display, response = self.call_question_agent()

                self.questions.append(response)
                self.add_message(self.evaluator, response)
                rm_txt = '[Reference information]:\nAll costs are per one person, one night.\n' + str(self.evaluator.ref_data) + '\n\n' 
                display_response = response.replace(rm_txt, '')
                self.display_message(self.evaluator.name, display_response)

            # 4. Check end conditions for the interview
            if self.questions_count == (self.max_constraints+1):
                final_message = "Maximum number of constraints reached. Concluding Evaluation."
                self.console.print(f"\n[warning]{final_message}[/warning]")

                # quality_score = self.evaluator.check_quality(response)
                # just use the final hard constraint pass rate as the quality score
                # quality_score = total_pass_rate
                self.scores['quality'] = total_pass_rate
                panel_content = f"Total Pass Rate: {total_pass_rate}"

                self.display_message("Overall Quality Checker",
                                     panel_content)
                break
            
        # print(self.scores)
        results = {
            # "score": response.context_variables["score"],
            # score: 그동안의 pass rate의 평균
            "constraint_counts": sum(1 for item in self.scores['total_constraints_score'].values() if item.get('is_correct') is not None),
            "score": self.scores['total_pass_rate'],
            # "feedback": response.context_variables["comments"],
            "questions_asked": self.questions_count - 1,
            "all_total_scores": self.all_total_scores,
        }

        self.display_results(results)
        return results


