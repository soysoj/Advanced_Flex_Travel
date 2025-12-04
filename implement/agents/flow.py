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
from together import Together
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents.swarm import Agent, Result, Swarm
from agents.utils import get_json_prompt
from agents.constraints_checker import planning_validate_constraints
from agents.prompts import (
    CONSTRAINT_ADDING_W_HISTORY,
    CONSTRAINT_ADDING_WO_HISTORY,
    INITIAL_PROMPT,
)
from agents.constraints_generator import update_constraints_and_query
from agents.postprocess_plan import parse_plan
from agents.symbolic_checker import (
    run_symbolic_check,
    get_current_constraint_result,
    format_violation_messages,
)


def extract_cities_from_plan(plan: Any) -> set:
    """
    플랜의 day별 current_city 문자열에서 도시 후보를 대충 뽑아냅니다.
    예: 'from Fayetteville to White Plains' -> {'Fayetteville', 'White Plains'}
        'New York' -> {'New York'}
    """
    cities = set()
    if not isinstance(plan, (list, tuple)):
        return cities

    for day in plan:
        if not isinstance(day, dict):
            continue
        cc = str(day.get("current_city", "")).strip()
        if not cc:
            continue

        # 패턴: "from A to B"
        m = re.findall(r"from\s+([^,]+?)\s+to\s+([^,]+)", cc, flags=re.IGNORECASE)
        if m:
            for a, b in m:
                cities.add(a.strip())
                cities.add(b.strip())
            continue

        # 쉼표가 있으면 마지막 토큰을 도시로 가정
        if "," in cc:
            cities.add(cc.split(",")[-1].strip())
        else:
            cities.add(cc.strip())

    return {c for c in cities if c}


def slice_ref_data_for_violations(
    ref_data: Any,
    plan: Any,
    max_per_type: int = 20,
) -> Any:
    """
    에러가 난 뒤 self-refine에서만 쓰기 위한 ref_data 슬라이스 함수.
    - plan에서 등장하는 도시들만 남기고
    - type(예: hotel/restaurant)별로 price 기준 상위 max_per_type개만 남김.

    ref_data가 DataFrame이 아니면 건드리지 않고 그대로 반환합니다.
    """
    if not isinstance(ref_data, pd.DataFrame):
        # 문자열이나 다른 타입이면 그대로 사용
        return ref_data

    cities = extract_cities_from_plan(plan)
    if not cities:
        return ref_data

    df = ref_data.copy()

    # 1) 도시 필터링 (실제 컬럼명에 맞게 'city' 부분 조정 필요)
    if "city" in df.columns:
        df = df[df["city"].isin(cities)]

    if df.empty:
        return ref_data  # 너무 적게 걸러졌으면 원본 사용

    # 2) 가격 기준 상위 N개만 남기기
    price_col_candidates = ["price", "avg_price", "price_per_unit", "cost"]
    price_col = None
    for col in price_col_candidates:
        if col in df.columns:
            price_col = col
            break

    if price_col is None:
        # 가격 컬럼을 못 찾으면 도시 필터링만 적용
        return df

    if "type" in df.columns:
        # type (예: 'hotel', 'restaurant') 별로 가장 싼 max_per_type개씩
        sliced_list = []
        for t, group in df.groupby("type"):
            g = group.sort_values(price_col, ascending=True).head(max_per_type)
            sliced_list.append(g)
        df_sliced = pd.concat(sliced_list, ignore_index=True)
    else:
        # type 컬럼이 없다면 그냥 전체에서 가장 싼 것들만
        df_sliced = df.sort_values(price_col, ascending=True).head(
            max_per_type * 3
        )

    return df_sliced






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
        # client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
        if client_kwargs:
            client_kwargs['base_url'] = "https://api.upstage.ai/v1"
            client = OpenAI(**client_kwargs)
        else:
            client = OpenAI(base_url="https://api.upstage.ai/v1")
        task = config['task']
        instructions = ""
        super().__init__(name=name, instructions=instructions, client=client)
        self.task = task
        self.evaluator_config = evaluator_config

    # 예전 pass-rate 계산 로직은 symbolic_checker로 통합했으므로 사용하지 않음
    """
        # self.constraints_dict = self.inital_constraints
        
    def check_constraint(self, constraints: str, response: str):
        if self.task == "test":
            current_constraint_score, total_score = planning_validate_constraints(
                turn_constraint=constraints,
                updated_constraints=self.constraints_dict,
                response_data=response,
                ref_data=self.ref_data,
            )
            correct_count = sum(
                1 for item in total_score.values() if item.get("is_correct") is True
            )
            total_count = sum(
                1 for item in total_score.values() if item.get("query_value") is not None
            )
            if total_count == 0:
                print("No constraints to evaluate")
                return current_constraint_score, total_score, 0
            total_pass_rate = correct_count / total_count
        return current_constraint_score, total_score, total_pass_rate
    """

    def check_constraint(self, constraints: str, response: Any):
        """
        LLM이 생성한 계획(response)에 대해 심볼릭 체커(run_symbolic_check)를 호출하여
        제약 만족 여부와 pass rate를 계산합니다.
        """
        if self.task == "test":
            check_result = run_symbolic_check(
                updated_constraints=self.constraints_dict,
                response_data=response,
                ref_data=self.ref_data,
            )

            total_score = check_result["total_score"]
            total_pass_rate = check_result["pass_rate"]

            print("Current Turn Constraint:", constraints)
            current_constraint_score = get_current_constraint_result(
                turn_constraint=constraints,
                total_score=total_score,
            )

            return current_constraint_score, total_score, total_pass_rate

        # task != "test" 인 경우 기본값
        return "No constraint check", {}, 0.0

    def check_quality(self, response) -> Result:
        final_quality_scores = "None"
        return final_quality_scores

    def conclude_evaluate(self, score: int, comments: str) -> Result:
        return Result(
            value=f"Interview concluded. Score: {score}\nComments: {comments}",
            context_variables={
                "interview_complete": True,
                "score": score,
                "comments": comments,
            },
        )


class Evaluatee(Agent):
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
            client = Together(api_key=evaluatee_config.get("client", {}).get("api_key"))
        else:
            # client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
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
            progress.add_task("Processing response...", total=None)
            full_params = {
                "model": agent.model,
                "messages": messages,
            }

            if json:
                full_params["response_format"] = {"type": "json_object"}

            full_params.update(chat_params)
            raw_response = agent.client.chat.completions.create(**full_params)
            content = raw_response.choices[0].message.content
            return Response(
                messages=[{"role": "assistant", "content": content}],
                agent=agent,
                context_variables={},
            )

    def add_message(self, speaker, content):
        if speaker == self.evaluator:
            self.evaluator_messages.extend(
                [{"role": "assistant", "content": content}]
            )
            self.evaluatee_messages.extend(
                [{"role": "user", "content": content}]
            )
        else:
            self.evaluator_messages.extend(
                [{"role": "user", "content": content}]
            )
            self.evaluatee_messages.extend(
                [{"role": "assistant", "content": content}]
            )

    def call_feedback_agent(self, response):
        constraints = (
            self.evaluator.new_constraints[self.questions_count - 1]
            if self.questions_count > 0
            else ""
        )
        current_score, total_score, total_pass_rate = self.evaluator.check_constraint(
            constraints, response
        )
        return current_score, total_score, total_pass_rate

    def call_question_agent(self):
        if self.questions_count == 0:
            query_to_display = self.evaluator.seed_question
            response = INITIAL_PROMPT.format(
                question=self.evaluator.seed_question,
                ref_data=self.evaluator.ref_data,
            )
        else:
            constraint_seed = self.evaluator.new_constraints[self.questions_count - 1]
            previous_condition = "\n".join(
                f"{key}: {value}"
                for key, value in self.evaluator.constraints_dict.items()
            )
            self.evaluator.constraints_dict, constraint = update_constraints_and_query(
                self.evaluator.constraints_dict,
                constraint_seed,
            )
            query_to_display = constraint

            if self.config["history"]:
                response = CONSTRAINT_ADDING_W_HISTORY.format(constraint=constraint)
            else:
                response = CONSTRAINT_ADDING_WO_HISTORY.format(
                    ref_data=self.evaluator.ref_data,
                    previous_condition=previous_condition,
                    additional_condition=constraint,
                    response=self.responses[-1],
                )
        return query_to_display, response

    def run(self) -> Dict[str, Any]:
        self.console.print(f"\n[info]Constraints {self.questions_count}[/info]")

        self.seed_question_used = True
        query_to_display, response = self.call_question_agent()

        self.questions.append(response)
        self.add_message(self.evaluator, response)
        self.display_message(self.evaluator.name, query_to_display)

        while self.questions_count <= self.max_constraints:
            # 1) 플랜 생성
            question = self.evaluator_messages[-1]["content"]

            response = self._get_response(
                self.evaluatee, self.evaluatee_messages, {}
            ).messages[-1]["content"]
            parsed_plan = parse_plan(response)
            if len(parsed_plan) == 0:
                self.console.print(
                    "\n[warning]No plan found in the response. Please provide a valid plan.[/warning]"
                )

            self.responses.append(parsed_plan)
            self.add_message(self.evaluatee, parsed_plan)
            self.display_message(self.evaluatee.name, str(parsed_plan))

            # 2) 심볼릭 체커
            current_score, total_score, total_pass_rate = self.call_feedback_agent(
                parsed_plan
            )

            self.scores["constraints"].append(current_score)
            self.scores["total_constraints_score"] = total_score
            self.scores["total_pass_rate"] = total_pass_rate
            self.all_total_scores.append(total_score)
            self.scores["all_total_constraints_score"] = self.all_total_scores

            self.display_message(
                "Constraint Checker",
                "Current_constraints_score: {}\nTotal_score: {}".format(
                    current_score, total_score
                ),
            )

            # 3) self-refine: 여러 번 돌리면서 best plan 유지
            if self.config.get("self_refine", False):
                max_refine_loops = self.config.get("self_refine_loops", 1)

                # 마지막 제약 턴이고, 아직 전부 만족하지 못했을 때만 self-refine 시작
                if (
                    self.questions_count == self.max_constraints
                    and total_pass_rate < 1.0
                ):
                    refine_round = 0

                    # 현재 플랜 / 점수
                    current_plan = parsed_plan
                    current_total_score = total_score
                    current_pass_rate = total_pass_rate

                    # 지금까지 중 가장 좋은 플랜
                    best_plan = parsed_plan
                    best_total_score = total_score
                    best_pass_rate = total_pass_rate

                    # best_pass_rate 기준으로 반복
                    while refine_round < max_refine_loops and best_pass_rate < 1.0:
                        # 위반된 제약 위주의 기본 피드백
                        feedback_text = format_violation_messages(
                            {"total_score": current_total_score},
                            show_satisfied=False,
                        )

                        # 어떤 제약이 깨졌는지 확인
                        violations = {
                            k: v
                            for k, v in current_total_score.items()
                            if v.get("is_correct") is False
                        }
                        budget_violated = "budget" in violations
                        cuisine_violated = "cuisine" in violations

                        # 현재 플랜 JSON 문자열화
                        try:
                            current_plan_json = json.dumps(
                                current_plan, ensure_ascii=False, indent=2
                            )
                        except Exception:
                            current_plan_json = str(current_plan)

                        people_count = self.evaluator.constraints_dict.get(
                            "people_number", "the group size"
                        )
                        budget_val = self.evaluator.constraints_dict.get(
                            "budget", "the budget"
                        )

                        # 여러 제약 위반 상황에서 강한 추가 지시문
                        extra_instructions = ""

                        if budget_violated:
                            extra_instructions += (
                                "\n[Budget-specific instructions]\n"
                                f"- For each day (Day 1 to Day {len(current_plan)}), "
                                "list the chosen accommodation and restaurants with their per-person prices "
                                "based on the reference information.\n"
                                "- Then explicitly compute:\n"
                                "    PerPersonCost = Flight_cost_per_person "
                                "+ sum(accommodation_cost_per_night) "
                                "+ sum(restaurant_cost_per_meal)\n"
                                f"    TotalTripCost = PerPersonCost * {people_count}\n"
                                f"- Verify that TotalTripCost <= {budget_val}. "
                                "If it is higher, replace the most expensive hotels/restaurants first "
                                "with cheaper options from the reference information.\n"
                                "\n[CRITICAL PRIORITY WARNING]\n"
                                "Budget is a HARD constraint. You MUST satisfy it.\n"
                                "If you cannot meet the budget while satisfying preferences (like Cuisine or Rating),\n"
                                "you MUST DROP the preferences (Soft constraints) to save money.\n"
                                "Action: Choose cheaper hotels/restaurants immediately, even if they don't match the preference.\n"
                                "- Show this step-by-step cost calculation BEFORE you output the final JSON plan.\n"
                            )

                        if cuisine_violated:
                            extra_instructions += (
                                "\n[Cuisine-specific instructions]\n"
                                "- Make sure the selected restaurants actually match the requested cuisine types.\n"
                                "- Prefer American or Mexican restaurants from the reference information; "
                                "avoid other cuisines unless necessary.\n"
                                "- Do NOT invent any new restaurant names that are not present in the reference information.\n"
                            )

                        # history=False 인 경우, self-refine 라운드에서만 ref_data를 잘라서 다시 붙여줌
                        ref_block = ""
                        if not self.config.get("history", False):
                            # budget / cuisine이 깨진 경우에만 슬라이스 사용
                            need_focus = any(
                                key in current_total_score
                                and current_total_score[key].get("is_correct") is False
                                for key in ("budget", "cuisine")
                            )

                            ref_for_prompt = self.evaluator.ref_data
                            if self.config.get("slice_ref_for_violations", False) and need_focus:
                                ref_for_prompt = slice_ref_data_for_violations(
                                    self.evaluator.ref_data,
                                    current_plan,
                                    max_per_type=self.config.get(
                                        "slice_max_per_type", 20
                                    ),
                                )

                            ref_block = (
                                "[Reference information]:\n"
                                "All costs are per one person, one night.\n"
                                f"{ref_for_prompt}\n\n"
                            )

                        revise_instruction = (
                            f"{ref_block}"
                            f"[Self-Refine Round {refine_round + 1}]\n"
                            "The above travel plan does not satisfy all constraints.\n"
                            "Here is feedback from a symbolic constraint checker:\n"
                            f"{feedback_text}\n\n"
                            "You MUST revise your previous plan so that it satisfies all violated constraints.\n"
                            "Important rules:\n"
                            "1) Keep the number of days, city sequence, and people consistent with the original request.\n"
                            "2) For each violated constraint, carefully read the error message and fix the plan accordingly.\n"
                            f"3) For budget-related violations, adjust your choice of flights, accommodations, and restaurants so that\n"
                            f"   the total cost for {people_count} people clearly fits within the budget of {budget_val}.\n"
                            "4) Avoid using obviously unrealistic or overly luxurious options when the budget is small.\n"
                            "5) Do NOT change the requested trip length or the origin/destination cities.\n"
                            "6) Return ONLY the revised plan in the SAME JSON format as before (list of day-wise dictionaries).\n"
                            f"{extra_instructions}"
                            "\nHere is your previous plan in JSON format:\n"
                            f"{current_plan_json}\n"
                        )

                        self.add_message(self.evaluator, revise_instruction)
                        self.display_message(self.evaluator.name, revise_instruction)

                        revised_response = self._get_response(
                            self.evaluatee, self.evaluatee_messages, {}
                        ).messages[-1]["content"]
                        revised_plan = parse_plan(revised_response)

                        if len(revised_plan) == 0:
                            self.console.print(
                                f"\n[warning]Refine round {refine_round + 1}: "
                                "no valid plan parsed. Stopping self-refine.\n"
                            )
                            break

                        current_plan = revised_plan
                        self.responses[-1] = revised_plan
                        self.add_message(self.evaluatee, revised_plan)
                        self.display_message(self.evaluatee.name, str(revised_plan))

                        current_score, current_total_score, current_pass_rate = (
                            self.call_feedback_agent(revised_plan)
                        )

                        # 이번 라운드 결과가 지금까지 중 최고면 best 갱신
                        if current_pass_rate >= best_pass_rate:
                            best_plan = revised_plan
                            best_total_score = current_total_score
                            best_pass_rate = current_pass_rate

                        self.display_message(
                            "Constraint Checker",
                            f"After revision round {refine_round + 1} - "
                            f"Current_constraints_score: {current_score}\n"
                            f"Total_score: {current_total_score}\n"
                            f"Pass_rate (this round): {current_pass_rate}\n"
                            f"Best_pass_rate so far: {best_pass_rate}",
                        )

                        refine_round += 1

                    # self-refine 종료 후: 항상 best 플랜/점수로 최종 반영
                    self.responses[-1] = best_plan
                    self.scores["total_constraints_score"] = best_total_score
                    self.scores["total_pass_rate"] = best_pass_rate
                    self.all_total_scores[-1] = best_total_score
                    self.scores["all_total_constraints_score"] = self.all_total_scores

                    # 바깥 루프에서도 best 기준으로 계속 사용
                    total_score = best_total_score
                    total_pass_rate = best_pass_rate

            # 4) 다음 제약 턴으로 이동
            self.questions_count += 1
            if self.questions_count <= self.max_constraints:
                self.console.print(f"\n[info]Constraints {self.questions_count}")
                query_to_display, response = self.call_question_agent()
                self.questions.append(response)
                self.add_message(self.evaluator, response)

                rm_txt = (
                    "[Reference information]:\nAll costs are per one person, one night.\n"
                    + str(self.evaluator.ref_data)
                    + "\n\n"
                )
                display_response = response.replace(rm_txt, "")
                self.display_message(self.evaluator.name, display_response)

            # 5) 종료 조건
            if self.questions_count == (self.max_constraints + 1):
                final_message = "Maximum number of constraints reached. Concluding Evaluation."
                self.console.print(f"\n[warning]{final_message}[/warning]")

                # self-refine까지 포함한 최종 pass rate 사용
                self.scores["quality"] = self.scores["total_pass_rate"]
                panel_content = f"Total Pass Rate: {self.scores['total_pass_rate']}"

                self.display_message("Overall Quality Checker", panel_content)
                break

        results = {
            "constraint_counts": sum(
                1
                for key, item in self.scores["total_constraints_score"].items()
                if item.get("is_correct") is not None and key != 'symbolic_rule'
            ),
            "score": self.scores["total_pass_rate"],
            "questions_asked": self.questions_count - 1,
            "all_total_scores": self.all_total_scores,
        }

        self.display_results(results)
        return results
