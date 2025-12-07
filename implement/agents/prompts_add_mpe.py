# 추가) 과거 Constraint reinjection
CONSTRAINT_BLOCK_TEMPLATE = """
[CURRENT CONSTRAINTS LIST]
The following list summarizes the constraints you must satisfy. 
If there is the priority system, you must satisfy the constraints based on the priority system below:
{memory_lines}
"""
###

#추가) Priority block
## A) Numerical Weight.
GUIDELINE_NUMERICAL = """
[PRIORITY SYSTEM INSTRUCTION]
The following list summarizes the constraints you must satisfy.
The 'priority' vlaue (0.0 to 1.0) indicates the strictness of the constraint.
- Priority 0.8~1.0: CRITICAL /HARD constraint (Must be satisfied)
- Priority 0.5~0.8: IMPORTANT constraint (Try hard to satisfy)
- Priority ~0.5: PREFERENCE constraint (Satisfy if possible, but tradable)

{priority_lines}
"""
## B) NATURAL LANGUAGE LABEL
GUIDELINE_LABEL = """
[PRIORITY SYSTEM INSTRUCTION]
The following list summarizes the constraints.
Each constraint has an Importance Level:
- [CRITICAL] : Absolute hard constraints. You MUST satisfy this. If violated, the plan is invalid.
- [HIGH] : Very Important constraints. Prioritize this over [MEDIUM] or [LOW].
- [MEDIUM] : Preferences. Try to satisfy if possible, but you can compromise if it conflicts with higher priorities.
- [LOW] : Optional. Nice to have, but the first to be discarded in conflicts.

{priority_lines}
"""

## C) RANK 
GUIDELINE_RANK_ONLY = """
[PRIORITY SYSTEM INSTRUCTION]
The following list summarizes the constraints.
Each constraint is assigned a Rank:
- Lower Rank number = Higher Priority.
- Rank 1 is the most critical constraint.
- Higher rank numbers (e.g., Rank 5, 6) are less important and can be compromised if necessary.

{priority_lines}
"""

## D) HYBRID (Label + RANK)
GUIDELINE_HYBRID_RANK = """
[PRIORITY SYSTEM INSTRUCTION]
The following list summarizes the constraints.
Each constraint has an Importance Level and a Rank:
1. Importance Levels:
- [CRITICAL] : Absolute hard constraints. You MUST satisfy this. If violated, the plan is invalid.
- [HIGH] : Very Important constraints. Prioritize this over [MEDIUM] or [LOW].
- [MEDIUM] : Preferences. Try to satisfy if possible, but you can compromise if it conflicts with higher priorities.
- [LOW] : Optional. Nice to have, but the first to be discarded in conflicts.

2. Tie-Breaking Rule:
   - If two constraints have the same Importance Level, check the Rank.
   - Lower Rank number = Higher Priority (e.g., Rank 2 beats Rank 3).

{priority_lines}
"""
## E) HYBRID (Label + Weight)
GUIDELINE_HYBRID_WEIGHT = """
[PRIORITY SYSTEM INSTRUCTION]
The following list summarizes the constraints.
Each constraint has an Importance Level and a Priority Score:

1. Importance Levels:
   - [CRITICAL]: Absolute hard constraints. Must be satisfied.
   - [HIGH]: Very important. Prioritize over Medium/Low.
   - [MEDIUM]: Preferences. Compromise possible.
   - [LOW]: Optional. First to be discarded.

2. Priority Score (0.0 ~ 1.0):
   - Higher score means higher priority.
   - Use this score to resolve conflicts between constraints with the same Importance Level.

{priority_lines}
"""
#####

# 추가) Self eval
SELF_EVAL_PROMPT = """
Before finalizing your answer, evaluate the plan using the following four metrics.
Use the RULES below to compute each metric numerically (deterministically).

[RULES]

A. HARD PASS (0 or 1)
- Set to 1 only if ALL mandatory constraints are satisfied:
  • Budget: total trip cost ≤ user budget
  • House rules (e.g., “No children under 10”, “No pets”)
  • Room type (e.g., “not shared room”, “Private room”)
  • Transportation feasibility (e.g., city-to-city legs possible via provided reference data or self-driving/taxi with plausible distance/time)
- Otherwise 0.

B. SOFT COVERAGE (0~1)
- Preference constraints only (e.g., cuisine, rating).
- Compute as: (# satisfied preferences) / (total # preferences)
  • Cuisine: count how many days (or meals) match requested cuisines; mark satisfied if it reaches the requested count n.
  • Rating: count restaurants with rating ≥ threshold; mark satisfied if it reaches n.
- If no preferences exist, set to 1.0.

C. COMMONSENSE (0~1)
- Start at 1.0, subtract 0.25 for each violation below (floor at 0):
  • Impossible route (e.g., no flight/self-driving option given between cities and you still move)
  • Overpacked day (e.g., >4 major attractions + cross-city travel in one day)
  • Time inconsistency (arrive after you visit, or double-book meals/accommodation)
  • Budget obviously inconsistent with itemized choices (e.g., repeated ultra-high prices when budget is low)

D. DIVERSITY (0~1)
- Compute unique activity categories per trip (e.g., museum/park/landmark/beach/food/market/theme-park/children-activity/hiking).
- Let U = number of unique categories across all days.
- Let T = min(6, total possible categories visible in reference for the cities).
- Diversity = min(1.0, U / T).
- If trip is 1–2 days, cap T at 4.

[TOTAL SCORE]
- If HARD PASS == 0 → TOTAL = 0.49 (cap; cannot be high without mandatory constraints)
- Else TOTAL = 0.4*HARD_PASS + 0.25*SOFT_COVERAGE + 0.2*COMMONSENSE + 0.15*DIVERSITY
  (This keeps TOTAL ∈ [0,1] and prioritizes mandatory constraints)

If total < 0.85, REVISE the plan ONCE to increase the lowest-scoring component(s), then re-run the calculations and output:
"""
###

### 추가) Helper Functions
def get_priority_guideline(mode):
    """모드에 맞는 가이드라인 텍스트 반환"""
    if mode == "hybrid_rank": return GUIDELINE_HYBRID_RANK
    elif mode == "hybrid_weight": return GUIDELINE_HYBRID_WEIGHT
    elif mode == "label": return GUIDELINE_LABEL
    elif mode == "rank_only": return GUIDELINE_RANK_ONLY
    elif mode == "numerical": return GUIDELINE_NUMERICAL
    else: return ""

#추가) build_memory_lines
def build_constraint_memory(previous_constraints, new_constraints, priority_info=None):
    """
    priority_info: 
      - Numerical/Label 모드일 때: {'budget': 1.0} or {'budget': '[HIGH]'}
      - Hybrid 모드일 때: {'budget': {'label': '[HIGH]', 'rank': 2}}
    """
    lines = []
    
    # 데이터 유효성 및 타입감지
    use_priority = (priority_info is not None and len(priority_info) > 0)
    mode = "basic"
    
    if use_priority:
        first_val = next(iter(priority_info.values()))
        if isinstance(first_val, dict):
            if 'weight' in first_val:
                mode = "hybrid_weight"
            else:
                mode = "hybrid_rank"
        elif isinstance(first_val, str):
            mode = "label"   
        elif isinstance(first_val, int):
            mode = "rank_only"
        else:
            mode = "numerical"

    # 포맷팅 헬퍼함수
    def get_p_str(key):
        if not use_priority: return ""
        val = priority_info.get(key)
        if val is None: return ""

        if isinstance(val, dict) and val.get('label') == "[CRITICAL]":
            return " (Importance: [CRITICAL])"

        if mode == "hybrid_rank":
            return f" (Importance: {val['label']} - Rank {val['rank']})"
        elif mode == "hybrid_weight":
            return f" (Importance: {val['label']} - Priority {val['weight']:.2f})"
        elif mode == "label":
            return f" (Importance: {val})"
        elif mode == "rank_only":
            return f" (Rank {val})"
        else: # numerical
            return f" (priority={val:.2f})"

    # 1. Previous Constraints
    for k, v in previous_constraints.items():
        if v in [None, ""]: continue
        lines.append(f"- {k}: {v}{get_p_str(k)}")

    # 2. New Constraints
    for nc in new_constraints:
        if isinstance(nc, dict):
            for k, v in nc.items():
                if v in [None, ""]: continue
                lines.append(f"- {k}: {v}{get_p_str(k)}")

    if not lines:
        return ""

    return CONSTRAINT_BLOCK_TEMPLATE.format(memory_lines="\n".join(lines))
###

PREFERENCE_RATING = "I prefer to visit a restaurant with a minimum rating of {rating} at least {n} times if possible."
PREFERENCE_CUISINE = "I prefer to try a {cuisine} place at least {n} times if possible."
CONSTRAINT_ADDING_W_HISTORY = '''Revise the previous plan to additionally satisfy the newly provided condition: {constraint} Please refer to the given reference information only. \n[Plan]: '''

CONSTRAINT_ADDING_WO_HISTORY = '''[Reference information]:
All costs are per one person for restaurants, and per one night for accommodations.
{ref_data}

The following plan is designed to satisfy the previously given condition: \n{previous_condition}\n
Revise it to additionally satisfy the newly provided condition: {additional_condition} \nPlease refer to the given reference information only. \n[Plan]: {response}'''

PREFERENCE_RATING = "I prefer to visit a restaurant with a rating of more than {rating} at least {n} times if possible."
PREFERENCE_CUISINE = "I prefer to try a {cuisine} place at least {n} times if possible."

PLAN_PARSING = """Please assist me in extracting valid information from a given natural language text and reconstructing it in JSON format, as demonstrated in the following example. If transportation details indicate a journey from one city to another (e.g., from A to B), the 'current_city' should be updated to the destination city (in this case, B). Use a ';' to separate different attractions, with each attraction formatted as 'Name, City'. If there's information about transportation, ensure that the 'current_city' aligns with the destination mentioned in the transportation details (i.e., the current city should follow the format 'from A to B'). Also, ensure that all flight numbers and costs are followed by a colon (i.e., 'Flight Number:' and 'Cost:'), consistent with the provided example. Each item should include ['day', 'current_city', 'transportation', 'breakfast', 'attraction', 'lunch', 'dinner', 'accommodation']. Replace non-specific information like 'eat at home/on the road' with '-'. Additionally, delete any '$' symbols and Cost information (e.g., Cost: 46). Just parse the Plan section of the text.
-----EXAMPLE-----
[{{
        "days": 1,
        "current_city": "from Dallas to Peoria",
        "transportation": "Flight Number: 4044830, from Dallas to Peoria, Departure Time: 13:10, Arrival Time: 15:01",
        "breakfast": "-",
        "attraction": "Peoria Historical Society, Peoria;Peoria Holocaust Memorial, Peoria;",
        "lunch": "-",
        "dinner": "Tandoor Ka Zaika, Peoria",
        "accommodation": "Bushwick Music Mansion, Peoria"
    }},
    {{
        "days": 2,
        "current_city": "Peoria",
        "transportation": "-",
        "breakfast": "Tandoor Ka Zaika, Peoria",
        "attraction": "Peoria Riverfront Park, Peoria;The Peoria PlayHouse, Peoria;Glen Oak Park, Peoria;",
        "lunch": "Cafe Hashtag LoL, Peoria",
        "dinner": "The Curzon Room - Maidens Hotel, Peoria",
        "accommodation": "Bushwick Music Mansion, Peoria"
    }},
    {{
        "days": 3,
        "current_city": "from Peoria to Dallas",
        "transportation": "Flight Number: 4045904, from Peoria to Dallas, Departure Time: 07:09, Arrival Time: 09:20",
        "breakfast": "-",
        "attraction": "-",
        "lunch": "-",
        "dinner": "-",
        "accommodation": "-"
    }}]
-----EXAMPLE END-----
"""

INITIAL_PROMPT = '''
[Reference information]:
All costs are per one person, one night.
{ref_data}

[Plan Format]: [{{
        "days": 1,
        "current_city": "from Dallas to Peoria",
        "transportation": "Flight Number: 4044830, from Dallas to Peoria, Departure Time: 13:10, Arrival Time: 15:01",
        "breakfast": "-",
        "attraction": "Peoria Historical Society, Peoria;Peoria Holocaust Memorial, Peoria;",
        "lunch": "-",
        "dinner": "Tandoor Ka Zaika, Peoria",
        "accommodation": "Bushwick Music Mansion, Peoria"
    }},
    {{
        "days": 2,
        "current_city": "Peoria",
        "transportation": "-",
        "breakfast": "Tandoor Ka Zaika, Peoria",
        "attraction": "Peoria Riverfront Park, Peoria;The Peoria PlayHouse, Peoria;Glen Oak Park, Peoria;",
        "lunch": "Cafe Hashtag LoL, Peoria",
        "dinner": "The Curzon Room - Maidens Hotel, Peoria",
        "accommodation": "Bushwick Music Mansion, Peoria"
    }},
    {{
        "days": 3,
        "current_city": "from Peoria to Dallas",
        "transportation": "Flight Number: 4045904, from Peoria to Dallas, Departure Time: 07:09, Arrival Time: 09:20",
        "breakfast": "-",
        "attraction": "-",
        "lunch": "-",
        "dinner": "-",
        "accommodation": "-"
    }}]

{question}
Please refer to the given reference information only.

[Plan]:
'''
 
    

