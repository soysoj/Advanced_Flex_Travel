# 추가) 과거 Constraint reinjection
CONSTRAINT_MEMORY_BLOCK = """
[CONSTRAINT MEMORY]
{memory_lines}
[/CONSTRAINT MEMORY]

"""

def build_constraint_memory(global_constraints, new_constraints, priority_map = None):
    lines = []
    for k,v in global_constraints.items():
        if v is None:
            continue
        p = f"(priority={priority_map.get(k):.2f})" if priority_map else ""
        lines.append(f"-{k}:{v}{p}")

    for nc in new_constraints:
        for k,v in nc.items():
            p = f"(priority={priority_map.get(k):.2f})" if priority_map else ""
            lines.append(f"- {k}: {v}{p}")

    return CONSTRAINT_MEMORY_BLOCK.format(memory_lines = "\n".join(lines))
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


# 추가) Priority Weighting
PRIORITY_BLOCK = """
[PRIORITY GUIDE]
Each constraint has an importance weight (0–1). The larger the value, the higher the priority.
Hard constraints (e.g., budget, room type such as "not shared room") must always be satisfied. If any are violated, revise the plan.
Soft constraints (e.g., cuisine preference, rating preference, etc.) should be satisfied as much as possible according to their weight ratio.
Weight Table:
{priority_lines}
[/PRIORITY GUIDE]
"""

def build_priority_block(priority_map):
    """
    Build priority guide block from priority map.
    
    Args:
        priority_map: Dictionary mapping constraint names to weights (0-1)
        
    Returns:
        Formatted priority guide string, or empty string if no priority map
    """
    if not priority_map:
        return ""
    lines = [f"- {k}: weight={float(w):.2f}" for k, w in priority_map.items()]
    return PRIORITY_BLOCK.format(priority_lines="\n".join(lines))
###

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

[OUTPUT FORMAT]
Output your evaluation in JSON (first), then the plan (second). The JSON must be a single object:

{
  "hard_pass": 0 or 1,
  "soft_coverage": number between 0 and 1 (two decimals),
  "commonsense": number between 0 and 1 (two decimals),
  "diversity": number between 0 and 1 (two decimals),
  "total": number between 0 and 1 (two decimals),
  "explanations": {
    "hard_pass": "which mandatory constraints are (not) satisfied and why",
    "soft_coverage": "which preferences counted as satisfied (with counts)",
    "commonsense": "violations found or 'none'",
    "diversity": "categories detected and U/T"
  }
}

If total < 0.85, REVISE the plan ONCE to increase the lowest-scoring component(s), then re-run the calculations and output:
1) Self-evaluation JSON (final)
2) Revised plan in the required JSON plan format

"""
###