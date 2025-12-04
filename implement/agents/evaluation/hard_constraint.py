import os
import sys
import json
import math
import re
import numpy as np
from tqdm import tqdm
import argparse
import unicodedata

def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data

def convert_bool_values(item):
    if isinstance(item, dict):
        return {key: convert_bool_values(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [convert_bool_values(value) for value in item]
    elif isinstance(item, tuple):
        return tuple(convert_bool_values(value) for value in item)
    elif isinstance(item, np.bool_):
        return bool(item)
    else:
        return item

def extract_from_to(text: str):
    pattern = r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)"
    matches = re.search(pattern, text)
    return matches.groups() if matches else (None, None)

def get_valid_name_city(text):
    if '(' in text:
        name = text.split('(')[0].strip()
        city = text.split('(')[1].split(')')[0].strip()
        return name, city
    elif ',' in text:
        parts = text.rsplit(',', 1)
        name = parts[0].strip()
        city = parts[1].strip() if len(parts) > 1 else None
        return name, city
    return text, None

# def preprocess_text(text):
#     """
#     Preprocess text for consistent comparison.
    
#     Args:
#         text (str): Input text to preprocess
    
#     Returns:
#         str: Cleaned text
#     """
#     # Replace common punctuation variations
#     text = re.sub(r"[‘’´`]", "'", text)  # Normalize single quotes
#     text = re.sub(r'[“”]', '"', text)    # Normalize double quotes
    
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     return text

def preprocess_text(text):
    """
    Preprocess text for consistent comparison with Unicode normalization.
    
    Args:
        text (str): Input text to preprocess
    
    Returns:
        str: Cleaned and normalized text
    """
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Replace common punctuation variations
    text = re.sub(r"[''´`]", "'", text)  # Normalize single quotes
    text = re.sub(r'[""]', '"', text)    # Normalize double quotes
    
    # Remove diacritical marks
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()

def get_ngrams(text, n=2):
    """
    Generate n-grams from input text.
    
    Args:
        text (str): Input text to generate n-grams from
        n (int): Size of n-grams (default: 2)
    
    Returns:
        list: List of n-grams
    """
    # Convert to lowercase and split into tokens
    tokens = text.lower().split()
    
    # Generate n-grams
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i + n])
        ngrams.append(ngram)
    
    return ngrams

# def calculate_ngram_similarity(seq1, seq2, n=2):
#     """
#     Calculate n-gram similarity between two sequences.
    
#     Args:
#         seq1 (str): First sequence
#         seq2 (str): Second sequence
#         n (int): Size of n-grams (default: 2)
    
#     Returns:
#         dict: Dictionary containing match statistics
#     """
#     # Preprocess input sequences
#     seq1 = preprocess_text(seq1)
#     seq2 = preprocess_text(seq2)

#     # Generate n-grams for both sequences
#     ngrams1 = get_ngrams(seq1, n)
#     ngrams2 = get_ngrams(seq2, n)
    
#     # Find matching n-grams
#     matches = set(ngrams1) & set(ngrams2)
    
#     # Calculate statistics
#     total_ngrams = max(len(ngrams1), len(ngrams2))
#     match_count = len(matches)
    
#     similarity_score = match_count / total_ngrams if total_ngrams > 0 else 0
    
#     return {
#         'matching_ngrams': list(matches),
#         'match_count': match_count,
#         'total_ngrams': total_ngrams,
#         'similarity_score': similarity_score,
#         'ngrams_seq1': ngrams1,
#         'ngrams_seq2': ngrams2
#     }

def calculate_ngram_similarity(seq1, seq2, n=2):
    """
    Calculate n-gram similarity between two sequences with Unicode handling.
    
    Args:
        seq1 (str): First sequence
        seq2 (str): Second sequence
        n (int): Size of n-grams (default: 2)
    
    Returns:
        dict: Dictionary containing match statistics
    """
    # Preprocess input sequences
    seq1 = preprocess_text(seq1)
    seq2 = preprocess_text(seq2)

    # Generate n-grams for both sequences
    ngrams1 = get_ngrams(seq1, n)
    ngrams2 = get_ngrams(seq2, n)
    
    # Find matching n-grams
    matches = set(ngrams1) & set(ngrams2)
    
    # Calculate statistics
    total_ngrams = max(len(ngrams1), len(ngrams2))
    match_count = len(matches)
    
    similarity_score = match_count / total_ngrams if total_ngrams > 0 else 0
    
    return {
        'matching_ngrams': list(matches),
        'match_count': match_count,
        'total_ngrams': total_ngrams,
        'similarity_score': similarity_score,
        'ngrams_seq1': ngrams1,
        'ngrams_seq2': ngrams2
    }

def find_reference_info(ref_data, item_type, name, city=None):
    """
    Finds reference information from structured data based on item type and name.
    For non-flight items, requires city parameter.
    For flights, only uses flight number to search.
    
    Parameters:
    ref_data (dict): Dictionary containing reference data with dynamic keys
    item_type (str): Type of item to search for ('restaurant', 'accommodation', 'attraction', 'flight')
    name (str): Name of the specific item to find (or flight number for flights)
    city (str, optional): City name for the search (required for non-flight items)
    
    Returns:
    dict or None: Reference information if found, None otherwise
    """
    if item_type == 'distance':
        # Handle transportation methods (self-driving and taxi)
        dest_city = city  # In this case, 'name' is origin city, 'city' is destination
        transport_key = f'Self-driving from {name} to {dest_city}'
        taxi_key = f'Taxi from {name} to {dest_city}'
        
        transport_info = {}
        if transport_key in ref_data:
            transport_info['self-driving'] = ref_data[transport_key]
        if taxi_key in ref_data:
            transport_info['taxi'] = ref_data[taxi_key]
            
        return transport_info if transport_info else None

    elif item_type == 'flight':
        # For flights, search all flight-related keys using only flight number
        flight_keys = [key for key in ref_data.keys() if key.startswith('Flight from')]
        for key in flight_keys:
            flight_list = ref_data[key]
            if not flight_list:
                return None
            elif type(flight_list) == str and flight_list.startswith('There is no flight'):
                # get avg cost of other flights
                return None
            else:
                for flight in flight_list:
                    if type(flight) != dict:
                        flight = eval(flight)
                    if flight['Flight Number'] == name:
                        return flight
    else:
        # Define name field mapping for different types
        name_fields = {
            'restaurant': 'Name',
            'accommodation': 'NAME',
            'attraction': 'Name'
        }
        
        # Create mapping of item types to their corresponding dictionary keys
        if city: 
            # Create mapping of item types to their corresponding dictionary keys
            type_patterns = {
                'restaurant': f'Restaurants in {city}',
                'accommodation': f'Accommodations in {city}',
                'attraction': f'Attractions in {city}'
            }
            # Get the relevant data list based on item type
            key = type_patterns.get(item_type)
            if key not in ref_data:
                return None
                
            items_list = ref_data[key]
            
            # Get the correct name field for this type
            name_field = name_fields.get(item_type)
            if not name_field:
                return None
                
            # Search for the item in the list
            for item in items_list:
                if item.get(name_field) == name:
                    return item
            if item == None and item_type == 'accommodation':
                for item in items_list:
                    if calculate_ngram_similarity(name, item[name_field])['match_count'] > 1:
                        return item
            elif item == None and item_type == 'restaurant':
                for item in items_list:
                    if calculate_ngram_similarity(name, item[name_field], n=1)['match_count'] > 1:
                        return item
        else:
            type_patterns = {
                'restaurant': [key for key in ref_data.keys() if key.startswith('Restaurants in ')],
                'accommodation': [key for key in ref_data.keys() if key.startswith('Accommodations in ')],
                'attraction': [key for key in ref_data.keys() if key.startswith('Attractions in ')]
            }
            # Get all relevant data lists for this item type
            keys = type_patterns.get(item_type, [])
            if not keys:
                return None
            
            name_field = name_fields.get(item_type)
            if not name_field:
                return None
            #  Search through all cities' data
            for key in keys:
                items_list = ref_data[key]
                # Search for exact name match
                for item in items_list:
                    if item.get(name_field) == name:
                        return item
                
                # If no exact match, try fuzzy matching based on item type
                if item_type == 'accommodation':
                    for item in items_list:
                        if calculate_ngram_similarity(name, item[name_field])['match_count'] > 1:
                            return item
                elif item_type == 'restaurant':
                    for item in items_list:
                        if calculate_ngram_similarity(name, item[name_field], n=1)['match_count'] > 1:
                            return item
    
    return None

def get_avg_cost(item_type, city, ref_data):
    """
    Get the average cost of items in a given city from reference data.
    
    Args:
        item_type (str): Type of item to search for ('restaurant', 'accommodation')
        city (str): City name to search for
        ref_data (dict): Reference data dictionary
    
    Returns:
        float: Average cost of items in the city
    """

    cost_field = 'Average Cost' if item_type == 'restaurant' else 'price'
    
    # Try to get city-specific data
    key = f'{item_type.capitalize()}s in {city}'  # e.g., 'Restaurants in New York'
    items_list = ref_data.get(key, [])
    
    # If city-specific data is found, calculate the average cost
    if items_list:
        costs = [item[cost_field] for item in items_list if cost_field in item]
        if costs:
            return round(np.mean(costs))
        
    # Fallback: Calculate average cost across all cities
    all_items = []
    for city_key in ref_data.keys():
        if city_key.startswith(f'{item_type.capitalize()}'):
            all_items.extend(ref_data[city_key])
    
    if all_items:
        costs = [item[cost_field] for item in all_items if cost_field in item]
        if costs:
            return round(np.mean(costs))

def get_city_list_from_reference_data(ref_data):
    city_list = []
    for key in ref_data.keys():
        if key.startswith('Restaurants in') or key.startswith('Accommodations in'):
            city_list.append(key.rsplit(' in ', 1)[-1].strip())
    return city_list

### 정현: TravelPlanner.st 예산 계산에 사용할 multiplier 정의 ###
BUDGET_MULTIPLIERS = {
    1: {"flight": 2, "hotel": 1, "restaurant": 3},
    3: {"flight": 2, "hotel": 3, "restaurant": 9},
    5: {"flight": 3, "hotel": 5, "restaurant": 15},
    7: {"flight": 4, "hotel": 7, "restaurant": 21},
}


# [수정 후] 딕셔너리가 아니면(숫자 등) 무시하도록 강화
def _ensure_dict(record):
    if isinstance(record, dict):
        return record
    try:
        # eval 결과가 딕셔너리일 때만 반환하고, 숫자면 None 반환
        result = eval(record)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return None


### 정현: 항공편/숙소/식당 가격 샘플을 수집하는 헬퍼 ###
def _collect_flight_prices(reference_data, org, dest):
    prices = []
    preferred_keys = [f"Flight from {org} to {dest}", f"Flight from {dest} to {org}"]
    keys = preferred_keys + [k for k in reference_data.keys() if k.startswith("Flight from")]
    seen = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        for record in reference_data.get(key, []):
            rec = _ensure_dict(record)
            if not rec:
                continue
            price = rec.get("Price")
            try:
                if price is not None:
                    prices.append(float(price))
            except Exception:
                continue
    return prices


def _collect_accommodation_prices(reference_data, city, room_type=None, house_rule=None):
    prices = []
    candidate_keys = [f"Accommodations in {city}"]
    candidate_keys += [k for k in reference_data.keys() if k.startswith("Accommodations in ")]
    seen = set()
    for key in candidate_keys:
        if key in seen:
            continue
        seen.add(key)
        for record in reference_data.get(key, []):
            rec = _ensure_dict(record)
            if not rec:
                continue
            if room_type and rec.get("room type") != room_type:
                continue
            if house_rule and rec.get("house_rules") and house_rule in rec.get("house_rules"):
                continue
            price = rec.get("price")
            try:
                if price is not None:
                    prices.append(float(price))
            except Exception:
                continue
    return prices


def _collect_restaurant_prices(reference_data, city, cuisines=None):
    prices = []
    keys = [f"Restaurants in {city}"]
    keys += [k for k in reference_data.keys() if k.startswith("Restaurants in ")]
    seen = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        for record in reference_data.get(key, []):
            rec = _ensure_dict(record)
            if not rec:
                continue
            if cuisines:
                cuisine_field = rec.get("Cuisines", "")
                if not any(c in cuisine_field for c in cuisines):
                    continue
            price = rec.get("Average Cost")
            try:
                if price is not None:
                    prices.append(float(price))
            except Exception:
                continue
    return prices


def _calculate_price_stat(prices, mode):
    if not prices:
        return None
    if mode == "lowest":
        return min(prices)
    if mode == "highest":
        return max(prices)
    if mode == "average":
        return sum(prices) / len(prices)
    raise ValueError("Unknown mode")


### 정현: reference 기반 TravelPlanner 스타일 예산 추정 ###
def estimate_symbolic_budget(query_data, reference_data):
    days = query_data.get("days")
    multipliers = BUDGET_MULTIPLIERS.get(days)
    if not multipliers:
        return None

    local_constraint = query_data.get("local_constraint", {}) or {}
    org = query_data.get("org")
    dest = query_data.get("dest")
    if not (org and dest):
        return None

    flight_prices = []
    if local_constraint.get("transportation") != "no flight":
        flight_prices = _collect_flight_prices(reference_data, org, dest)

    room_type = local_constraint.get("room type")
    house_rule = local_constraint.get("house rule")
    accommodation_prices = _collect_accommodation_prices(reference_data, dest, room_type, house_rule)

    cuisines = local_constraint.get("cuisine")
    restaurant_prices = _collect_restaurant_prices(reference_data, dest, cuisines)

    budgets = {}
    for mode in ("lowest", "highest", "average"):
        flight_cost = _calculate_price_stat(flight_prices, mode) or 0
        hotel_cost = _calculate_price_stat(accommodation_prices, mode)
        restaurant_cost = _calculate_price_stat(restaurant_prices, mode)

        if hotel_cost is None or restaurant_cost is None:
            return None

        total = (
            flight_cost * multipliers["flight"]
            + hotel_cost * multipliers["hotel"]
            + restaurant_cost * multipliers["restaurant"]
        )
        budgets[mode] = round(total, 2)

    return budgets


def _parse_flight_key(key):
    match = re.match(r"Flight from (.+?) to (.+?)(?: on (.+))?$", key)
    if match:
        origin = match.group(1).strip()
        dest = match.group(2).strip()
        date = match.group(3).strip() if match.group(3) else None
        return origin, dest, date
    return None, None, None


### 정현: TravelPlanner feasibility 룰 구현 ###
def travelplanner_feasibility_check(query, reference_data):
    org = query["org"]
    dest = query["dest"]
    days = query["days"]
    date_list = query.get("date", [])
    if isinstance(date_list, str):
        try:
            date_list = eval(date_list)
        except Exception:
            date_list = []
    local = query.get("local_constraint", {}) or {}
    ppl = query.get("people_number")

    flight_records = []
    for key, vals in reference_data.items():
        if not key.startswith("Flight from"):
            continue
        origin, destination, key_date = _parse_flight_key(key)
        if origin != org or destination != dest:
            continue
        for f in vals:
            fd = _ensure_dict(f)
            if not fd:
                continue
            flight_date = fd.get("FlightDate") or key_date
            if date_list and flight_date not in date_list:
                continue
            flight_records.append(fd)

    hotel_records = []
    hotel_key = f"Accommodations in {dest}"
    for key, vals in reference_data.items():
        if key == hotel_key:
            for h in vals:
                hd = _ensure_dict(h)
                if hd:
                    hotel_records.append(hd)

    restaurant_records = []
    rest_key = f"Restaurants in {dest}"
    for key, vals in reference_data.items():
        if key == rest_key:
            for r in vals:
                rd = _ensure_dict(r)
                if rd:
                    restaurant_records.append(rd)

    if local.get("transportation") == "no self-driving":
        if len(flight_records) < 2:
            return False, "No flight data available for the given constraints."

    if local.get("transportation") == "no flight":
        if len(flight_records) < 2:
            return False, "Impossible (no valid flights)."
        distance = None
        for record in flight_records:
            dist_val = record.get("Distance")
            if dist_val is not None:
                distance = dist_val
                break
        if distance is not None and distance > 800:
            return False, "Impossible (distance > 800 km)."

    room_type = local.get("room type")
    room_map = {
        "shared room": "Shared room",
        "private room": "Private room",
        "entire room": "Entire home/apt",
    }
    if room_type:
        if room_type == "not shared room":
            filtered = [h for h in hotel_records if h.get("room type") and h.get("room type") != "Shared room"]
        else:
            expected = room_map.get(room_type)
            filtered = [h for h in hotel_records if h.get("room type") == expected]
        if len(filtered) < days:
            return False, f"No hotel data available for room type = {room_type}"

    hrule = local.get("house rule")
    if hrule:
        mapping = {
            "smoking": "No smoking",
            "pets": "No pets",
            "parties": "No parties",
            "visitors": "No visitors",
            "children under 10": "No children under 10"
        }
        banned = mapping.get(hrule)
        if banned:
            filtered = [h for h in hotel_records if banned not in str(h.get("house_rules", ""))]
            if len(filtered) < days:
                return False, f"No hotel satisfies given house rule = {hrule}"

    cuisine_list = local.get("cuisine")
    if cuisine_list:
        filtered = [r for r in restaurant_records if any(c in str(r.get("Cuisines", "")) for c in cuisine_list)]
        if len(filtered) < days:
            return False, "Not enough restaurants satisfy cuisine constraints."

    if ppl:
        filtered = [h for h in hotel_records if h.get("maximum occupancy", 0) >= ppl]
        if len(filtered) < days:
            return False, f"No accommodation available for {ppl} people."

    return True, None

def get_total_cost(question, tested_data, reference_data):
    total_cost = 0
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        
        # Transportation cost
        if type(question['people_number']) != int:
            ppl_num = 1
        else:
            ppl_num = question['people_number']
        if unit['transportation'] and unit['transportation'] != '-':
            value = unit['transportation']
            if 'flight number' in value.lower():
                flight_num = value.split('Flight Number: ')[1].split(',')[0]
                flight_info = find_reference_info(reference_data, 'flight', flight_num, None)
                if flight_info:
                    total_cost += flight_info['Price'] * ppl_num
            
            elif 'self-driving' in value.lower() or 'taxi' in value.lower():
                org_city, dest_city = extract_from_to(value)
                if org_city and dest_city:
                    distance_info = find_reference_info(reference_data, 'distance', org_city, dest_city)
                    if distance_info:
                        if 'self-driving' in value.lower():
                            raw_cost = distance_info['self-driving']
                            # extract the number that comes after 'cost: '
                            cost = int(re.findall(r'cost: (\d+)', raw_cost)[0])
                            total_cost += cost * math.ceil(ppl_num / 5)
                        else:
                            raw_cost = distance_info['taxi']
                            cost = int(re.findall(r'cost: (\d+)', raw_cost)[0])
                            total_cost += cost * math.ceil(ppl_num / 4)

        # Meals cost
        for meal_type in ['breakfast', 'lunch', 'dinner']:
            if unit[meal_type] and unit[meal_type] != '-':
                name, city = get_valid_name_city(unit[meal_type])
                if city:  # If we now have a valid city (either original or current_city)
                    if city not in unit['current_city']:
                        if '(' in unit[meal_type] and ')' in unit[meal_type]:
                            temp_name = unit[meal_type].split('(')[0].strip()
                            if ',' in temp_name:
                                name = temp_name.split(',')[0].strip()
                                city = temp_name.split(',')[1].strip()
                
                    if name:  # As long as we have a name, try to find the cost
                        restaurant_info = find_reference_info(reference_data, 'restaurant', name, city)
                        if restaurant_info:
                            total_cost += restaurant_info['Average Cost'] * ppl_num
                        else:
                            avg_cost = get_avg_cost('restaurant', city, reference_data)
                            total_cost += avg_cost * ppl_num
                else:
                    # If we still don't have a city, calculate based on all cities average
                    all_cities_avg = get_avg_cost('restaurant', None, reference_data)
                    total_cost += all_cities_avg * ppl_num

        # Accommodation cost
        try:
            if unit['accommodation'] and unit['accommodation'] != '-':
                name, city = get_valid_name_city(unit['accommodation'])
                def calculate_total_cost(name, city):
                    acc_info = find_reference_info(reference_data, 'accommodation', name, city)
                    if acc_info:
                        return acc_info['price'] * math.ceil(ppl_num / acc_info['maximum occupancy'])
                    else:
                        avg_cost = get_avg_cost('accommodation', city, reference_data)
                        return avg_cost * ppl_num

                if name and city:
                    city_list = get_city_list_from_reference_data(reference_data)
                    if city not in city_list:
                        if len(unit['accommodation'].split(',')) > 1:
                            try:
                                name, city = unit['accommodation'].split(',')
                                city = city.strip()
                                if city not in city_list:
                                    name = unit['accommodation']
                                    city = None
                            except:
                                name = unit['accommodation']
                                city = None
                        else:
                            name = unit['accommodation']
                            city = None
                    
                    total_cost += calculate_total_cost(name, city)
                else:
                    total_cost += calculate_total_cost(unit['accommodation'], city=None)
        except:
            # import pdb; pdb.set_trace()
            print("No accommodation info provided")
            continue
    return total_cost

def is_valid_room_rule(question, tested_data, reference_data):
    if question['local_constraint']['house rule'] is None:
        return None, None
    
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        if unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            if name and city:
                acc_info = find_reference_info(reference_data, 'accommodation', name, city)
                if acc_info:
                    rule_map = {
                        'smoking': 'No smoking',
                        'parties': 'No parties',
                        'children under 10': 'No children under 10',
                        'visitors': 'No visitors',
                        'pets': 'No pets'
                    }
                    if question['local_constraint']['house rule'] in rule_map:
                        try:
                            # if rule_map[question['local_constraint']['house rule']] in acc_info['house_rules']:
                            #     return False, f"Accommodation {name} does not meet the house rule of {question['local_constraint']['house rule']}. The house rule of current accommodation is {acc_info['house_rules']}."
                            # else:
                            #     return True, None
                            if rule_map[question['local_constraint']['house rule']] in acc_info['house_rules']:
                                return False, f"Accommodation {name} does not meet the house rule of {question['local_constraint']['house rule']}."
                        except TypeError:
                            # type error
                            continue
                            
                        # if rule_map[question['local_constraint']['house rule']] in acc_info['house_rules']:
                        #     # return False, f"The house rule should be {question['local_constraint']['house rule']}."
                        #     return False, f"Accommodation {name} does not meet the house rule of {question['local_constraint']['house rule']}."
                    else:
                        return False, f"The accommodation {name} is not in the accommodation database."
    
    return True, None

def is_valid_cuisine(question, tested_data, reference_data):
    if not question['local_constraint']['cuisine']:
        return None, None
        
    cuisine_set = set()
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        for meal_type in ['breakfast', 'lunch', 'dinner']:
            if unit[meal_type] and unit[meal_type] != '-':
                name, city = get_valid_name_city(unit[meal_type])
                if name and city and city != question['org']:
                    restaurant_info = find_reference_info(reference_data, 'restaurant', name, city)
                    if restaurant_info:
                        cuisines = restaurant_info['Cuisines']
                        # cuisines = 'Bakery, Desserts, Fast Food'
                        cuisines = cuisines.split(', ')
                        cuisine_set.update(set(cuisines) & set(question['local_constraint']['cuisine']))
                    else:
                        return False, f"The restaurant {name} is not in the restaurant database."

    if len(cuisine_set) == len(question['local_constraint']['cuisine']):
        return True, None
    else:
        missing = set(question['local_constraint']['cuisine']) - cuisine_set
        return False, f"The cuisine {', '.join(missing)} is not satisfied."
    
def is_valid_cuisine_pref(question, tested_data, reference_data):
    """
    Validates if the cuisine preference is satisfied based on the tested data.
    
    Returns:
        tuple: (bool, str or None)
               - True if the cuisine preferences are satisfied.
               - False with an error message if any preference is not satisfied.
               - None, None if there are no cuisine preferences.
    """
    if not question['cuisine_pref']:
        return None, None  # No cuisine preference to validate.

    # Initialize a counter to track cuisine occurrences.
    # cuisine_count = {cuisine_type: 0 for cuisine_type, _ in question['cuisine_pref']}
    cuisine_type = question['cuisine_pref'][0]
    cuisine_count = {cuisine_type: 0}

    # Iterate through each day's data up to the specified number of days.
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        for meal_type in ['breakfast', 'lunch', 'dinner']:
            if unit[meal_type] and unit[meal_type] != '-':  # Check if the meal entry exists.
                name, city = get_valid_name_city(unit[meal_type])
                if name and city and city != question['org']:  # Ensure it's not in the origin city.
                    restaurant_info = find_reference_info(reference_data, 'restaurant', name, city)
                    if restaurant_info:
                        cuisines = restaurant_info['Cuisines']
                        if isinstance(cuisines, list):
                            cuisine_list = cuisines
                        else:
                            # If it's a string, split it into a list
                            cuisine_list = cuisines.split(', ')
                        if cuisine_type in cuisine_list:  # Check if the cuisine matches.
                            cuisine_count[cuisine_type] += 1

    # Check if the cuisine count meets the required number of occurrences.
    required_count = question['cuisine_pref'][1]
    if cuisine_count[cuisine_type] < required_count:
        missing_count = required_count - cuisine_count[cuisine_type]
        return False, f"The cuisine {cuisine_type} is missing {missing_count} occurrences."

    return True, None  # All preferences are satisfied.

def is_valid_ratings(question, tested_data, reference_data):
    if question['ratings'] is None:
        return None, None
    
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        for meal_type in ['breakfast', 'lunch', 'dinner']:
            if unit[meal_type] and unit[meal_type] != '-':
                name, city = get_valid_name_city(unit[meal_type])
                if name and city:
                    restaurant_info = find_reference_info(reference_data, 'restaurant', name, city)
                    if restaurant_info:
                        restaurant_rating = restaurant_info['Aggregate Rating']
                        if restaurant_rating < question['ratings']:
                            return False, f"{meal_type.capitalize()} place {name} does not meet the minimum ratings of {question['ratings']}"
                    else:
                        return False, f"The restaurant {name} is not in the restaurant database."
    
    return True, None

def is_valid_ratings_pref(question, tested_data, reference_data):
    """
    Checks if there are at least N times mentions of restaurants with a rating
    equal to or higher than the minimum rating in the given tested_data,
    skipping restaurants not found in the database.

    Args:
        question (list): A list where the first element is the minimum rating (min_rating)
                         and the second element is the number of times to visit (N).
        tested_data (list): Data to be tested. It is a list of dictionaries containing
                            meal types and restaurant information.
        reference_data (list): Reference data containing restaurant details and ratings.

    Returns:
        (bool, str): A tuple where the first value indicates whether the condition is satisfied,
                     and the second value provides a message if the condition fails.
    """
    min_rating, required_visits = question['rating_pref']
    valid_visits = 0

    for unit in tested_data:
        for meal_type in ['breakfast', 'lunch', 'dinner']:
            if unit.get(meal_type) and unit[meal_type] != '-':
                name, city = get_valid_name_city(unit[meal_type])
                if name and city:
                    restaurant_info = find_reference_info(reference_data, 'restaurant', name, city)
                    if restaurant_info:
                        restaurant_rating = restaurant_info['Aggregate Rating']
                        if restaurant_rating >= min_rating:
                            valid_visits += 1
                            if valid_visits >= required_visits:
                                return True, None
                # If the restaurant is not found, skip and continue to the next one.
    if valid_visits < required_visits:
        return False, f"There are only {valid_visits} valid visits with ratings >= {min_rating}, but {required_visits} are required."

    return True, None

def is_valid_transportation(question, tested_data, reference_data):
    if question['local_constraint']['transportation'] is None:
        return None, None
    
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        if unit['transportation'] and unit['transportation'] != '-':
            value = unit['transportation']
            if question['local_constraint']['transportation'] == 'no flight' and 'Flight' in value:
                return False, f"The transportation should not be {question['local_constraint']['transportation']}."
            elif question['local_constraint']['transportation'] == 'no self-driving' and 'Self-driving' in value:
                return False, f"The transportation should not be {question['local_constraint']['transportation']}."
    
    return True, None

def is_valid_room_type(question, tested_data, reference_data):
    if question['local_constraint']['room type'] is None:
        return None, None
    
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        if unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            if name and city:
                acc_info = find_reference_info(reference_data, 'accommodation', name, city)
                if acc_info:
                    room_type = acc_info['room type']
                    
                    room_type_map = {
                        'not shared room': lambda x: x != 'Shared room',
                        'shared room': lambda x: x == 'Shared room',
                        'private room': lambda x: x == 'Private room',
                        'entire room': lambda x: x == 'Entire home/apt'
                    }
                    
                    if question['local_constraint']['room type'] in room_type_map:
                        if not room_type_map[question['local_constraint']['room type']](room_type):
                            return False, f"The room type should be {question['local_constraint']['room type']}."
                
                    else:
                        return False, f"The accommodation {name} is not in the accommodation database."
    
    return True, None


def can_accommodate_people(question, tested_data, reference_data):
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        if unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            if name and city:
                acc_info = find_reference_info(reference_data, 'accommodation', name, city)
                if acc_info:
                    ppl_num = question['people_number']
                    if ppl_num > acc_info['maximum occupancy']:
                        return False, f"The accommodation {name} cannot accommodate {ppl_num} people."
                else:
                    return False, f"The accommodation {name} is not in the accommodation database."
    
    return True, None


def evaluation(query_data, tested_data, reference_data):
    if type(query_data) == str:
        query_data = eval(query_data)
    if type(query_data['local_constraint']) == str:
        query_data['local_constraint'] = eval(query_data['local_constraint'])

    # [핵심 수정] evaluate.py에서 포장되어 온 데이터(days, budget, people_number 등)를
    # 계산 전에 미리 전부 뜯어서 숫자로 복구합니다. (위치 변경됨: 맨 위로!)
    for k, v in query_data.items():
        if isinstance(v, dict) and v.get('is_wrapped_int'):
            query_data[k] = v['value']

    ### 정현: total_cost와 symbolic budget 미리 계산 ###
    total_cost = get_total_cost(query_data, tested_data, reference_data)
    symbolic_budget = estimate_symbolic_budget(query_data, reference_data)

    budget_value = query_data.get('budget')  # 위에서 이미 포장을 뜯었으므로 그냥 가져오면 됩니다.

    budget_is_correct = None
    budget_error = None
    if budget_value is not None:
        budget_is_correct = total_cost <= budget_value
        if not budget_is_correct:
            budget_error = f"The total cost ({total_cost}) exceeds the budget of {budget_value}."

    ### 정현: TravelPlanner feasibility rule 검사 ###
    rule_is_correct, rule_error = travelplanner_feasibility_check(query_data, reference_data)
    return_info = {
        'cuisine': {
            'query_value': query_data['local_constraint']['cuisine'],
            'is_correct': is_valid_cuisine(query_data, tested_data, reference_data)[0],
            'error_message': is_valid_cuisine(query_data, tested_data, reference_data)[1]
        },
        'room_rule': {
            'query_value': query_data['local_constraint']['house rule'],
            'is_correct': is_valid_room_rule(query_data, tested_data, reference_data)[0],
            'error_message': is_valid_room_rule(query_data, tested_data, reference_data)[1]
        },
        'transportation': {
            'query_value': query_data['local_constraint']['transportation'],
            'is_correct': is_valid_transportation(query_data, tested_data, reference_data)[0],
            'error_message': is_valid_transportation(query_data, tested_data, reference_data)[1]
        },
        'room_type': {
            'query_value': query_data['local_constraint']['room type'],
            'is_correct': is_valid_room_type(query_data, tested_data, reference_data)[0],
            'error_message': is_valid_room_type(query_data, tested_data, reference_data)[1]
        },
        ### 정현: TravelPlanner budget rule 결과를 budget 항목에 포함 ###
        'budget': {
            'query_value': query_data['budget'],
            'is_correct': budget_is_correct,
            'error_message': budget_error,
            'symbolic_budget': symbolic_budget
        },
        'ratings': {
            'query_value': query_data.get('ratings', None),
            'is_correct': is_valid_ratings(query_data, tested_data, reference_data)[0] if query_data.get(
                'ratings') is not None else None,
            'error_message': is_valid_ratings(query_data, tested_data, reference_data)[1] if query_data.get(
                'ratings') is not None else None
        },
        'people_number': {
            'query_value': query_data['people_number'],
            'is_correct': total_cost <= budget_value if budget_value is not None and query_data[
                'people_number'] is not None else None,
            'error_message': (
                f"The total cost ({total_cost}) exceeds the budget of {budget_value}."
                if budget_value is not None and query_data['people_number'] is not None and total_cost > budget_value
                else None
            )
        },
        'rating_pref': {
            'query_value': query_data.get('rating_pref', None),
            'is_correct': is_valid_ratings_pref(query_data, tested_data, reference_data)[0] if query_data.get(
                'rating_pref') is not None else None,
            'error_message': is_valid_ratings_pref(query_data, tested_data, reference_data)[1] if query_data.get(
                'rating_pref') is not None else None
        },
        'cuisine_pref': {
            'query_value': query_data.get('cuisine_pref', None),
            'is_correct': is_valid_cuisine_pref(query_data, tested_data, reference_data)[0] if query_data.get(
                'cuisine_pref') is not None else None,
            'error_message': is_valid_cuisine_pref(query_data, tested_data, reference_data)[1] if query_data.get(
                'cuisine_pref') is not None else None
        },

        ### 정현: TravelPlanner rule 엔진 결과 ###
        'symbolic_rule': {
            'query_value': None,
            'is_correct': rule_is_correct,
            'error_message': rule_error,
            'calculated_value': total_cost,  # 주영님 브랜치 값 그대로 보존
        },
    }
    return return_info

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--evaluation_file_path", type=str, default="./")
    # parser.add_argument("--reference_data_path", type=str, required=True)
    # args = parser.parse_args()

    # Load test data and reference data
    # test_data = load_line_json_data(args.evaluation_file_path)
    # reference_data = load_line_json_data(args.reference_data_path)
    reference_data = load_line_json_data("/home/juhyun/FlexibleReasoningBench/implement/agents/evaluation/database/train_ref_info.jsonl")
    # updated_constraints = {'people_number': 1, 'room rule': None, 'cuisine': None, 'room type': None, 'transportation': None, 'budget': 1700, 'ratings': None}
    updated_constraints = {
            "org": "St. Petersburg",
            "dest": "Rockford",
            "days": 3,
            "visiting_city_number": 1,
            "date": "['2022-03-16', '2022-03-17', '2022-03-18']",
            "local_constraint": {
                "house rule": None,
                "cuisine": None,
                "room type": None,
                "transportation": None
            },
            "budget": 1700,
            "people_number": 1,
            "ratings": 4.0
        }
    response_data = [{'days': 1, 'current_city': 'from St. Petersburg to Rockford', 'transportation': 'Flight Number: F3573659, from St. Petersburg to Rockford, Departure Time: 15:40, Arrival Time: 17:04', 'breakfast': '-', 'attraction': '-', 'lunch': '-', 'dinner': 'Coco Bambu, Rockford', 'accommodation': 'Pure luxury one bdrm + sofa bed on Central Park, Rockford'}, {'days': 2, 'current_city': 'Rockford', 'transportation': '-', 'breakfast': 'Dial A Cake, Rockford', 'attraction': 'Burpee Museum of Natural History, Rockford;Midway Village Museum, Rockford;Discovery Center Museum, Rockford;', 'lunch': 'Flying Mango, Rockford', 'dinner': 'Cafe Southall, Rockford', 'accommodation': 'Pure luxury one bdrm + sofa bed on Central Park, Rockford'}, {'days': 3, 'current_city': 'from Rockford to St. Petersburg', 'transportation': 'Flight Number: F3573120, from Rockford to St. Petersburg, Departure Time: 19:00, Arrival Time: 22:43', 'breakfast': 'Subway, Rockford', 'attraction': 'Klehm Arboretum & Botanic Garden, Rockford;Sinnissippi Park, Rockford;', 'lunch': 'Gajalee Sea Food, Rockford', 'dinner': 'Nutri Punch, Rockford', 'accommodation': '-'}, {}, {}, {}, {}]
    tested_plan = {'query': updated_constraints, 'plan': response_data}
    eval_results = evaluation(tested_plan['query'], tested_plan['plan'], reference_data[0])
    
    # for idx, tested_plan in enumerate(tqdm(test_data)):
    #     if tested_plan['plan']:
    #         # eval_results = evaluation(tested_plan['query'], tested_plan['plan'], reference_data[idx])
    #         eval_results = evaluation(tested_plan['query'], tested_plan['plan'], reference_data[0])
    #         is_valid = all(result[0] for result in eval_results.values() if result[0] is not None)
            
    #         print(f"Plan {idx} is {'valid' if is_valid else 'invalid'}")
    #         if not is_valid:
    #             print("Validation failures:")
    #             for key, (valid, message) in eval_results.items():
    #                 if valid is False:
    #                     print(f"- {key}: {message}")
