#!/usr/bin/env python3
"""
Test script to verify Priority Reweighting functionality
"""
from agents.constraints_generator import (
    parse_priority_ranks,
    calculate_priority_weights
)

# Test case 1: Constraints with rank information
print("=" * 60)
print("Test Case 1: Constraints with rank information")
print("=" * 60)

test_constraints_1 = {
    'budget': {'value': 2000, 'rank': 1},
    'cuisine_pref': {'value': ('Chinese', 2), 'rank': 2},
    'rating_pref': {'value': (4.0, 3), 'rank': 3}
}

priority_ranks = parse_priority_ranks(test_constraints_1)
print(f"Parsed priority ranks: {priority_ranks}")

priority_weights = calculate_priority_weights(priority_ranks)
print(f"Calculated priority weights: {priority_weights}")

# Test case 2: Constraints without rank information
print("\n" + "=" * 60)
print("Test Case 2: Constraints without rank information")
print("=" * 60)

test_constraints_2 = {
    'budget': 2000,
    'cuisine_pref': ('Chinese', 2),
    'rating_pref': (4.0, 3)
}

priority_ranks_2 = parse_priority_ranks(test_constraints_2)
print(f"Parsed priority ranks: {priority_ranks_2}")

priority_weights_2 = calculate_priority_weights(priority_ranks_2)
print(f"Calculated priority weights: {priority_weights_2}")

# Test case 3: Mixed format
print("\n" + "=" * 60)
print("Test Case 3: Mixed format (dict with priority field)")
print("=" * 60)

test_constraints_3 = {
    'budget': {'value': 2000, 'priority': 1},
    'cuisine_pref': {'value': ('Chinese', 2), 'priority': 2},
}

priority_ranks_3 = parse_priority_ranks(test_constraints_3)
print(f"Parsed priority ranks: {priority_ranks_3}")

priority_weights_3 = calculate_priority_weights(priority_ranks_3)
print(f"Calculated priority weights: {priority_weights_3}")

# Test build_priority_block
print("\n" + "=" * 60)
print("Test Case 4: Priority block generation")
print("=" * 60)

from agents.prompts_add_memory_eval import build_priority_block

priority_block = build_priority_block(priority_weights)
print("Generated Priority Block:")
print(priority_block)

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)

