"""Small independent enumeration oracle; never used by the library."""

import itertools


def enumerate_pmf(probs):
    if len(probs) > 12:
        raise ValueError("enumeration is restricted to n <= 12")
    result = [0.0] * (len(probs) + 1)
    for outcome in itertools.product((0, 1), repeat=len(probs)):
        probability = 1.0
        for p, success in zip(probs, outcome):
            probability *= p if success else 1 - p
        result[sum(outcome)] += probability
    return result


def enumerate_proportion(probs, mask, beta, side):
    numerator, denominator = float(beta).as_integer_ratio()
    result = 0.0
    for outcome in itertools.product((0, 1), repeat=len(probs)):
        total = sum(outcome)
        selected = sum(y for y, interesting in zip(outcome, mask) if interesting)
        # Cross multiplication implements strict comparison of the exact float.
        left = selected * denominator
        right = total * numerator
        if total == 0 or not (left < right if side == "left" else left > right):
            continue
        probability = 1.0
        for p, success in zip(probs, outcome):
            probability *= p if success else 1 - p
        result += probability
    return result
