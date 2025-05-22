import bisect
import json
import math
from collections.abc import Sequence
from pathlib import Path


def circle_area(radius: float) -> float:
    """
    Computes the area of a circle given its radius.

    :param radius: Radius of the circle (in meters).
    :type radius: float
    :return: Area of the circle (in square meters).
    :rtype: float
    """
    return math.pi * radius ** 2


def find_first_greater_sorted(sequence: Sequence[float], pivot: float) -> int | None:
    # Find the index where `pivot` would be inserted to maintain sorted order
    index = bisect.bisect_right(sequence, pivot)

    # Check if this index is within bounds and corresponds to a value greater than pivot
    if index < len(sequence):
        return index
    return None


def round_to_nearest(number: float, tol: float) -> float:
    """
    Rounds a number to the nearest multiple of a specified tolerance.

    :param number: The number to be rounded.
    :param tol: The tolerance to round to.
    :return: The rounded number.
    """
    return round(number / tol) * tol


def import_from_json(filepath: Path) -> dict:
    """
    Imports a .json file and converts it into a dictionary
    """
    with open(filepath, 'r') as jsonfile:
        return json.loads(jsonfile.read())


def export_to_json(filepath: Path, data: dict) -> None:
    """
    Exports a given dict into a json file
    """
    with open(filepath, 'w') as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False, indent=4)
