import bisect
import math
from collections.abc import Sequence


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
