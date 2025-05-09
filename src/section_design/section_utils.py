import math
from copy import copy
from dataclasses import dataclass, field
from itertools import product

from src.utils import find_first_greater_sorted

# Unit conversion constants
mm_m = 1e-3  # mm to m
mmq_mq = 1e-6  # mm^2 to m^2


def reinforcement_area(count: int, diameter: float) -> float:
    """
    Computes the total area of steel reinforcement based on the number of bars and their diameter.

    :param count: Number of steel bars.
    :param diameter: Diameter of the steel bars (in mm).
    :return: Total area of the steel reinforcement (in mm^2).
    """
    return count * math.pi * (diameter) ** 2 / 4


@dataclass
class ReinforcementCombination:
    section_width: float
    min_diameter: int = field(default=6)
    max_diameter: int = field(default=30)
    min_count: int = field(default=2)
    max_count: int = field(default=12)
    available_combinations: dict[tuple[int, int], float] = field(init=False)
    sorted_combinations: list[tuple[int, int]] = field(init=False)

    def __post_init__(self):
        self.available_combinations = self._generate_reinforcements(
            min_bars=self.min_count,
            max_bars=self.max_count,
            min_d=self.min_diameter,
            max_d=self.max_diameter
        )
        # trust me this works
        self.sorted_combinations = sorted(
            self.available_combinations,
            key=lambda x: self.available_combinations[x]
        )

    @staticmethod
    def _generate_reinforcements(
        min_bars: int,
        max_bars: int,
        min_d: int,
        max_d: int
    ) -> dict[tuple[int, int], float]:
        """
        Generates a dictionary of reinforcement combinations based on the
        number of bars and their diameters. The keys in the dictionary are tuples
        representing combinations of (bars, diameter), and the values are floating-point
        numbers representing reinforcement area.

        :param min_bars: Minimum number of reinforcement bars, defaults to 2
        :param max_bars: Maximum number of reinforcement bars, defaults to 12
        :param min_d: Minimum diameter of the bars in mm, defaults to 6
        :param max_d: Maximum diameter of the bars in mm, defaults to 30
        :return: A dictionary where keys are tuples of (bars, diameter), and
                 values are the calculated reinforcement area.
        """
        n_bars = range(min_bars, max_bars + 1)       # Every integer between min and max
        diameters = range(min_d, max_d + 1, 2)  # Only even integer between min and max

        # Instantiate return dict
        reinf_area = {}

        for num, diameter in product(n_bars, diameters):
            reinf_area[(num, diameter)] = reinforcement_area(num, diameter) * mmq_mq

        return reinf_area

    @staticmethod
    def minimum_section_width(n_bars: int, diameter: float, cop: float = .03) -> float:
        return 2 * cop + n_bars * diameter + (n_bars - 1) * max(0.02, diameter)

    def find_combination(self, min_area: float, cover: float = .03) -> tuple[int, int] | None:
        """
        Finds the smallest reinforcement combination that meets the required area
        and fits within the section width.

        Iterates through the sorted available reinforcement combinations in
        ascending order of area and returns the first one that meets both the
        minimum required reinforcement area and fits within the section width
        when accounting for concrete cover and spacing.

        :param min_area: Minimum required reinforcement area in square meters.
        :param cover: Concrete cover in meters to be considered on both sides,
                      defaults to 0.03 m (30 mm).
        :return: A tuple of (number of bars, diameter in mm) if a valid combination
                 is found; otherwise, returns None.
        """
        sorted_combinations = copy(self.sorted_combinations)
        combination = None

        # While there possible solutions
        while sorted_combinations:
            index = find_first_greater_sorted(
                [self.available_combinations[key] for key in sorted_combinations],
                pivot=min_area
            )

            if index is None:
                break

            combination = sorted_combinations[index]

            n_bars, diameter_mm = combination

            min_section_width = self.minimum_section_width(n_bars, diameter_mm * mm_m, cover)

            if self.section_width >= min_section_width:
                return combination

            sorted_combinations.pop(index)

        # If no combination is found, returns none
        return None
