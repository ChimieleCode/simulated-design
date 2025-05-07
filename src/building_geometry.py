from dataclasses import dataclass
from typing import List


@dataclass
class BuildingGeometry:
    """
    Represents the geometric characteristics of a building for structural load calculations.
    """
    floors: int
    span_main: float
    span_cross: float
    floor_height: float
    n_main_spans: int
    n_cross_spans: int
    is_fully_braced: bool = False # If false then only perimeter frames are alogn cross direction

    @property
    def bay_area(self) -> float:
        """
        Area of a single structural bay.
        """
        return self.span_main * self.span_cross

    @property
    def area(self) -> float:
        """
        Total floor area of the building (for one floor).
        """
        return self.n_main_spans * self.n_cross_spans * self.bay_area

    @property
    def column_count(self) -> int:
        """
        Total number of vertical structural columns at each floor.
        """
        return (self.n_main_spans + 1) * (self.n_cross_spans + 1)

    @property
    def perimeter(self) -> float:
        """
        Total perimeter length of the building (for infill walls).
        """
        return 2 * (self.n_main_spans * self.span_main + self.n_cross_spans * self.span_cross)

    @property
    def comulative_floor_height(self) -> List[float]:
        """
        Returns the cumulative height from the base to each floor (for seismic load distribution).
        """
        return [self.floor_height * (i + 1) for i in range(self.floors)]

    @property
    def n_main_frames(self) -> int:
        """
        Number of frames in the main direction.
        """
        return self.n_main_spans + 1

    @property
    def n_cross_frames(self) -> int:
        """
        Number of frames in the cross direction.
        """
        if self.is_fully_braced:
            return self.n_cross_spans + 1
        return 2

    def get_total_beam_length(self, include_main: bool = True, include_cross: bool = True) -> float:
        """
        Computes total beam length in the floor plan for specified directions.

        :param include_main: Whether to include beams in the main direction.
        :param include_cross: Whether to include beams in the cross direction.
        :return: Total beam length.
        """
        total_length = 0.0
        if include_main:
            total_length += self.n_main_spans * (self.n_cross_spans + 1) * self.span_main
        if include_cross:
            total_length += self.n_cross_spans * (self.n_main_spans + 1) * self.span_cross
        return total_length
