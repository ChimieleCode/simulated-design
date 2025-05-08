import math
from dataclasses import dataclass


@dataclass
class BuildingBox:
    length: float
    width: float
    height: float
    floors: int


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
    is_fully_braced: bool = False # If false then only perimeter frames are along cross direction

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
    def comulative_floor_height(self) -> list[float]:
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


@dataclass
class BuildingGeometryFactory:
    main_target_span: float     # Target span length in the main direction, in meters, 4.
    cross_target_span: float    # Target span length in the cross direction, in meters, 4.5
    floor_height: float

    def _main_spans(self, length: float) -> int:
        """
        Computes the number of spans in the main direction.

        :param length: Total length of the building in the main direction.
        :return: Number of spans in the main direction.
        """
        # Computes n_spans
        n_spans = round(length/self.main_target_span)
        l_span = length/n_spans
        # Handle edge cases
        if l_span > 5:
            n_spans +=1
        return n_spans

    def _cross_spans(self, length: float) -> int:
        """
        Computes the number of spans in the cross direction.

        :param length: Total length of the building in the cross direction.
        :return: Number of spans in the cross direction.
        """
        # Computes number of cross spans
        return math.ceil(length/self.cross_target_span)

    def create(self, bbox: BuildingBox, is_fully_braced: bool = True) -> BuildingGeometry:
        """
        Creates a BuildingGeometry object based on the provided bounding box dimensions.

        :param bbox: Bounding box dimensions.
        :param is_fully_braced: Indicates if the building is fully braced.
        :return: BuildingGeometry object.
        """
        # Computes n_spans
        n_main_spans = self._main_spans(bbox.length)
        n_cross_spans = self._cross_spans(bbox.width)

        # Creates and returns the BuildingGeometry object
        return BuildingGeometry(
            floors=bbox.floors,
            span_main=round(bbox.length/n_main_spans, 2),
            span_cross=round(bbox.width/n_cross_spans, 2),
            floor_height=self.floor_height,
            n_main_spans=n_main_spans,
            n_cross_spans=n_cross_spans,
            is_fully_braced=is_fully_braced
        )
