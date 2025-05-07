import math
from dataclasses import dataclass

from src.building_geometry import BuildingGeometry


@dataclass
class BuildingBox:
    h: float
    b: float
    floors: int


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
        n_main_spans = self._main_spans(bbox.b)
        n_cross_spans = self._cross_spans(bbox.h)

        # Creates and returns the BuildingGeometry object
        return BuildingGeometry(
            floors=bbox.floors,
            span_main=bbox.b/n_main_spans,
            span_cross=bbox.h/n_cross_spans,
            floor_height=self.floor_height,
            n_main_spans=n_main_spans,
            n_cross_spans=n_cross_spans,
            is_fully_braced=is_fully_braced
        )
