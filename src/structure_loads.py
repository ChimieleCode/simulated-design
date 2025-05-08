from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.building_geometry import BuildingGeometry
from src.portal_frame_method import (RegularSpanFrameSollicitations,
                                     get_portal_frame_method_sollicitations)
from src.seismic_codes import (BuildingCode, SeismicCat, SeismicForces,
                               SeismicWeight)
from src.sollicitations import MemberSollicitation


@dataclass
class StructureLoads:
    """
    Represents various structural loads applied to a building.

    :param floaring_load: The permanent load (G) applied to each interstorey per unit area (kN/m²).
    :param infill_load: The permanent load (G) for the infill panels per unit length (kN/m).
    :param beam_load: The self-weight of the beams as a permanent load (G) per unit length (kN/m).
    :param overload: The live load (Q) applied to each interstorey per unit area (kN/m²).
    :param roof_overload: The live load (Q) applied to the roof per unit area (kN/m²).
    :param column_load: The permanent load (G) applied by columns per vertical meter (kN/m).
    """
    floaring_load: float        # G for each inter-storey [kN/m²]
    infill_load: float          # G for the infill panels [kN/m]
    beam_load: float            # G for the beam self loads [kN/m]
    overload: float             # Q for the interstorey [kN/m²]
    roof_overload: float        # Q for the roof [kN/m²]
    column_load: float          # G for the column per vertical meter [kN/m]


def compute_total_load(
        area_loads: Sequence[tuple[float, float]] | None = None,
        line_loads: Sequence[tuple[float, float]] | None = None,
        punctual_loads: Sequence[tuple[float, float]] | None = None
    ) -> float:
    """
    Computes the total structural load by summing up loads from various sources,
    including area loads, line loads, and punctual (point) loads.

    :param area_loads: A list of tuples, where each tuple contains an area in square meters (m²) and
                       the corresponding load in kilonewtons per square meter (kN/m²).
    :param line_loads: A list of tuples, where each tuple contains a length in meters (m) and
                       the corresponding load in kilonewtons per meter (kN/m).
    :param punctual_loads: A list of tuples, where each tuple contains a count (number of points) and
                           the corresponding load per point in kilonewtons (kN).
    :return: The total load in kilonewtons (kN) summed from all provided load types.
    """
    total_load = 0

    # Sum area loads if provided
    if area_loads is not None:
        total_load += sum(area * load for area, load in area_loads)

    # Sum line loads if provided
    if line_loads is not None:
        total_load += sum(length * load for length, load in line_loads)

    # Sum punctual loads if provided
    if punctual_loads is not None:
        total_load += sum(count * load for count, load in punctual_loads)

    return total_load


# --------------------------------------------------------------------------------------------------------------------------
# Seismic Load Design
# --------------------------------------------------------------------------------------------------------------------------

# Seismic Load
@dataclass
class SeismicLoadDesign(StructureLoads):
    seismic_cat: SeismicCat
    building_code: BuildingCode

    def compute_frame_sollicitations(self, building_geometry: BuildingGeometry) -> tuple[RegularSpanFrameSollicitations, RegularSpanFrameSollicitations]:
        """
        Compute the seismic frame solicitations for both the main and cross directions
        of the building.
        """
        permanent_weight_floor, overloads_floor = self._compute_floor_weights(building_geometry)
        permanent_weight_roof, overloads_roof = self._compute_roof_weights(building_geometry)

        seismic_weight_floor, seismic_weight_roof = self._compute_seismic_weights(
            permanent_weight_floor, overloads_floor, permanent_weight_roof, overloads_roof, building_geometry
        )

        seismic_forces = self._compute_seismic_forces(seismic_weight_floor, seismic_weight_roof, building_geometry)

        main_frame_forces, cross_frame_forces = self._compute_frame_forces(seismic_forces, building_geometry)

        seismic_stresses_main = self._compute_seismic_stresses(main_frame_forces, building_geometry, is_main=True)
        seismic_stresses_cross = self._compute_seismic_stresses(cross_frame_forces, building_geometry, is_main=False)

        return seismic_stresses_main, seismic_stresses_cross

    def _compute_floor_weights(self, building_geometry: BuildingGeometry) -> tuple[float, float]:
        """Computes the permanent and overload weights for the floor."""
        floor_height = building_geometry.floor_height
        permanent_weight_floor = compute_total_load(
            area_loads=[(building_geometry.area, self.floaring_load)],
            line_loads=[
                (building_geometry.perimeter, self.infill_load),
                (building_geometry.get_total_beam_length(True, True), self.beam_load)
            ],
            punctual_loads=[(building_geometry.column_count, self.column_load * floor_height)]
        )

        overloads_floor = compute_total_load(
            area_loads=[(building_geometry.area, self.overload)]
        )
        return permanent_weight_floor, overloads_floor

    def _compute_roof_weights(self, building_geometry: BuildingGeometry) -> tuple[float, float]:
        """Computes the permanent and overload weights for the roof."""
        floor_height = building_geometry.floor_height
        permanent_weight_roof = compute_total_load(
            area_loads=[(building_geometry.area, self.floaring_load)],
            line_loads=[
                (building_geometry.perimeter, 0.5 * self.infill_load),
                (building_geometry.get_total_beam_length(True, True), self.beam_load)
            ],
            punctual_loads=[(building_geometry.column_count, 0.5 * self.column_load * floor_height)]
        )

        overloads_roof = compute_total_load(
            area_loads=[(building_geometry.area, self.roof_overload)]
        )
        return permanent_weight_roof, overloads_roof

    def _compute_seismic_weights(self, permanent_weight_floor, overloads_floor, permanent_weight_roof, overloads_roof, building_geometry: BuildingGeometry) -> tuple[float, float]:
        """Computes the seismic weights for both the floors and roof."""
        seismic_forces_factory = SeismicWeight()
        seismic_weight_floor = seismic_forces_factory.compute_weight(
            building_code=self.building_code,
            G=permanent_weight_floor,
            Q=overloads_floor,
            seismic_cat=self.seismic_cat
        )

        seismic_weight_roof = seismic_forces_factory.compute_weight(
            building_code=self.building_code,
            G=permanent_weight_roof,
            Q=overloads_roof,
            seismic_cat=self.seismic_cat
        )

        return seismic_weight_floor, seismic_weight_roof

    def _compute_seismic_forces(self, seismic_weight_floor, seismic_weight_roof, building_geometry: BuildingGeometry) -> list[float]:
        """Computes the seismic forces based on seismic weight distribution."""
        seismic_weights = [seismic_weight_floor] * (building_geometry.floors - 1) + [seismic_weight_roof]
        seismic_forces = SeismicForces().compute_forces(
            building_code=self.building_code,
            weights=seismic_weights,
            seismic_cat=self.seismic_cat,
            floor_heights=building_geometry.comulative_floor_height
        )
        return seismic_forces

    def _compute_frame_forces(self, seismic_forces: list[float], building_geometry: BuildingGeometry) -> tuple[list[float], list[float]]:
        """Computes the forces for both the main and cross frames."""
        main_frame_forces = [force / (building_geometry.n_cross_spans + 1) for force in seismic_forces]
        cross_frame_forces = [force / (building_geometry.n_main_spans + 1) for force in seismic_forces]
        return main_frame_forces, cross_frame_forces

    def _compute_seismic_stresses(self, frame_forces: list[float], building_geometry: BuildingGeometry, is_main: bool) -> RegularSpanFrameSollicitations:
        """Computes the seismic stresses using the portal frame method."""
        span_length = building_geometry.span_main if is_main else building_geometry.span_cross
        column_count = building_geometry.n_main_spans + 1 if is_main else building_geometry.n_cross_spans + 1

        return get_portal_frame_method_sollicitations(
            heights=[building_geometry.floor_height] * building_geometry.floors,
            forces=frame_forces,
            span_length=span_length,
            column_count=column_count
        )


# --------------------------------------------------------------------------------------------------------------------------
# Gravity Load Design
# --------------------------------------------------------------------------------------------------------------------------

def compute_beam_moment_end(load: float, length: float) -> float:
    """
    Compute the bending moment at the end of a simply supported beam.

    :param load: The uniform load applied to the beam (per unit length).
    :param length: The length of the beam.
    :return: The moment at the beam's end.
    """
    return load * length**2 / 12


def compute_beam_shear_end(load: float, length: float) -> float:
    """
    Compute the shear force at the end of a simply supported beam.

    :param load: The uniform load applied to the beam (per unit length).
    :param length: The length of the beam.
    :return: The shear force at the beam's end.
    """
    return load * length / 2


@dataclass
class BeamSollicitations:
    """
    Stores the solicitations (forces and moments) for different beam types.
    """
    border_beam_main: MemberSollicitation
    border_beam_cross: MemberSollicitation
    internal_beam_main: MemberSollicitation
    internal_beam_cross: MemberSollicitation


@dataclass
class ColumnSollicitations:
    """
    Stores the solicitations (forces and moments) for different column types.
    """
    central_column: MemberSollicitation
    border_column_main: MemberSollicitation
    border_column_cross: MemberSollicitation
    corner_column: MemberSollicitation


@dataclass
class BuildingSollicitations:
    """
    Stores the solicitations for beams and columns across different floors of a building.
    """
    beams_sollicitations: Sequence[BeamSollicitations]
    columns_sollicitations: Sequence[ColumnSollicitations]

    def get_column_sollicitations(self, floor: int) -> ColumnSollicitations:
        """
        Retrieve the column solicitations for a specific floor.

        :param floor: The floor index for which to retrieve the solicitations.
        :return: The column solicitations for the specified floor.
        """
        return self.columns_sollicitations[floor]

    def get_beam_sollicitations(self, floor: int) -> BeamSollicitations:
        """
        Retrieve the beam solicitations for a specific floor.

        :param floor: The floor index for which to retrieve the solicitations.
        :return: The beam solicitations for the specified floor.
        """
        return self.beams_sollicitations[floor]


@dataclass
class GravityLoadDesignFullSpan(StructureLoads):
    """
    Class to compute and store the design loads (moments, shears, axial forces) for beams and columns
    due to gravity loads over the full span of a building.
    """

    def compute_beam_moments(self, building_geometry: BuildingGeometry) -> list[BeamSollicitations]:
        """
        Compute the moments and shears for beams in the building due to gravity loads.

        :param building_geometry: Object that holds the building's geometrical information.
        :return: A list of beam solicitations for each floor of the building.
        """

        floors = building_geometry.floors

        # Calculate the total area load for floor and roof
        floor_area_load = self.floaring_load + self.overload
        roof_area_load = self.floaring_load + self.roof_overload

        # Load from beams and infill walls
        beam_load = self.beam_load
        infill_load = self.infill_load

        # Lengths of beams in the main and cross directions
        beam_length_main = building_geometry.span_main
        beam_length_cross = building_geometry.span_cross

        # Influence areas for different beam positions
        beams_area = [.5 * beam_length_cross, 0, beam_length_cross, 0]
        beams_infill = [1, 1, 0, 0]
        beams_length = [beam_length_main, beam_length_cross, beam_length_main, beam_length_cross]

        # Compute distributed loads on beams for floor and roof
        beams_floor_load = [
            floor_area_load * area + infill_load * infill + beam_load
            for area, infill in zip(beams_area, beams_infill)
        ]
        beams_roof_load = [
            roof_area_load * area + beam_load
            for area in beams_area
        ]

        # Compute moments and shears for floor and roof
        beams_floor_moment = [
            compute_beam_moment_end(load, length)
            for load, length in zip(beams_floor_load, beams_length)
        ]
        beams_roof_moment = [
            compute_beam_moment_end(load, length)
            for load, length in zip(beams_roof_load, beams_length)
        ]

        beams_floor_shear = [
            compute_beam_shear_end(load, length)
            for load, length in zip(beams_floor_load, beams_length)
        ]
        beams_roof_shear = [
            compute_beam_shear_end(load, length)
            for load, length in zip(beams_roof_load, beams_length)
        ]

        # Assemble beam solicitations
        beam_sollicitations_floor = [
            MemberSollicitation(M=moment, V=shear)
            for moment, shear in zip(beams_floor_moment, beams_floor_shear)
        ]
        beam_sollicitations_roof = [
            MemberSollicitation(M=moment, V=shear)
            for moment, shear in zip(beams_roof_moment, beams_roof_shear)
        ]

        # Collect results for each floor and the roof
        beams_sollicitation = [beam_sollicitations_floor] * (floors - 1) + [beam_sollicitations_roof]

        return [BeamSollicitations(*beams) for beams in beams_sollicitation]


    def compute_column_axials(self, building_geometry: BuildingGeometry) -> list[ColumnSollicitations]:
        """
        Compute the axial forces for columns in the building due to gravity loads.

        :param building_geometry: Object that holds the building's geometrical information.
        :return: A list of column solicitations for each floor of the building.
        """

        floors = building_geometry.floors

        # Total area load on floors and roof
        floor_area_load = self.floaring_load + self.overload
        roof_area_load = self.floaring_load + self.roof_overload

        # Axial load from columns
        column_load = self.column_load * building_geometry.floor_height

        # Beam lengths
        beam_length_main = building_geometry.span_main
        beam_length_cross = building_geometry.span_cross

        # Influence area of different column types
        area = beam_length_main * beam_length_cross
        columns_area = [area, .5 * area, .5 * area, .25 * area]

        # Infill and beam lengths for columns
        columns_infill_length = [0, beam_length_main, beam_length_cross, .5 * (beam_length_main + beam_length_cross)]
        columns_beam_length = [beam_length_main + beam_length_cross, beam_length_main + .5 * beam_length_cross,
                       .5 * beam_length_main + beam_length_cross, .5 * (beam_length_main + beam_length_cross)]

        # Axial forces for floors and roof
        columns_floor_axial = [
            a * floor_area_load + b * self.beam_load + c * self.infill_load + column_load
            for a, b, c in zip(columns_area, columns_beam_length, columns_infill_length)
        ]
        columns_roof_axial = [
            a * roof_area_load + b * self.beam_load + column_load
            for a, b in zip(columns_area, columns_beam_length)
        ]

        # Convert to member solicitations
        column_sollicitations_roof = [
            MemberSollicitation(N=axial) for axial in columns_roof_axial
        ]
        column_sollicitations_floor = [
            MemberSollicitation(N=axial) for axial in columns_floor_axial
        ]

        # Collect axial forces across floors
        columns_axial = [np.array(column_sollicitations_floor)] * (floors - 1) + [np.array(column_sollicitations_roof)]

        # Sum axial forces from top to bottom for each floor
        columns_total_axial = [sum(columns_axial[i:], np.array(MemberSollicitation())) for i in range(len(columns_axial))]

        return [ColumnSollicitations(*columns) for columns in columns_total_axial]


    def compute_gravity_loads(self, building_geometry: BuildingGeometry) -> BuildingSollicitations:
        """
        Compute the gravity loads for the entire building, including both beam and column solicitations.

        :param building_geometry: Object that holds the building's geometrical information.
        :return: A structure containing the solicitations for all beams and columns.
        """

        columns_sollicitations = self.compute_column_axials(building_geometry)
        beams_sollicitations = self.compute_beam_moments(building_geometry)
        # Assemble the floors
        return BuildingSollicitations(
            beams_sollicitations=beams_sollicitations,
            columns_sollicitations=columns_sollicitations
        )
