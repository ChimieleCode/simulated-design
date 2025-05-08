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
            permanent_weight_floor,
            overloads_floor,
            permanent_weight_roof,
            overloads_roof
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

    def _compute_seismic_weights(self,
                                 permanent_weight_floor: float,
                                 overloads_floor: float,
                                 permanent_weight_roof: float,
                                 overloads_roof: float) -> tuple[float, float]:
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

    def _compute_seismic_forces(self,
                                seismic_weight_floor: float,
                                seismic_weight_roof: float,
                                building_geometry: BuildingGeometry) -> list[float]:
        """Computes the seismic forces based on seismic weight distribution."""
        seismic_weights = [seismic_weight_floor] * (building_geometry.floors - 1) + [seismic_weight_roof]
        seismic_forces = SeismicForces().compute_forces(
            building_code=self.building_code,
            weights=seismic_weights,
            seismic_cat=self.seismic_cat,
            floor_heights=building_geometry.comulative_floor_height
        )
        return seismic_forces

    def _compute_frame_forces(self,
                              seismic_forces: list[float],
                              building_geometry: BuildingGeometry) -> tuple[list[float], list[float]]:
        """Computes the forces for both the main and cross frames."""
        main_frame_forces = [force / (building_geometry.n_cross_spans + 1) for force in seismic_forces]
        cross_frame_forces = [force / (building_geometry.n_main_spans + 1) for force in seismic_forces]
        return main_frame_forces, cross_frame_forces

    def _compute_seismic_stresses(self,
                                  frame_forces: list[float],
                                  building_geometry: BuildingGeometry, is_main: bool) -> RegularSpanFrameSollicitations:
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

    def compute_gravity_loads(self, building_geometry: BuildingGeometry) -> BuildingSollicitations:
        """
        Compute the gravity loads for the entire building, including both beam and column solicitations.
        """
        columns = self.compute_column_axials(building_geometry)
        beams = self.compute_beam_moments(building_geometry)
        return BuildingSollicitations(beams_sollicitations=beams, columns_sollicitations=columns)

    def compute_beam_moments(self, building_geometry: BuildingGeometry) -> list[BeamSollicitations]:
        floors = building_geometry.floors
        loads_floor, loads_roof = self._get_beam_loads(building_geometry)

        floor_moments = [compute_beam_moment_end(q, L) for q, L in loads_floor]
        roof_moments = [compute_beam_moment_end(q, L) for q, L in loads_roof]
        floor_shears = [compute_beam_shear_end(q, L) for q, L in loads_floor]
        roof_shears = [compute_beam_shear_end(q, L) for q, L in loads_roof]

        solicitations_floor = [MemberSollicitation(M=m, V=v) for m, v in zip(floor_moments, floor_shears)]
        solicitations_roof = [MemberSollicitation(M=m, V=v) for m, v in zip(roof_moments, roof_shears)]

        all_solicitations = [solicitations_floor] * (floors - 1) + [solicitations_roof]
        return [BeamSollicitations(*floor) for floor in all_solicitations]

    def compute_column_axials(self, building_geometry: BuildingGeometry) -> list[ColumnSollicitations]:
        floors = building_geometry.floors
        roof, floor = self._get_column_loads(building_geometry)

        solicitations_roof = [MemberSollicitation(N=load) for load in roof]
        solicitations_floor = [MemberSollicitation(N=load) for load in floor]

        combined = [np.array(solicitations_floor)] * (floors - 1) + [np.array(solicitations_roof)]
        total = [sum(combined[i:], np.array(MemberSollicitation())) for i in range(floors)]

        return [ColumnSollicitations(*floor) for floor in total]

    def _get_beam_loads(self, bg: BuildingGeometry) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        floor_load = self.floaring_load + self.overload
        roof_load = self.floaring_load + self.roof_overload
        Lx, Ly = bg.span_main, bg.span_cross

        beams_area = [.5 * Ly, 0, Ly, 0]
        beams_infill = [1, 1, 0, 0]
        beams_length = [Lx, Ly, Lx, Ly]

        floor_loads = [
            (floor_load * A + self.infill_load * F + self.beam_load, L)
            for A, F, L in zip(beams_area, beams_infill, beams_length)
        ]
        roof_loads = [
            (roof_load * A + self.beam_load, L)
            for A, L in zip(beams_area, beams_length)
        ]
        return floor_loads, roof_loads

    def _get_column_loads(self, bg: BuildingGeometry) -> tuple[list[float], list[float]]:
        floor_area = self.floaring_load + self.overload
        roof_area = self.floaring_load + self.roof_overload
        h = bg.floor_height
        Lx, Ly = bg.span_main, bg.span_cross
        A = Lx * Ly

        column_load = self.column_load * h
        area_factors = np.array([A, 0.5 * A, 0.5 * A, 0.25 * A])
        beam_lengths = np.array([Lx + Ly, Lx + 0.5 * Ly, 0.5 * Lx + Ly, 0.5 * (Lx + Ly)])
        infill_lengths = np.array([0, Lx, Ly, 0.5 * (Lx + Ly)])

        floor = area_factors * floor_area + beam_lengths * self.beam_load + infill_lengths * self.infill_load + column_load
        roof = area_factors * roof_area + beam_lengths * self.beam_load + column_load
        # Return tolist to convert numpy arrays to lists
        return roof.tolist(), floor.tolist()
