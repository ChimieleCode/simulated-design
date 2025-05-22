


from dataclasses import dataclass, field

from src.frame_definition import BuildingGeometry
from src.section_design.beam_design import design_beam_element
from src.section_design.column_design import bidirectional_column_design
from src.section_design.detailing_minimums import (DetailingCode, RebarsType,
                                                   get_shear_stirrups_share)
from src.section_design.section_geometry import RectangularSectionElement
from src.sollicitations import MemberSollicitation
from src.structure_loads import GravityLoadDesignFullSpan, SeismicLoadDesign


@dataclass
class BeamDesignSollicitations:
    """
    Class to represent the design solicitations for beams.
    """
    Mpos: float
    Mneg: float
    Vmax: float


@dataclass
class ColumnDesignSollicitations:
    """
    Class to represent the design solicitations for columns.
    """
    M_main: float
    M_cross: float
    V_main: float
    V_cross: float
    N: float


@dataclass
class BuildingSecitions:
    """
    Class to represent the sections of a building.
    """
    # Define the attributes of the class
    columns: list[                  # List of columns in the building for each floor
        tuple[    # Along main span          # Along cross span
            tuple[RectangularSectionElement, RectangularSectionElement],  # Internal columns
            tuple[RectangularSectionElement, RectangularSectionElement],  # Border columns along the main span
            tuple[RectangularSectionElement, RectangularSectionElement],  # Border columns along thr cross span
            tuple[RectangularSectionElement, RectangularSectionElement]   # Corner columns
            ]
        ]
    beams: list[                    # List of beams in the building for each floor
        tuple[
            RectangularSectionElement,  # Internal beams along the main span
            RectangularSectionElement,  # Border beams along the main span
            RectangularSectionElement,  # Internal beams along the cross span
            RectangularSectionElement   # Border beams along the cross span
            ]
        ]


@dataclass
class BeamDesignOptions:
    cop: float
    shear_rebar_diameter: int
    b_section: float


@dataclass
class ColumnDesignOptions:
    cop: float
    shear_rebar_diameter: int


@dataclass
class AdmStresses:
    sigma_cls_adm: float
    sigma_s_adm: float
    n: float = field(default=15.)



class BuildingSectionDesign:

    def __init__(self,
                 building_geometry: BuildingGeometry,
                 gravity_load_design: GravityLoadDesignFullSpan,
                 detailing_code: DetailingCode,
                 rebar_type: RebarsType,
                 seismic_load_design: SeismicLoadDesign | None = None,
                 ):
        self.building_geometry = building_geometry
        self.gravity_load_design = gravity_load_design
        self.seismic_load_design = seismic_load_design
        self.detailing_code = detailing_code
        self.rebar_type = rebar_type


        # Compute sollicitations
        self.gravity_loads = gravity_load_design.compute_gravity_loads(building_geometry)
        self.permanent_loads = gravity_load_design.compute_gravity_loads(building_geometry, include_overload=False)

        # Seismic Loads
        if seismic_load_design is not None:
            self.seismic_loads = seismic_load_design.compute_frame_sollicitations(building_geometry)

        # Shear Factor Beams
        self.beta = get_shear_stirrups_share(detailing_code)

    def get_beam_sollicitations(self) -> list[tuple[BeamDesignSollicitations, BeamDesignSollicitations, BeamDesignSollicitations, BeamDesignSollicitations]]:
        """
        Get the design solicitations for beams.
        """
        beam_gravity_loads = self.gravity_loads.beams_sollicitations
        beam_permanent_loads = self.permanent_loads.beams_sollicitations

        # Compute the design solicitations for beams
        beams: list[tuple[BeamDesignSollicitations, BeamDesignSollicitations, BeamDesignSollicitations, BeamDesignSollicitations]] = list()
        for floor, (beam_gravity_soll, beam_permanent_soll) in enumerate(zip(beam_gravity_loads, beam_permanent_loads)):
            # Get the design solicitations for each beam
            beams_gq = (
                beam_gravity_soll.internal_beam_main,   # Main span internal
                beam_gravity_soll.border_beam_main,     # Main span border
                beam_gravity_soll.internal_beam_cross,  # Cross span internal
                beam_gravity_soll.border_beam_cross     # Cross span border
            )
            beams_g = (
                beam_permanent_soll.internal_beam_main,  # Main span internal
                beam_permanent_soll.border_beam_main,    # Main span border
                beam_permanent_soll.internal_beam_cross, # Cross span internal
                beam_permanent_soll.border_beam_cross    # Cross span border
            )
            # Seismic
            beam_e = (MemberSollicitation(),) * 4  # Initialize with empty MemberSollicitation
            if self.seismic_load_design is not None:
                beam_e = (
                    self.seismic_loads[0].get_beam_sollicitations(floor),   # Main span internal
                    self.seismic_loads[0].get_beam_sollicitations(floor),   # Main span border
                    self.seismic_loads[1].get_beam_sollicitations(floor),   # Cross span internal
                    self.seismic_loads[1].get_beam_sollicitations(floor),   # Cross span border
                )

            # Compute the design solicitations for each beam
            beam_design_soll: list[BeamDesignSollicitations] = list()
            for beam_gq, beam_g, beam_e in zip(beams_gq, beams_g, beam_e):
                beam_design_soll.append(
                    BeamDesignSollicitations(
                        Mpos=beam_gq.M + beam_e.M,
                        Mneg=max(beam_e.M - beam_g.M, 0),
                        Vmax=self.beta * beam_gq.V + beam_e.V
                    )
                )
            # Append the design solicitations for the current floor
            beams.append(tuple(beam_design_soll))   # type: ignore[reportArgumentType]
        return beams

    def get_column_sollicitations(self) -> list[tuple[ColumnDesignSollicitations, ColumnDesignSollicitations, ColumnDesignSollicitations, ColumnDesignSollicitations]]:
        """
        Get the design solicitations for columns.
        """
        column_gravity_sollicitations = self.gravity_loads.columns_sollicitations
        # Compute the design solicitations for beams
        columns: list[tuple[ColumnDesignSollicitations, ColumnDesignSollicitations, ColumnDesignSollicitations, ColumnDesignSollicitations]] = list()
        for floor, column_gravity_sollicitations in enumerate(column_gravity_sollicitations):
            # Get the design solicitations for each beam
            columns_g = (
                column_gravity_sollicitations.central_column,       # Internal columns
                column_gravity_sollicitations.border_column_main,   # Border columns along the main span
                column_gravity_sollicitations.border_column_cross,  # Border columns along thr cross span
                column_gravity_sollicitations.corner_column         # Corner columns
            )
            # Seismic
            columns_e_main = (MemberSollicitation(),) * 4  # Initialize with empty MemberSollicitation
            columns_e_cross = (MemberSollicitation(),) * 4  # Initialize with empty MemberSollicitation
            if self.seismic_load_design is not None:
                columns_e_main = (
                    self.seismic_loads[0].get_internal_column_sollicitations(floor),   # Internal columns
                    self.seismic_loads[0].get_internal_column_sollicitations(floor),   # Border columns along the main span
                    self.seismic_loads[0].get_external_column_sollicitations(floor),   # Border columns along thr cross span
                    self.seismic_loads[0].get_external_column_sollicitations(floor),   # Corner columns
                )
                columns_e_cross = (
                    self.seismic_loads[1].get_internal_column_sollicitations(floor),   # Internal columns
                    self.seismic_loads[1].get_internal_column_sollicitations(floor),   # Border columns along the main span
                    self.seismic_loads[1].get_external_column_sollicitations(floor),   # Border columns along thr cross span
                    self.seismic_loads[1].get_external_column_sollicitations(floor),   # Corner columns
                )

            # Compute the design solicitations for each beam
            column_design_soll: list[ColumnDesignSollicitations] = list()
            for cols_g, cols_e_main, cols_e_cross in zip(columns_g, columns_e_main, columns_e_cross):
                column_design_soll.append(
                    ColumnDesignSollicitations(
                        M_main=cols_e_main.M,
                        M_cross=cols_e_cross.M,
                        V_main=cols_e_main.V,
                        V_cross=cols_e_cross.V,
                        N=cols_g.N
                    )
                )
            # Append the design solicitations for the current floor
            columns.append(tuple(column_design_soll))   # type: ignore[reportArgumentType]
        return columns

    def design_building(self,
                        beam_design_options: BeamDesignOptions,
                        column_design_options: ColumnDesignOptions,
                        beam_adm_stresses: AdmStresses,
                        column_adm_stresses: AdmStresses) -> BuildingSecitions:
        """
        Design the building sections based on the provided loads and detailing code.
        """
        beam_sollicitations = self.get_beam_sollicitations()
        column_sollicitations = self.get_column_sollicitations()

        # Create the sections for the building
        beam_sections: list[
            tuple[
                RectangularSectionElement,  # Main span internal
                RectangularSectionElement,  # Main span border
                RectangularSectionElement,  # Cross span internal
                RectangularSectionElement   # Cross span border
            ]
        ] = list()
        for floor, beam_soll_floor in enumerate(beam_sollicitations):
            # Create the sections for each beam
            beam_section_floor = tuple()
            for i, beam_soll in enumerate(beam_soll_floor):
                print(f'Design beam section at {floor} {i}', beam_soll)
                beam_sec = design_beam_element(
                    **beam_soll.__dict__,
                    **beam_adm_stresses.__dict__,
                    **beam_design_options.__dict__,
                    detailing_code=self.detailing_code,
                    rebar_type=self.rebar_type
                )
                beam_section_floor += (beam_sec,)
            beam_sections.append(beam_section_floor)    # type: ignore[reportArgumentType]

        # Columns
        column_sections: list[
            tuple[
                tuple[RectangularSectionElement, RectangularSectionElement],    # Internal columns
                tuple[RectangularSectionElement, RectangularSectionElement],    # Border columns along the main span
                tuple[RectangularSectionElement, RectangularSectionElement],    # Border columns along thr cross span
                tuple[RectangularSectionElement, RectangularSectionElement]     # Corner columns
            ]
        ] = list()
        for floor, column_soll_floor in enumerate(column_sollicitations):
            # Create the sections for each column
            column_sections_floor = tuple()
            for i, column_soll in enumerate(column_soll_floor):
                print(f'Designed column section at {floor} {i}', column_soll)
                main, cross = bidirectional_column_design(
                    **column_soll.__dict__,
                    **column_adm_stresses.__dict__,
                    **column_design_options.__dict__,
                    detailing_code=self.detailing_code
                )
                column_sections_floor += ((main, cross),)
            column_sections.append(column_sections_floor)    # type: ignore[reportArgumentType]

        return BuildingSecitions(
            columns=column_sections,
            beams=beam_sections
        )
