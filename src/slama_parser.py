
import numpy as np

from src.frame_definition import BuildingGeometry
from src.section_design.section_geometry import RectangularSectionElement
from src.sollicitation_section_dispatch import BuildingSecitions
from src.structure_loads import (ColumnSollicitations,
                                 GravityLoadDesignFullSpan, MassDefinition,
                                 StructureLoads)

# Column
INTERNAL: int = 0
BORDER_MAIN: int = 1
BORDER_CROSS: int = 2
CORNER: int = 3

# Beams
MAIN_INT: int = 0
MAIN_EXT: int = 1
CROSS_INT: int = 2
CROSS_EXT: int = 3

# Units
mm_m: float = 0.001


class SLaMAInputBuilder:
    """
    A class to parse SLaMA (Sparse Language Model Architecture) model files.
    """

    def __init__(
            self,
            building_geometry: BuildingGeometry,
            sections: BuildingSecitions,
            structure_loads: StructureLoads,
            overload_factor: float = 1.0
            ):
        """
        Initialize the SLaMAInputBuilder with the given parameters.

        :param building_geometry: building geometry object containing dimensions and properties.
        :param sections: building sections object containing section properties.
        :param structure_loads: structure loads object containing load properties.
        :param overload_factor: overload factor for the structure loads. Default is 1.0.
        """
        self.building_geometry = building_geometry
        self.sections = sections
        self.overload_factor = overload_factor

        # Added
        if building_geometry.is_fully_braced:
            vertical_loads = GravityLoadDesignFullSpan(**structure_loads.__dict__)\
                .compute_vertical_sollicitations(building_geometry, overload_factor=overload_factor)

            self.vertical_loads = [ColumnSollicitations(*floor) for floor in vertical_loads]
        else:
            raise NotImplementedError(
                'The SLaMAInputBuilder only supports fully braced structures at the moment.'
            )

        self.masses = MassDefinition(**structure_loads.__dict__)\
            .get_mass(building_geometry, overload_factor=overload_factor)

        # Define the section_dictionary
        self.beam_sections: list[RectangularSectionElement] = []
        self.column_sections: list[RectangularSectionElement] = []


    def _build_basic_frame(self) -> dict:
        # Geometry
        H = self.building_geometry.comulative_floor_height
        L = [i * self.building_geometry.span_main for i in range(self.building_geometry.n_cross_frames)]

        return {
            'H': H,
            'L': L
        }

    def _build_interal_main_frame(self) -> dict:
        """
        Load the SLaMA model from the specified path.
        """
        basic_frame = self._build_basic_frame()

        # Loads
        loads: list[float] = [0.] * self.building_geometry.n_cross_frames
        for floor in range(self.building_geometry.floors):
            ext_col = self.vertical_loads[floor].border_column_cross.N
            int_col = self.vertical_loads[floor].central_column.N
            loads += [ext_col] + [int_col] * (self.building_geometry.n_cross_frames - 2) + [ext_col]

        # Section
        columns: list[list[int]] = []
        beams: list[list[int]] = []
        for floor in range(self.building_geometry.floors):
            # Columns, 0 along main
            ext_col = self.sections.columns[floor][BORDER_CROSS][0]
            ext_col_id = len(self.column_sections)
            self.column_sections.append(ext_col)

            int_col = self.sections.columns[floor][INTERNAL][0]
            int_col_id = len(self.column_sections)
            self.column_sections.append(int_col)

            columns.append(
                [int_col_id] + [ext_col_id] * (self.building_geometry.n_cross_frames - 2) + [int_col_id]
            )

            # Beams
            beam = self.sections.beams[floor][MAIN_INT]
            beam_id = len(self.beam_sections)
            self.beam_sections.append(beam)

            beams.append(
                [beam_id] * self.building_geometry.n_main_spans
            )

        # Masses
        masses = list(np.array(self.masses) / self.building_geometry.n_main_frames)

        return {
            **basic_frame,
            'm': masses,
            'loads': loads,
            'columns': columns,
            'beams': beams
        }



    def _build_external_main_frame(self) -> dict:
        """
        Parse the loaded SLaMA model.
        """
        basic_frame = self._build_basic_frame()

        # Loads
        loads: list[float] = [0.] * self.building_geometry.n_cross_frames
        for floor in range(self.building_geometry.floors):
            ext_col = self.vertical_loads[floor].corner_column.N
            int_col = self.vertical_loads[floor].border_column_main.N
            loads += [ext_col] + [int_col] * (self.building_geometry.n_cross_frames - 2) + [ext_col]

        # Section
        columns: list[list[int]] = []
        beams: list[list[int]] = []
        for floor in range(self.building_geometry.floors):
            # Columns, 0 along main
            ext_col = self.sections.columns[floor][CORNER][0]
            ext_col_id = len(self.column_sections)
            self.column_sections.append(ext_col)

            int_col = self.sections.columns[floor][BORDER_MAIN][0]
            int_col_id = len(self.column_sections)
            self.column_sections.append(int_col)

            columns.append(
                [int_col_id] + [ext_col_id] * (self.building_geometry.n_cross_frames - 2) + [int_col_id]
            )

            # Beams
            beam = self.sections.beams[floor][MAIN_EXT]
            beam_id = len(self.beam_sections)
            self.beam_sections.append(beam)

            beams.append(
                [beam_id] * self.building_geometry.n_main_spans
            )

        # Masses
        masses = list(np.array(self.masses) / self.building_geometry.n_main_frames)

        return {
            **basic_frame,
            'm': masses,
            'loads': loads,
            'columns': columns,
            'beams': beams
        }

    def _build_internal_cross_frame(self) -> dict:
        """
        Extract the internal cross frame from the parsed SLaMA model.
        """
        basic_frame = self._build_basic_frame()

        # Loads
        loads: list[float] = [0.] * self.building_geometry.n_cross_frames
        for floor in range(self.building_geometry.floors):
            ext_col = self.vertical_loads[floor].border_column_main.N
            int_col = self.vertical_loads[floor].central_column.N
            loads += [ext_col] + [int_col] * (self.building_geometry.n_cross_frames - 2) + [ext_col]

        # Section
        columns: list[list[int]] = []
        beams: list[list[int]] = []
        for floor in range(self.building_geometry.floors):
            # Columns, 0 along main
            ext_col = self.sections.columns[floor][BORDER_MAIN][1]
            ext_col_id = len(self.column_sections)
            self.column_sections.append(ext_col)

            int_col = self.sections.columns[floor][INTERNAL][1]
            int_col_id = len(self.column_sections)
            self.column_sections.append(int_col)

            columns.append(
                [int_col_id] + [ext_col_id] * (self.building_geometry.n_cross_frames - 2) + [int_col_id]
            )

            # Beams
            beam = self.sections.beams[floor][CROSS_INT]
            beam_id = len(self.beam_sections)
            self.beam_sections.append(beam)

            beams.append(
                [beam_id] * self.building_geometry.n_main_spans
            )

        # Masses
        masses = list(np.array(self.masses) / self.building_geometry.n_cross_frames)
        return {
            **basic_frame,
            'm': masses,
            'loads': loads,
            'columns': columns,
            'beams': beams
        }

    def _build_external_cross_frame(self) -> dict:
        """
        Extract the external cross frame from the parsed SLaMA model.
        """
        basic_frame = self._build_basic_frame()

        # Loads
        loads: list[float] = [0.] * self.building_geometry.n_cross_frames
        for floor in range(self.building_geometry.floors):
            ext_col = self.vertical_loads[floor].corner_column.N
            int_col = self.vertical_loads[floor].border_column_cross.N
            loads += [ext_col] + [int_col] * (self.building_geometry.n_cross_frames - 2) + [ext_col]

        # Section
        columns: list[list[int]] = []
        beams: list[list[int]] = []
        for floor in range(self.building_geometry.floors):
            # Columns, 0 along main
            ext_col = self.sections.columns[floor][CORNER][1]
            ext_col_id = len(self.column_sections)
            self.column_sections.append(ext_col)

            int_col = self.sections.columns[floor][BORDER_CROSS][1]
            int_col_id = len(self.column_sections)
            self.column_sections.append(int_col)

            columns.append(
                [int_col_id] + [ext_col_id] * (self.building_geometry.n_cross_frames - 2) + [int_col_id]
            )

            # Beams
            beam = self.sections.beams[floor][CROSS_EXT]
            beam_id = len(self.beam_sections)
            self.beam_sections.append(beam)

            beams.append(
                [beam_id] * self.building_geometry.n_main_spans
            )

        # Masses
        masses = list(np.array(self.masses) / self.building_geometry.n_cross_frames)
        return {
            **basic_frame,
            'm': masses,
            'loads': loads,
            'columns': columns,
            'beams': beams
        }

    def build_input_file(self, tag: str, concrete: dict, steel: dict) -> dict:
        """
        Generate the input file for the frame analysis.
        """
        # Must be called before the sections
        main_frame = self._build_interal_main_frame()
        main_border_frame = self._build_external_main_frame()
        cross_frame = self._build_internal_cross_frame()
        cross_border_frame = self._build_external_cross_frame()

        # Sections
        sections_dict = self.build_sections_input_file()

        self.beam_sections.clear()
        self.column_sections.clear()

        return {
            'tag': tag,
            'materials': {
                'concrete': concrete,
                'steel': steel
            },
            'sections': sections_dict,
            'frames': {
                'main_frames': [
                    (main_frame, (self.building_geometry.n_main_frames - 2)),
                    (main_border_frame, 2)
                ],
                'border_frames': [
                    (cross_frame, (self.building_geometry.n_cross_frames - 2)),
                    (cross_border_frame, 2)
                ]
            },
            'masses': self.masses
        }


    def build_sections_input_file(self) -> dict:
        """
        Generate the input file for the section analysis.
        """
        columns = [
            parse_rectangular_section(
                section,
                name=f"Column {i}"
            ) for i, section in enumerate(self.column_sections)
        ]
        beams = [
            parse_rectangular_section(
                section,
                name=f"Beam {i}",
            ) for i, section in enumerate(self.beam_sections)
        ]
        return {
            'columns': columns,
            'beams': beams
        }


def parse_rectangular_section(section: RectangularSectionElement, name: str | None = None, swap_reinf: bool = False) -> dict:
    """
    Parse a rectangular section from the input dictionary.
    """
    if name is None:
        name = ''

    As, As1 = section.bot_reinf_area, section.top_reinf_area
    if swap_reinf:
        As, As1 = As1, As

    return {
        'id': name,
        'h': section.h,
        'b': section.b,
        'As': As,       # Bottom
        'As1': As1,     # Top
        'cover': section.cop,
        'eq_bar_diameter': max(section.bot_reinf_d, section.top_reinf_d) * mm_m,
        'Ast': section.stirrups_reinf_area,
        's': section.stirrups_spacing
    }
