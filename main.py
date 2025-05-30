from __future__ import annotations

import argparse
from pathlib import Path

from model.input_validation import InputModel
from src.frame_definition import BuildingBox, BuildingGeometryFactory
from src.section_design.detailing_minimums import DetailingCode, RebarsType
from src.slama_parser import SLaMAInputBuilder
from src.sollicitation_section_dispatch import (AdmStresses, BeamDesignOptions,
                                                BuildingSectionDesign,
                                                ColumnDesignOptions)
from src.structure_loads import (GravityLoadDesignFullSpan, SeismicLoadDesign,
                                 StructureLoads)
from src.utils import export_to_json, import_from_json


def main(input_dct: dict, output_path: Path) -> None:
    """
    Main function to create a BuildingGeometry object and print its details.

    :param input: Dictionary containing building geometry parameters.
    """
    # STEP 1: create the frame grid system

    # Create a BuildingBox object from the input dictionary
    bbox = BuildingBox(**input_dct['geometry'])

    # Create a BuildingGeometryFactory object with specified parameters
    factory = BuildingGeometryFactory(
        main_target_span=4.0,
        cross_target_span=4.5,
        floor_height=3.0,
    )

    # Create a BuildingGeometry object using the factory
    building_geometry = factory.create(bbox, is_fully_braced=True)

    # Design Options
    beam_adm_stress = AdmStresses(
        **input_dct['beams']['adm_stress']
    )
    beam_opts = BeamDesignOptions(
        **input_dct['beams']['options']
    )
    col_adm_stress = AdmStresses(
        **input_dct['columns']['adm_stress']
    )
    col_opts = ColumnDesignOptions(
        **input_dct['columns']['options']
    )
    detailing_code = DetailingCode(input_dct['details'])
    rebar_type = RebarsType(input_dct['rebar_type'])

    # STEP 2: define the loads
    # Get and parse vertical loads
    strucutre_loads = StructureLoads(
        **input_dct['gravity']
    )

    # Compute seismic loads for the structure
    seismic_design = SeismicLoadDesign(
        **strucutre_loads.__dict__,
        **input_dct['seismic']
    )
    seismic_loads = seismic_design.compute_frame_sollicitations(building_geometry)
    print(seismic_loads)

    # Compute gravity loads for the structure
    gravity_design = GravityLoadDesignFullSpan(
        **strucutre_loads.__dict__
    )
    gravity_loads = gravity_design.compute_gravity_loads(building_geometry)
    print(gravity_loads)

    # Compute permanent loads for the structure
    permanent_loads = gravity_design.compute_gravity_loads(building_geometry, include_overload=False)
    print(permanent_loads)

    # STEP 3: define the design options
    buidling_section_design = BuildingSectionDesign(
        building_geometry=building_geometry,
        detailing_code=detailing_code,
        rebar_type=rebar_type,
        gravity_load_design=gravity_design,
        seismic_load_design=seismic_design
    )
    sections = buidling_section_design.design_building(
        beam_design_options=beam_opts,
        column_design_options=col_opts,
        beam_adm_stresses=beam_adm_stress,
        column_adm_stresses=col_adm_stress
    )

    # STEP 4: parse
    slama_parser = SLaMAInputBuilder(
        building_geometry=building_geometry,
        sections=sections,
        structure_loads=strucutre_loads,
        overload_factor=.3
    )
    # Refactor and include in the input
    slama_file = slama_parser.build_input_file(
        'Test',
        concrete={
            'id': 'C14',
            'fc': 16000.0,
            'E': 25186010.36,
            'epsilon_0': 0.0007,
            'epsilon_u': 0.0035
        },
        steel={
            'id': 'Fe430',
            'fy': 392000.0,
            'fu': 420000.0,
            'E': 210000000,
            'epsilon_u': 0.06
        }
    )

    # STEP 5: Save the file
    export_to_json(
        output_path,
        slama_file
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process building design input and output files.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--output', type=str, default='./example.json', help='Path to the output JSON file')
    args = parser.parse_args()


    input_dict = import_from_json(Path(args.input))
    output_path = Path(args.output)

    validated_input = InputModel(**input_dict)

    main(validated_input.model_dump(), output_path=output_path)
