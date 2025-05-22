from __future__ import annotations

from pathlib import Path

from src.frame_definition import BuildingBox, BuildingGeometryFactory
from src.section_design.detailing_minimums import DetailingCode, RebarsType
from src.seismic_codes import BuildingCode, SeismicCat
from src.slama_parser import SLaMAInputBuilder
from src.sollicitation_section_dispatch import (AdmStresses, BeamDesignOptions,
                                                BuildingSectionDesign,
                                                ColumnDesignOptions)
from src.structure_loads import (GravityLoadDesignFullSpan, SeismicLoadDesign,
                                 StructureLoads)
from src.utils import export_to_json

EXAMPLE_INPUT: dict = {
    'geometry': {
        'length': 20.0,
        'width': 16.0,
        'height': 7.0,
        'floors': 3,
    },
    'seismic': {
        'building_code': BuildingCode.RDL_2105,
        'seismic_cat': SeismicCat.CatII
    },
    'gravity': {
        'floaring_load': 5,
        'infill_load': 7.5,
        'beam_load': 3.75,
        'overload': 3,
        'roof_overload': 1.5,
        'column_load': 13.5,
    },
    'columns': {
        'adm_stress': {
            'sigma_s_adm': 180000,
            'sigma_cls_adm': 8000,
            'n': 15.
        },
        'options': {
            'cop': 0.03,
            'shear_rebar_diameter': 8
        },
    },
    'beams': {
        'adm_stress': {
            'sigma_s_adm': 180000,
            'sigma_cls_adm': 8000,
            'n': 15.
        },
        'options': {
            'cop': 0.03,
            'shear_rebar_diameter': 8,
            'b_section': 0.3
        },
    },
    'details': 'DM_76',
    'rebar_type': 'Deformed'
}

def main(input: dict):
    """
    Main function to create a BuildingGeometry object and print its details.

    :param input: Dictionary containing building geometry parameters.
    """
    # STEP 1: create the frame grid system

    # Create a BuildingBox object from the input dictionary
    bbox = BuildingBox(**input['geometry'])

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
        **input['beams']['adm_stress']
    )
    beam_opts = BeamDesignOptions(
        **input['beams']['options']
    )
    col_adm_stress = AdmStresses(
        **input['columns']['adm_stress']
    )
    col_opts = ColumnDesignOptions(
        **input['columns']['options']
    )
    detailing_code = DetailingCode(input['details'])
    rebar_type = RebarsType(input['rebar_type'])

    # STEP 2: define the loads
    # Get and parse vertical loads
    strucutre_loads = StructureLoads(
        **input['gravity']
    )

    # Compute seismic loads for the structure
    seismic_design = SeismicLoadDesign(
        **strucutre_loads.__dict__,
        **input['seismic']
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
        Path('./example.json'),
        slama_file
    )

if __name__ == '__main__':
    main(EXAMPLE_INPUT)
