from __future__ import annotations

from src.frame_definition import BuildingBox, BuildingGeometryFactory
from src.seismic_codes import BuildingCode, SeismicCat
from src.structure_loads import GravityLoadDesignFullSpan, SeismicLoadDesign

EXAMPLE_INPUT: dict = {
    'geometry': {
        'length': 20.0,
        'width': 16.0,
        'height': 7.0,
        'floors': 2,
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
    }
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

    # STEP 2: define the loads
    # Cmpute seismic loads for the structure
    seismic_design = SeismicLoadDesign(
        **input['seismic'],
        **input['gravity']
    )
    seismic_loads = seismic_design.compute_frame_sollicitations(building_geometry)
    print(seismic_loads)

    # Compute gravity loads for the structure
    gravity_design = GravityLoadDesignFullSpan(
        **input['gravity']
    )
    gravity_loads = gravity_design.compute_gravity_loads(building_geometry)
    print(gravity_loads)

    # Compute permanent loads for the structure
    permanent_loads = gravity_design.compute_gravity_loads(building_geometry, include_overload=False)
    print(permanent_loads)



if __name__ == '__main__':
    main(EXAMPLE_INPUT)
