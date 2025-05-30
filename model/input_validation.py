from pydantic import BaseModel

from src.section_design.detailing_minimums import DetailingCode, RebarsType
from src.seismic_codes import BuildingCode, SeismicCat


class Geometry(BaseModel):
    length: float
    width: float
    height: float
    floors: int

class Seismic(BaseModel):
    building_code: BuildingCode
    seismic_cat: SeismicCat

class Gravity(BaseModel):
    floaring_load: float
    infill_load: float
    beam_load: float
    overload: float
    roof_overload: float
    column_load: float

class AdmStress(BaseModel):
    sigma_s_adm: int
    sigma_cls_adm: int
    n: int

class ColumnOptions(BaseModel):
    cop: float
    shear_rebar_diameter: int

class BeamOptions(ColumnOptions):
    b_section: float | None

class Columns(BaseModel):
    adm_stress: AdmStress
    options: ColumnOptions

class Beams(BaseModel):
    adm_stress: AdmStress
    options: BeamOptions

class InputModel(BaseModel):
    geometry: Geometry
    seismic: Seismic
    gravity: Gravity
    columns: Columns
    beams: Beams
    details: DetailingCode
    rebar_type: RebarsType
