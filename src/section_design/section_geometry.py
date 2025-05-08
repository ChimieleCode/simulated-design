from dataclasses import dataclass


@dataclass
class SectionGeometry:
    h: float
    b: float
    cop: float


@dataclass
class RectangularSection(SectionGeometry):
    top_reinf_count: int
    top_reinf_d: float
    bot_reinf_count: int
    bot_reinf_d: float
