from dataclasses import dataclass

from src.section_design.section_utils import reinforcement_area

# Units
mm_m = 1e-3  # mm to m


@dataclass
class SectionGeometry:
    h: float
    b: float
    cop: float

    def rotate_90(self, new_section: bool = True) -> 'SectionGeometry':
        """
        Rotates the section geometry by 90 degrees.
        Swaps the height and width of the section.
        """
        if new_section:
            return type(self)(self.b, self.h, self.cop)
        else:
            self.h, self.b = self.b, self.h
            return self

    @property
    def area(self) -> float:
        """
        Computes the area of the section.
        """
        return self.h * self.b


@dataclass
class RectangularSection(SectionGeometry):
    top_reinf_count: int
    top_reinf_d: int    # mm
    bot_reinf_count: int
    bot_reinf_d: int    # mm

    @property
    def top_reinf_area(self) -> float:
        """
        Computes the area of the top reinforcement.
        """
        return reinforcement_area(
            count=self.top_reinf_count,
            diameter=self.top_reinf_d * mm_m  # Convert to meters
        )

    @property
    def bot_reinf_area(self) -> float:
        """
        Computes the area of the bot reinforcement.
        """
        return reinforcement_area(
            count=self.bot_reinf_count,
            diameter=self.bot_reinf_d * mm_m  # Convert to meters
        )


@dataclass
class RectangularSectionElement(RectangularSection):
    stirrups_reinf_d: int   # mm
    stirrups_spacing: float

    @property
    def stirrups_reinf_area(self) -> float:
        """
        Computes the area of the stirrups reinforcement.
        """
        return reinforcement_area(
            count=2,
            diameter=self.stirrups_reinf_d * mm_m  # Convert to meters
        )
