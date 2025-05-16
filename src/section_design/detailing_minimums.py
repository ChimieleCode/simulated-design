from enum import Enum

from src.section_design.section_geometry import (RectangularSection,
                                                 RectangularSectionElement)
from src.section_design.section_utils import reinforcement_area

# Constants
mmq_mq: float = 1e-6  # Conversion factor from mm^2 to m^2


class RebarsType(Enum):
    DEFORMED = 'Deformed'
    PLAIN = 'Plain'


class DetailingCode(Enum):
    RD_39 = 'RD_39'
    DM_76 = 'DM_76'


class BeamDetailsRD39:
    @staticmethod
    def get_shear_stirrups_share() -> float:
        """Minimum share of shear action that has to be taken by stirrups."""
        return 0.5

    @classmethod
    def check_section(cls, section: RectangularSection) -> bool:
        """Check if the section meets the minimum requirements for the detailing code."""
        return True


class BeamDetailsDM76:
    _rebar_min_ratio = {
        RebarsType.DEFORMED: 0.0015,
        RebarsType.PLAIN: 0.0025
    }   # DM 76 Par 2.17

    @staticmethod
    def get_min_rebar_diameter() -> int:
        """
        Minimum diameter of the main reinforcement bars in mm.

        :return: diameter in mm.
        """
        return 12

    @staticmethod
    def get_shear_stirrups_share() -> float:
        """
        Minimum share of shear action that has to be taken by stirrups. Par 2.4.1

        :return: share as ratio
        """
        return 0.4

    @classmethod
    def get_min_long_rebar_area(
        cls,
        section_area: float,
        rebar_type: RebarsType = RebarsType.DEFORMED,
    ) -> float:
        """
        Minimum bottom reinforcement area in m² based on DM 76 Par 2.17.

        :return: area in m².
        """
        return cls._rebar_min_ratio[rebar_type] * section_area

    @staticmethod
    def get_max_stirrups_spacing(
        section_depth: float
    ) -> float:
        """
        Maximum spacing between stirrups in meters based on DM 76 Par 2.4.1.

        :param section_depth: Effective depth of the section in meters.
        :return: Maximum spacing between stirrups in meters.
        """
        return min(
            1 / 3,  # 3 stirrups per meter (0.333... m)
            0.8 * section_depth # 80% of the depth
        )

    @staticmethod
    def get_min_stirrups_area(
    ) -> float:
        """
        Minimum area of stirrups per meter in m²/m.
        As per DM 76 Par 2.4.1, the minimum area of stirrups is 3 cm²/m (0.0003 m²/m).

        :return: Minimum area per meter.
        """
        return 0.0003 # 3 cm²/m par 2.17

    @classmethod
    def check_section(
        cls,
        section: RectangularSectionElement,
        rebar_type: RebarsType
    ) -> bool:
        """
        Check if the section meets the minimum requirements for the detailing code.

        :param section: The section to be checked.
        :param rebar_type: Type of the reinforcement bars (Deformed or Plain).
        :return: True if the section meets the requirements, False otherwise.
        """
        # Check minimum longitudinal reinforcement diameter
        if min(section.bot_reinf_d, section.top_reinf_d) < cls.get_min_rebar_diameter():
            return False

        # Check minimum reinforcement area (bottom and top)
        min_area = cls.get_min_long_rebar_area(section_area=section.area, rebar_type=rebar_type)
        if section.bot_reinf_area < min_area or section.top_reinf_area < min_area:
            return False

        # Check maximum stirrups spacing
        if section.stirrups_spacing > cls.get_max_stirrups_spacing(section_depth=section.h - section.cop):
            return False

        # Check minimum stirrups area per meter
        if section.stirrups_spacing == 0:
            return False  # Avoid division by zero
        if (section.stirrups_reinf_area / section.stirrups_spacing) < cls.get_min_stirrups_area():
            return False

        return True


class ColumnDetailsRD39:

    @staticmethod
    def get_min_rebar_diameter() -> int:
        """
        Minimum diameter of the main reinforcement bars in mm.

        :return: diameter in mm.
        """
        return 12

    @staticmethod
    def compute_min_long_reinf_area(
        column_min_cls_area: float
    ) -> float:
        """
        Computes the minimum longitudinal reinforcement area for a column based on its area and minimum class area.
        The formula is based on the ratio of the column area to the minimum class area, ensuring that the reinforcement

        :param column_min_cls_area: minimum strictly necessary area in m².
        :return: minimum longitudinal reinforcement area in m².
        """
        ratio = max(
            min(
                0.008,
                0.008 + (0.005 - 0.008) * (column_min_cls_area - 0.2) / (0.8 - 0.2)
            ),
            0.005
        )
        return column_min_cls_area * ratio

    @staticmethod
    def get_max_stirrups_spacing(
        long_bar_d: float,
        min_dim: float
    ) -> float:
        """
        Maximum spacing between transversal reinforcement bars in meters.

        :param long_bar_d: longitudinal bar diameter
        :param min_dim: minimum dimension of the section in meters
        :return: minimum spacing in meters
        """
        return min(
            0.5 * min_dim,
            10 * long_bar_d
        )

    @classmethod
    def check_section(
        cls,
        section: RectangularSectionElement,
        min_cls_area: float
    ) -> bool:
        """
        Check if the section meets the minimum requirements for the detailing code.

        :param section: The section to be checked.
        :param min_cls_area: Minimum cls area in m².
        :return: True if the section meets the requirements, False otherwise.
        """
        # Check minimum longitudinal bar diameter
        if min(section.bot_reinf_d, section.top_reinf_d) < cls.get_min_rebar_diameter():
            return False

        # Check minimum longitudinal reinforcement area
        min_reinf_area = cls.compute_min_long_reinf_area(column_min_cls_area=min_cls_area)
        if (section.bot_reinf_area + section.top_reinf_area) < min_reinf_area:
            return False

        # Check maximum stirrup spacing
        max_spacing = cls.get_max_stirrups_spacing(
            long_bar_d=section.bot_reinf_d,
            min_dim=min(section.b, section.h)
        )
        if section.stirrups_spacing > max_spacing:
            return False

        return True


class ColumnDetailsDM76:
    _rebar_adm_stress = {
        RebarsType.DEFORMED: 180_000,
        RebarsType.PLAIN: 120_000
    }

    @staticmethod
    def get_min_rebar_diameter() -> int:
        """
        Minimum diameter of the main reinforcement bars in mm.
        DM 76 Par 2.12.

        :return: diameter in mm.
        """
        return 12

    @staticmethod
    def compute_min_long_reinf_area(
        column_area: float,
        column_min_cls_area: float
    ) -> float:
        """
        Computes the minimum longitudinal reinforcement area for a column based on its area and minimum class area.
        The formula is based on the ratio of the column area to the minimum class area, ensuring that the reinforcement
        DM 76 Par 2.12.

        :param column_area: column area in m².
        :param column_min_cls_area: minimum strictly necessary area in m².
        :return: minimum longitudinal reinforcement area in m².
        """
        return max(
            0.006 * column_min_cls_area,
            0.003 * column_area
        )

    @staticmethod
    def compute_max_long_reinf_area(
        column_area: float
    ) -> float:
        """
        Computes the maximum longitudinal reinforcement area for a column based on its area and minimum class area.
        The formula is based on the ratio of the column area to the minimum class area, ensuring that the reinforcement
        DM 76 Par 2.12.

        :param column_area: column area in m².
        :return: maximum longitudinal reinforcement area in m².
        """
        return 0.05 * column_area

    @classmethod
    def get_rebar_adm_stress(
        cls,
        rebar_type: RebarsType = RebarsType.DEFORMED
    ) -> float:
        """
        Get the allowable stress for the reinforcement bars based on their type.
        DM 76 Par 2.12.

        :param rebar_type: Type of the reinforcement bars (Deformed or Plain).
        :return: Allowable stress in kPa.
        """
        return cls._rebar_adm_stress[rebar_type]

    @staticmethod
    def get_max_stirrups_spacing(
        long_bar_d: float,
    ) -> float:
        """
        Minimum spacing between stirrups in meters based on DM 76 Par 2.12.

        :param long_bar_d: Diameter of the longitudinal bars in mm.
        :return: Minimum spacing between stirrups in meters.
        """
        return min(
            15 * long_bar_d,
            .25
        )

    @classmethod
    def check_section(
        cls,
        section: RectangularSectionElement,
        column_min_cls_area: float
    ) -> bool:
        """
        Check if the section meets the minimum requirements for the detailing code.

        :param section: The section to be checked.
        :param column_min_cls_area: Minimum class area in m².
        :return: True if the section meets the requirements, False otherwise.
        """
        # Check minimum longitudinal bar diameter
        if min(section.bot_reinf_d, section.top_reinf_d) < cls.get_min_rebar_diameter():
            # Failed due to minimum bar diameter
            return False

        # Check minimum longitudinal reinforcement area
        min_reinf_area = cls.compute_min_long_reinf_area(
            column_area=section.area,
            column_min_cls_area=column_min_cls_area
        )
        if section.bot_reinf_area + section.top_reinf_area < min_reinf_area:
            # Failed due to minimum reinforcement area
            return False

        # Check maximum longitudinal reinforcement area
        max_reinf_area = cls.compute_max_long_reinf_area(column_area=section.area)
        if section.bot_reinf_area + section.top_reinf_area > max_reinf_area:
            # Failed due to maximum reinforcement area
            return False

        # Check minimum stirrup spacing
        max_spacing = cls.get_max_stirrups_spacing(long_bar_d=section.bot_reinf_d)
        if section.stirrups_spacing > max_spacing:
            # Failed due to minimum stirrup spacing
            return False

        # All checks passed
        return True

# Get functions
min_reinf_diameter_beams: dict[DetailingCode, int] = {
    DetailingCode.RD_39: 0,
    DetailingCode.DM_76: BeamDetailsDM76.get_min_rebar_diameter()
}

min_reinf_diameter_columns: dict[DetailingCode, int] = {
    DetailingCode.RD_39: ColumnDetailsRD39.get_min_rebar_diameter(),
    DetailingCode.DM_76: ColumnDetailsDM76.get_min_rebar_diameter()
}


def get_min_longitudinal_bar_area_column(
    column_area: float,
    column_min_cls_area: float,
    detailing_code: DetailingCode
) -> float:
    """
    Computes the minimum longitudinal reinforcement area for a column based on the provided detailing code.
    The calculation ensures that the reinforcement meets the requirements specified by the detailing code.

    For RD_39, the calculation is based on the minimum strictly necessary area.
    For DM_76, the calculation considers both the column area and the minimum strictly necessary area.

    :param column_area: Total column cross-sectional area in m².
    :param column_min_cls_area: Minimum strictly necessary cross-sectional area in m².
    :param detailing_code: Detailing code to be used for calculations (RD_39 or DM_76).
    :return: Minimum longitudinal reinforcement area in m².
    :raises ValueError: If an invalid detailing code is provided.
    """
    if detailing_code == DetailingCode.RD_39:
        return ColumnDetailsRD39.compute_min_long_reinf_area(
            column_min_cls_area
        )
    elif detailing_code == DetailingCode.DM_76:
        return ColumnDetailsDM76.compute_min_long_reinf_area(
            column_area,
            column_min_cls_area
        )
    else:
        raise ValueError('Invalid detailing code provided.')


def get_max_stirrup_spacing_column(
    long_bar_d: float,
    min_dim: float,
    detailing_code: DetailingCode
) -> float:
    """
    Computes the maximum stirrup spacing for a column based on the provided detailing code.
    The calculation ensures that the stirrup spacing meets the requirements specified by the detailing code.
    :param long_bar_d: Diameter of the longitudinal bars in mm.
    :param min_dim: Minimum dimension of the column section in meters.
    :param detailing_code: Detailing code to be used for calculations (RD_39 or DM_76).
    :return: Maximum stirrup spacing in meters.
    """
    if detailing_code == DetailingCode.RD_39:
        return ColumnDetailsRD39.get_max_stirrups_spacing(
            long_bar_d=long_bar_d,
            min_dim=min_dim
        )
    elif detailing_code == DetailingCode.DM_76:
        return ColumnDetailsDM76.get_max_stirrups_spacing(
            long_bar_d=long_bar_d,
        )
    else:
        raise ValueError('Invalid detailing code provided.')


def get_min_longitudinal_bar_area_beam(
    section_area: float,
    detailing_code: DetailingCode,
    bar_type: RebarsType
) -> float:
    """
    Computes the minimum longitudinal reinforcement area for a beam based on the provided detailing code.
    The calculation ensures that the reinforcement meets the requirements specified by the detailing code.

    :return: Minimum longitudinal reinforcement area in m².
    """
    if detailing_code == DetailingCode.RD_39:
        return 0
    elif detailing_code == DetailingCode.DM_76:
        return BeamDetailsDM76.get_min_long_rebar_area(
            section_area=section_area,
            rebar_type=bar_type
        )
    else:
        raise ValueError('Invalid detailing code provided.')


def get_max_stirrup_spacing_beam(
    section_depth: float,
    rebar_d: int,
    detailing_code: DetailingCode
) -> float:
    """
    Computes the max stirrup spacing for a beam based on the provided detailing code.
    The calculation ensures that the stirrup spacing meets the requirements specified by the detailing code.
    :param section_depth: Effective depth of the beam section in meters.
    :param detailing_code: Detailing code to be used for calculations (RD_39 or DM_76).
    :return: Minimum stirrup spacing in meters.
    :raises ValueError: If an invalid detailing code is provided.
    """
    if detailing_code == DetailingCode.RD_39:
        return .33
    elif detailing_code == DetailingCode.DM_76:
        max_spacing = BeamDetailsDM76.get_max_stirrups_spacing(
            section_depth
        )
        spacing_min_area = BeamDetailsDM76.get_min_stirrups_area()
        return min(
            max_spacing,
            reinforcement_area(2, rebar_d) * mmq_mq / spacing_min_area
        )
    else:
        raise ValueError('Invalid detailing code provided.')




# Checkers
def column_section_detail_checker(
    section: RectangularSectionElement,
    detailing_code: DetailingCode,
    min_cls_area: float
) -> bool:
    """
    Checks if the section meets the detailing requirements based on the provided detailing code.

    :param section: The section to be checked.
    :param detailing_code: The detailing code to be used for checking.
    :return: True if the section meets the detailing requirements, False otherwise.
    """
    if detailing_code == DetailingCode.RD_39:
        return ColumnDetailsRD39.check_section(
            section=section,
            min_cls_area=min_cls_area
        )
    elif detailing_code == DetailingCode.DM_76:
        return ColumnDetailsDM76.check_section(
            section=section,
            column_min_cls_area=min_cls_area
        )
    else:
        raise ValueError('Invalid detailing code provided.')


def beam_section_detail_checker(
    section: RectangularSectionElement,
    detailing_code: DetailingCode,
    rebar_type: RebarsType
) -> bool:
    """
    Checks if the section meets the detailing requirements based on the provided detailing code.
    :param section: The section to be checked.
    :param detailing_code: The detailing code to be used for checking.
    :param rebar_type: Type of the reinforcement bars (Deformed or Plain).
    :return: True if the section meets the detailing requirements, False otherwise.
    """
    if detailing_code == DetailingCode.RD_39:
        return BeamDetailsRD39.check_section(
            section=section
        )
    elif detailing_code == DetailingCode.DM_76:
        return BeamDetailsDM76.check_section(
            section=section,
            rebar_type=rebar_type
        )
    else:
        raise ValueError('Invalid detailing code provided.')
