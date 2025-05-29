from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from src.section_design.detailing_minimums import (
    DetailingCode, RebarsType, beam_section_detail_checker,
    get_max_stirrup_spacing_beam, get_min_longitudinal_bar_area_beam,
    min_reinf_diameter_beams)
from src.section_design.section_geometry import (RectangularSection,
                                                 RectangularSectionElement,
                                                 SectionGeometry)
from src.section_design.section_utils import (ReinforcementCombination,
                                              reinforcement_area)
from src.section_design.shear_design import ShearSectionDesignStirrups
from src.section_design.stress_functions import (compute_alpha_bottom_reinf,
                                                 compute_beta_bottom_reinf,
                                                 compute_neutral_axis_ratio)
from src.sollicitations import MemberSollicitation
from src.utils import round_to_nearest

# Unit conversion constants
mmq_mq = 1e-6   # mm^2 to m^2
mq_cmq = 1e4    # m^2 to cm^2
cmq_mq = 1e-4   # m^2 to cm^2


# Predimensioning of the bottom reinforcement
def unconditioned_design_bottom_reinf(
    M: float,
    b: float,
    sigma_cls_adm: float,
    sigma_s_adm: float,
    n: int = 15
) -> tuple[float, float]:
    """
    Calculates the unconditioned design of bottom reinforcement in a rectangular beam section.

    :param M: Applied bending moment [kNm].
    :param b: Width of the beam section [m].
    :param sigma_cls_adm: Allowable stress in the concrete [kPa].
    :param sigma_s_adm: Allowable stress in the steel reinforcement [kPa].
    :param n: Modular ratio, defaults to 15.
    :return: A tuple containing the effective depth [m] and area of reinforcement required [m²].
    """
    # Design parameters
    k = compute_neutral_axis_ratio(sigma_cls_adm, sigma_s_adm, n)
    alpha = compute_alpha_bottom_reinf(k, sigma_cls_adm)
    beta = compute_beta_bottom_reinf(alpha, k, sigma_s_adm)

    # Dimensioning
    d = alpha * np.sqrt(M / b)
    As = beta * np.sqrt(M * b)

    return d, As


# Design routines for section
class SkwBeamSectionDesign:
    """
    A class for designing a skew beam section under positive and negative bending
    moments using reinforced concrete design principles.

    :param section_geometry: Geometry of the beam section (height, width, and concrete cover).
    :param positive_design_sollicitations: Bending moment for the positive design case.
    :param negative_design_sollicitations: Bending moment for the negative design case.
    """

    def __init__(
        self,
        section_geometry: SectionGeometry,
        positive_design_sollicitations: MemberSollicitation,
        negative_design_sollicitations: MemberSollicitation
    ):
        self.__section_geometry = section_geometry
        self.__positive_design_sollicitations = positive_design_sollicitations
        self.__negative_design_sollicitations = negative_design_sollicitations

    @staticmethod
    def _compute_steel_stress(
        distance: float,
        neutral_axis: float,
        sigma_cls: float,
        n: float = 15.
    ) -> float:
        """
        Computes the stress in the steel reinforcement based on its distance to the neutral axis.

        :param distance: Distance from the reinforcement to the neutral axis [m].
        :param neutral_axis: Position of the neutral axis from the reference edge [m].
        :param sigma_cls: Compressive stress at the top fiber of the concrete [kPa].
        :param n: Modular ratio (Es/Ec), defaults to 15.
        :return: Stress in the reinforcing steel [kPa].
        """
        return n * sigma_cls * distance / neutral_axis

    def compute_minimum_steel_area(
        self,
        sigma_adm_cls: float,
        sigma_adm_steel: float,
        n: float = 15.,
        As_pred: float = 1e-4
    ) -> tuple[float, float, bool]:
        """
        Computes the minimum required steel reinforcement area to resist positive and negative
        bending moments using nonlinear constrained optimization.

        :param sigma_adm_cls: Maximum allowable compressive stress in concrete [kPa].
        :param sigma_adm_steel: Maximum allowable tensile stress in steel [kPa].
        :param n: Modular ratio (Es/Ec), defaults to 15.
        :param As_pred: Initial guess for steel area [m²], defaults to 1e-4.
        :return: Tuple containing minimum required steel area [m²] and the reinforcement ratio (top/bottom).
        """
        # Collect section geometry
        h = self.__section_geometry.h
        b = self.__section_geometry.b
        cop = self.__section_geometry.cop

        M_pos = self.__positive_design_sollicitations.M
        M_neg = self.__negative_design_sollicitations.M

        # Useful parameters
        d = h - cop
        d_d = d - cop

        # Objective function
        def obj_function(x) -> float:
            # Setup
            As, As_ratio, *_ = x
            # Obj is squared to improve convergence
            return (As * (1 + As_ratio))**2

        # Constraints
        def moment_balance_top_reinforcement_constraint_pos(x) -> float:
            # Setup
            As, _, y_pos, _, sigma_cls_pos, _ = x
            As *= cmq_mq
            sigma_cls_pos *= sigma_adm_cls
            # Compute
            sigma_s = self._compute_steel_stress(d - y_pos, y_pos, sigma_cls_pos, n)
            steel_contribution = As * sigma_s * d_d
            concrete_contribution = 0.5 * b * y_pos * sigma_cls_pos * (cop - y_pos / 3)
            # Return the difference between the contributions and the moment
            moment_balance = (steel_contribution + concrete_contribution - M_pos)
            return moment_balance / M_pos   # Normalized to  to improve convergence

        def moment_balance_bot_reinforcement_constraint_pos(x) -> float:
            # Setup
            As, As_ratio, y_pos, _, sigma_cls_pos, _ = x
            As *= cmq_mq
            sigma_cls_pos *= sigma_adm_cls
            # Compute
            sigma_s = self._compute_steel_stress(y_pos - cop, y_pos, sigma_cls_pos, n)
            steel_contribution = As * As_ratio * sigma_s * d_d
            concrete_contribution = 0.5 * b * y_pos * sigma_cls_pos * (d - y_pos / 3)
            # Return the difference between the contributions and the moment
            moment_balance = steel_contribution + concrete_contribution - M_pos
            return moment_balance / M_pos   # Normalized to  to improve convergence

        def moment_balance_top_reinforcement_constraint_neg(x) -> float:
            # Setup
            As, As_ratio, _, y_neg, _, sigma_cls_neg = x
            As *= cmq_mq
            sigma_cls_neg *= sigma_adm_cls
            # Compute
            sigma_s = self._compute_steel_stress(d - y_neg, y_neg, sigma_cls_neg, n)
            steel_contribution = As * As_ratio * sigma_s * d_d
            concrete_contribution = 0.5 * b * y_neg * sigma_cls_neg * (cop - y_neg / 3)
            # Return the difference between the contributions and the moment
            moment_balance = steel_contribution + concrete_contribution - M_neg
            return moment_balance / M_pos   # Normalized to  to improve convergence, M_neg can be 0

        def moment_balance_bot_reinforcement_constraint_neg(x) -> float:
            # Setup
            As, _, _, y_neg, _, sigma_cls_neg = x
            As *= cmq_mq
            sigma_cls_neg *= sigma_adm_cls
            # Compute
            sigma_s = self._compute_steel_stress(y_neg - cop, y_neg, sigma_cls_neg, n)
            steel_contribution = As * sigma_s * d_d
            concrete_contribution = 0.5 * b * y_neg * sigma_cls_neg * (d - y_neg / 3)
            # Return the difference between the contributions and the moment
            moment_balance = steel_contribution + concrete_contribution - M_neg
            return moment_balance / M_pos   # Normalized to  to improve convergence, M_neg can be 0


        def maximum_adm_steel_stress_top_constraint_pos(x) -> float:
            # Setup
            _, _, y_pos, _, sigma_cls_pos, _ = x
            sigma_cls_pos *= sigma_adm_cls
            # Compute
            sigma_s = self._compute_steel_stress(d - y_pos, y_pos, sigma_cls_pos, n)
            # Return the difference between the maximum allowable stress and the computed stress
            sigma_balance = sigma_adm_steel - sigma_s
            return sigma_balance / sigma_adm_steel  # Normalized to  to improve convergence

        def maximum_adm_steel_stress_bot_constraint_pos(x) -> float:
            # Setup
            _, _, y_pos, _, sigma_cls_pos, _ = x
            sigma_cls_pos *= sigma_adm_cls
            # Compute
            sigma_s = self._compute_steel_stress(y_pos - cop, y_pos, sigma_cls_pos, n)
            # Return the difference between the maximum allowable stress and the computed stress
            sigma_balance = sigma_adm_steel - sigma_s
            return sigma_balance / sigma_adm_steel  # Normalized to  to improve convergence

        def maximum_adm_steel_stress_top_constraint_neg(x) -> float:
            # Setup
            _, _, _, y_neg, _, sigma_cls_neg = x
            sigma_cls_neg *= sigma_adm_cls
            # Compute
            sigma_s = self._compute_steel_stress(d - y_neg, y_neg, sigma_cls_neg, n)
            # Return the difference between the maximum allowable stress and the computed stress
            sigma_balance = sigma_adm_steel - sigma_s
            return sigma_balance / sigma_adm_steel  # Normalized to  to improve convergence

        def maximum_adm_steel_stress_bot_constraint_neg(x) -> float:
            # Setup
            _, _, _, y_neg, _, sigma_cls_neg = x
            sigma_cls_neg *= sigma_adm_cls
            # Compute
            sigma_s = self._compute_steel_stress(y_neg - cop, y_neg, sigma_cls_neg, n)
            # Return the difference between the maximum allowable stress and the computed stress
            sigma_balance = sigma_adm_steel - sigma_s
            return sigma_balance / sigma_adm_steel  # Normalized to  to improve convergence

        # Initial guess
        initial_guess = [
            As_pred / mq_cmq,   # Bottom reinforcement area in cmq
            M_neg / M_pos,      # Ratio of top to bottom reinforcement area
            h / 3,              # Depth of the neutral axis for positive moment
            h / 3,              # Depth of the neutral axis for negative moment
            0.8,                # Ratio of top concrete stress to allowable concrete stress
            0.8                 # Ratio of bottom concrete stress to allowable concrete stress
        ]

        # Bounds
        bounds = [
            [0, None],          # Bottom reinforcement area in cmq
            [0, None],          # Ratio of top to bottom reinforcement area
            [0, h],             # Depth of the neutral axis for positive moment
            [0, h],             # Depth of the neutral axis for negative moment
            [0, 1],             # Ratio of top concrete stress to allowable concrete stress
            [0, 1]              # Ratio of bottom concrete stress to allowable concrete stress
        ]

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': moment_balance_bot_reinforcement_constraint_pos},
            {'type': 'eq', 'fun': moment_balance_top_reinforcement_constraint_pos},
            {'type': 'eq', 'fun': moment_balance_bot_reinforcement_constraint_neg},
            {'type': 'eq', 'fun': moment_balance_top_reinforcement_constraint_neg},
            {'type': 'ineq', 'fun': maximum_adm_steel_stress_bot_constraint_pos},
            {'type': 'ineq', 'fun': maximum_adm_steel_stress_top_constraint_pos},
            {'type': 'ineq', 'fun': maximum_adm_steel_stress_bot_constraint_neg},
            {'type': 'ineq', 'fun': maximum_adm_steel_stress_top_constraint_neg}
        ]

        # Solve optimization
        result = minimize(
            obj_function,
            initial_guess,
            method='trust-constr',  # SLSQP is not working for some reason
            bounds=bounds,
            constraints=constraints
        )

        As, As_ratio, *_ = result.x
        As *= cmq_mq  # Convert to m²
        return float(As), float(As_ratio), result.success


# Verification of the beam section
@dataclass
class VerifyRectangularBeam:
    """
    A class for verifying the structural integrity of a rectangular beam section.

    :param h: Height of the rectangular beam section [m].
    :param b: Width of the rectangular beam section [m].
    :param As_top: Area of the top steel reinforcement [m²].
    :param As_bot: Area of the bottom steel reinforcement [m²].
    :param cop: Distance from the top fiber to the centroid of the top reinforcement (cover of concrete) [m].
    """

    h: float
    b: float
    As_top: float
    As_bot: float
    cop: float

    def verify_section(
        self,
        sigma_cls_adm: float,
        sigma_s_adm: float,
        M: float,
        n: float = 15
    ) -> tuple[float, float]:
        """
        Verifies the beam section by calculating the maximum concrete and steel stresses.

        :param sigma_cls_adm: Allowable concrete stress [kPa].
        :param sigma_s_adm: Allowable steel stress [kPa].
        :param M: Applied moment on the section [kNm].
        :param n: Modular ratio (Es/Ec), defaults to 15.
        :return: A tuple containing:
            - The ratio of computed concrete stress to allowable stress.
            - The maximum ratio of steel stress to allowable stress.
        """
        b = self.b
        As_s = [self.As_top, self.As_bot]
        d_s = [self.cop, self.h - self.cop]

        # Solve quadratic equation for neutral axis depth (y)
        y = max(np.roots([
            .5 * b / n,
            sum(As_s),
            -sum(As * d for As, d in zip(As_s, d_s))
        ]))

        # Compute maximum stresses in concrete and steel
        sigma_cls = M * y / (n * sum(As * (d - y / 3) * (d - y) for As, d in zip(As_s, d_s)))
        sigma_steel = [np.abs(n * sigma_cls * (d - y) / y) for d in d_s]

        # Check if stresses exceed allowable limits
        check_cls = sigma_cls / sigma_cls_adm
        check_steel = [abs(sigma_s) / sigma_s_adm for sigma_s in sigma_steel]

        return check_cls, max(check_steel)


def design_beam_section(
        M_pos: float,
        M_neg: float,
        sigma_cls_adm: float,
        sigma_s_adm: float,
        detailing_code: DetailingCode,
        rebar_type: RebarsType,
        b_min: float = 0.3,
        cop: float = 0.03,
        n: float = 15,
        min_rebar_d: int = 12,
        max_rebar_count: int = 6,
        min_h: float = 0.4,
        section_geometry: SectionGeometry | None = None
    ) -> RectangularSection:
    """
    Designs a rectangular beam section based on positive and negative bending moments,
    allowable concrete stress, allowable steel stress, and optional section geometry.

    If no section geometry is provided, a default rectangular section is designed,
    reinforcement areas are calculated, and the beam is checked for stress compliance.

    :param M_pos: Positive bending moment in kNm
    :param M_neg: Negative bending moment in kNm
    :param sigma_cls_adm: Allowable compressive stress of concrete in kPa
    :param sigma_s_adm: Allowable tensile stress of steel in kPa
    :param section_geometry: Optional predefined section geometry, defaults to None
    :param b_min: Minimum width of the section in m, defaults to 0.3 m
    :param cop: Cover of concrete in m, defaults to 0.03 m
    :param n: Modular ratio (Es/Ec), defaults to 15
    :param min_As: Minimum required steel area in m², defaults to 5e-4 m²
    :param min_rebar_d: Minimum diameter of reinforcement bars in mm, defaults to 12 mm
    :param detailing_code: Detailing code to be used for the design
    :param rebar_type: Type of reinforcement bars to be used
    :param min_h: Minimum height of the section in m, defaults to 0.4 m
    :param max_rebar_count: Maximum number of reinforcement bars, defaults to 6
    :param section_geometry: Optional predefined section geometry, defaults to None
    :raises ValueError: Raised if no valid reinforcement solution is found for top or bottom reinforcement
    :return: Designed rectangular section with specified reinforcement
    """
    As_initial = None

    # If no predefined section is given, a new section will be designed
    if section_geometry is None:
        # Calculate depth and steel area for both positive and negative moments
        d_pos, As_pos = unconditioned_design_bottom_reinf(
            M=M_pos,
            b=b_min,
            sigma_cls_adm=sigma_cls_adm,
            sigma_s_adm=sigma_s_adm
        )
        d_neg, As_neg = unconditioned_design_bottom_reinf(
            M=M_neg,
            b=b_min,
            sigma_cls_adm=sigma_cls_adm,
            sigma_s_adm=sigma_s_adm
        )

        # Use the larger depth for the final design
        d_design = max(d_pos, d_neg)
        As_initial = max(As_pos, As_neg)

        # Round section height to nearest tolerance
        h_design = round_to_nearest(
            max(d_design + cop, min_h),
            tol=.05
        )

        # Define section geometry based on calculated height and default width
        section_geometry = SectionGeometry(
            h_design,
            b_min,
            cop
        )

    As_min = get_min_longitudinal_bar_area_beam(
        section_area=section_geometry.area,
        detailing_code=detailing_code,
        bar_type=rebar_type
    )
    if As_initial is None:
        # If no initial steel area is provided, use the minimum required area
        As_initial = As_min

    # Define action data for positive and negative moments
    section_actions_pos = MemberSollicitation(
        M=M_pos
    )
    section_actions_neg = MemberSollicitation(
        M=M_neg
    )

    # Design the required steel area for the section
    section_design = SkwBeamSectionDesign(
        section_geometry,
        section_actions_pos,
        section_actions_neg
    )

    # Compute minimum steel area required
    As, As_ratio, _ = section_design.compute_minimum_steel_area(
        sigma_adm_cls=sigma_cls_adm,
        sigma_adm_steel=sigma_s_adm,
        n=n,
        As_pred=As_initial
    )

    # Calculate required bottom and top reinforcement areas
    As_bot_design = max(As * As_ratio, As_min)
    As_top_design = max(As, As_min)

    # Create a reinforcement selector object for finding bar combinations
    reinf_selector = ReinforcementCombination(
        section_width=section_geometry.b,
        min_diameter=min_rebar_d,
        max_count=max_rebar_count
    )

    section_check_passed: bool = False
    while not section_check_passed:
        # Find the best bar combination for top and bottom reinforcement
        top_reinf = reinf_selector.find_combination(As_top_design)
        bot_reinf = reinf_selector.find_combination(As_bot_design)

        # Raise error if no valid reinforcement solution is found
        if top_reinf is None:
            raise ValueError(f"Warning, no solution was found with top reinf: {As_top_design * 10000:.2f} cmq")
        if bot_reinf is None:
            raise ValueError(f"Warning, no solution was found with bottom reinf: {As_bot_design * 10000:.2f} cmq")

        # Calculate actual reinforcement areas
        As_top = reinforcement_area(*top_reinf) * mmq_mq
        As_bot = reinforcement_area(*bot_reinf) * mmq_mq

        # Verify the section for positive moment
        section_check = VerifyRectangularBeam(
            h=section_geometry.h,
            b=section_geometry.b,
            As_top=As_bot,
            As_bot=As_top,
            cop=section_geometry.cop
        )
        pos_moment_check = section_check.verify_section(
            sigma_cls_adm=sigma_cls_adm,
            sigma_s_adm=sigma_s_adm,
            M=section_actions_pos.M
        )

        # Verify the section for negative moment
        section_check = VerifyRectangularBeam(
            h=section_geometry.h,
            b=section_geometry.b,
            As_top=As_top,
            As_bot=As_bot,
            cop=section_geometry.cop
        )
        neg_moment_check = section_check.verify_section(
            sigma_cls_adm=sigma_cls_adm,
            sigma_s_adm=sigma_s_adm,
            M=section_actions_neg.M
        )

        # Check if both positive and negative moment checks pass
        section_check_passed = all(check <= 1 for check in (pos_moment_check + neg_moment_check))

        # If section does not pass, increment the steel area and check again
        if not section_check_passed:
            As_top_design = As_top + 1e-8
            As_bot_design = As_bot + 1e-8

    # At least one loop cycle will be completed, get final top and bottom reinforcement
    top_reinf_count, top_reinf_d = top_reinf    # type: ignore[possiblyUnboundVariable] must be initialized in while loop
    bot_reinf_count, bot_reinf_d = bot_reinf    # type: ignore[possiblyUnboundVariable] must be initialized in while loop

    # Return the designed rectangular section with calculated reinforcement
    return RectangularSection(
        **section_geometry.__dict__,
        top_reinf_count=top_reinf_count,
        top_reinf_d=top_reinf_d,
        bot_reinf_count=bot_reinf_count,
        bot_reinf_d=bot_reinf_d
    )


def design_beam_element(
        Mpos: float,
        Mneg: float,
        Vmax: float,
        sigma_cls_adm: float,
        sigma_s_adm: float,
        detailing_code: DetailingCode,
        rebar_type: RebarsType,
        cop: float = 0.03,
        n: float = 15,
        shear_rebar_diameter: int = 6,
        b_section: float = 0.3,
        min_h: float = 0.4
) -> RectangularSectionElement:
    """
    Designs a beam element based on positive and negative bending moments, shear force,
    allowable concrete stress, allowable steel stress, detailing code, and optional parameters.

    :param Mpos: Positive bending moment in kNm
    :param Mneg: Negative bending moment in kNm
    :param Vmax: Maximum shear force in kN
    :param sigma_cls_adm: Allowable compressive stress of concrete in kPa
    :param sigma_s_adm: Allowable tensile stress of steel in kPa
    :param detailing_code: Detailing code for the design
    :param cop: Cover of concrete in m, defaults to 0.05 m
    :param shear_rebar_diameter: Diameter of shear reinforcement in mm, defaults to 6 mm
    :param n: Modular ratio (Es/Ec), defaults to 15
    :param b_section: Width of the beam section in m, defaults to 0.4 m
    :return: Designed rectangular section with specified reinforcement
    """
    # Minimum width
    min_rebar_diameter = min_reinf_diameter_beams.get(detailing_code, 0)  # mm

    # Design the beam section based on the provided parameters
    section = design_beam_section(
        M_pos=Mpos,
        M_neg=Mneg,
        sigma_cls_adm=sigma_cls_adm,
        sigma_s_adm=sigma_s_adm,
        detailing_code=detailing_code,
        rebar_type=rebar_type,
        b_min=b_section,
        cop=cop,
        n=n,
        min_rebar_d=min_rebar_diameter,
        max_rebar_count=6,
        min_h=min_h
    )

    # Calculate shear reinforcement area
    shear_design_cross = ShearSectionDesignStirrups(
        h=section.h,
        b=section.b,
        cop=section.cop
    )
    max_spacing = get_max_stirrup_spacing_beam(
        section_depth=section.h - section.cop,
        rebar_d=shear_rebar_diameter,
        detailing_code=detailing_code
    )
    max_spacing_stirrups = shear_design_cross.design_reinforcement(
        V_striups=Vmax,
        sigma_s_adm=sigma_s_adm,
        diameter=shear_rebar_diameter,
        max_spacing=max_spacing
    )

    # Check section
    section_element = RectangularSectionElement(
        **section.__dict__,
        stirrups_reinf_d=shear_rebar_diameter,
        stirrups_spacing=max_spacing_stirrups
    )

    beam_check = beam_section_detail_checker(
        detailing_code=detailing_code,
        section=section_element,
        rebar_type=rebar_type
    )

    assert beam_check, 'Beam section does not pass detailing code check'

    # Return the designed rectangular section element with calculated reinforcement
    return section_element
