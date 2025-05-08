from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from src.section_design.section_geometry import (RectangularSection,
                                                 SectionGeometry)
from src.section_design.section_utils import (ReinforcementCombination,
                                              reinforcement_area)
from src.section_design.stress_functions import (compute_alpha_bottom_reinf,
                                                 compute_beta_bottom_reinf,
                                                 compute_neutral_axis_ratio)
from src.sollicitations import MemberSollicitation
from src.units_conversion import mmq_mq
from src.utils import round_to_nearest


def unconditioned_design_bottom_reinf(
        M: float,
        b: float,
        sigma_cls_adm: float,
        sigma_s_adm: float,
        n: int = 15
) -> tuple[float, float]:
    """
    Calculates the unconditioned design of bottom reinforcement in a rectangular beam section.

    Parameters
    ----------
    M : float
        Applied bending moment.
    b : float
        Width of the beam section.
    sigma_cls_adm : float
        Allowable stress in the concrete.
    sigma_s_adm : float
        Allowable stress in the steel reinforcement.
    n : int, optional
        Modular ratio, by default 15.

    Returns
    -------
    Tuple[float, float]
        The effective depth and area of reinforcement required.
    """
    # Design parameters
    k = compute_neutral_axis_ratio(sigma_cls_adm, sigma_s_adm, n)
    alpha = compute_alpha_bottom_reinf(k, sigma_cls_adm)
    beta = compute_beta_bottom_reinf(alpha, k, sigma_s_adm)

    # Dimensioning
    d = alpha * np.sqrt(M / b)
    As = beta * np.sqrt(M * b)

    return d, As


class SkwBeamSectionDesign:
    """
    A class for designing a skew beam section under positive and negative bending
    moments using reinforced concrete design principles.

    :param section_geometry: Geometry of the beam section (height, width, and concrete cover).
    :type section_geometry: SectionGeometry
    :param positive_design_sollicitations: Bending moment for the positive design case.
    :type positive_design_sollicitations: MemberSollicitations
    :param negative_design_sollicitations: Bending moment for the negative design case.
    :type negative_design_sollicitations: MemberSollicitations
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
    def compute_steel_stress(
        distance: float,
        neutral_axis: float,
        sigma_cls: float,
        n: float = 15.
    ) -> float:
        """
        Computes the stress in the steel reinforcement based on its distance to the neutral axis.

        :param distance: Distance from the reinforcement to the neutral axis [m].
        :type distance: float
        :param neutral_axis: Position of the neutral axis from the reference edge [m].
        :type neutral_axis: float
        :param sigma_cls: Compressive stress at the top fiber of the concrete [kPa].
        :type sigma_cls: float
        :param n: Modular ratio (Es/Ec), defaults to 15.
        :type n: float, optional
        :return: Stress in the reinforcing steel [kPa].
        :rtype: float
        """
        return n * sigma_cls * distance / neutral_axis

    def compute_minimum_steel_area(
        self,
        sigma_adm_cls: float,
        sigma_adm_steel: float,
        n: float = 15.,
        As_pred: float = 1e-4
    ) -> tuple[float, float]:
        """
        Computes the minimum required steel reinforcement area to resist positive and negative
        bending moments using nonlinear constrained optimization.

        :param sigma_adm_cls: Maximum allowable compressive stress in concrete [kPa].
        :type sigma_adm_cls: float
        :param sigma_adm_steel: Maximum allowable tensile stress in steel [kPa].
        :type sigma_adm_steel: float
        :param n: Modular ratio (Es/Ec), defaults to 15.
        :type n: float, optional
        :param As_pred: Initial guess for steel area [m²], defaults to 1e-4.
        :type As_pred: float, optional
        :return: Tuple containing minimum required steel area [m²] and the reinforcement ratio (top/bottom).
        :rtype: Tuple[float, float]
        """
        # Constants
        cm_m = 10000  # Conversion from m² to cm²

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
            As, As_ratio = x[0], x[1]
            return As * (1 + As_ratio)

        # Constraints
        def moment_balance_top_reinforcement_constraint_pos(x) -> float:
            As, _, y_pos, _, sigma_cls_pos, _ = x
            As *= cm_m
            sigma_s = self.compute_steel_stress(d - y_pos, y_pos, sigma_cls_pos, n)
            steel_contribution = As * sigma_s * d_d
            concrete_contribution = 0.5 * b * y_pos * sigma_cls_pos * (cop - y_pos / 3)
            return steel_contribution + concrete_contribution - M_pos

        def moment_balance_bot_reinforcement_constraint_pos(x) -> float:
            As, As_ratio, y_pos, _, sigma_cls_pos, _ = x
            As *= cm_m
            sigma_s = self.compute_steel_stress(y_pos - cop, y_pos, sigma_cls_pos, n)
            steel_contribution = As * As_ratio * sigma_s * d_d
            concrete_contribution = 0.5 * b * y_pos * sigma_cls_pos * (d - y_pos / 3)
            return steel_contribution + concrete_contribution - M_pos

        def moment_balance_top_reinforcement_constraint_neg(x) -> float:
            As, As_ratio, _, y_neg, _, sigma_cls_neg = x
            As *= cm_m
            sigma_s = self.compute_steel_stress(d - y_neg, y_neg, sigma_cls_neg, n)
            steel_contribution = As * As_ratio * sigma_s * d_d
            concrete_contribution = 0.5 * b * y_neg * sigma_cls_neg * (cop - y_neg / 3)
            return steel_contribution + concrete_contribution - M_neg

        def moment_balance_bot_reinforcement_constraint_neg(x) -> float:
            As, _, _, y_neg, _, sigma_cls_neg = x
            As *= cm_m
            sigma_s = self.compute_steel_stress(y_neg - cop, y_neg, sigma_cls_neg, n)
            steel_contribution = As * sigma_s * d_d
            concrete_contribution = 0.5 * b * y_neg * sigma_cls_neg * (d - y_neg / 3)
            return steel_contribution + concrete_contribution - M_neg

        def maximum_adm_steel_stress_top_constraint_pos(x) -> float:
            _, _, y_pos, _, sigma_cls_pos, _ = x
            sigma_s = self.compute_steel_stress(d - y_pos, y_pos, sigma_cls_pos, n)
            return sigma_adm_steel - sigma_s

        def maximum_adm_steel_stress_bot_constraint_pos(x) -> float:
            _, _, y_pos, _, sigma_cls_pos, _ = x
            sigma_s = self.compute_steel_stress(y_pos - cop, y_pos, sigma_cls_pos, n)
            return sigma_adm_steel - sigma_s

        def maximum_adm_steel_stress_top_constraint_neg(x) -> float:
            _, _, _, y_neg, _, sigma_cls_neg = x
            sigma_s = self.compute_steel_stress(d - y_neg, y_neg, sigma_cls_neg, n)
            return sigma_adm_steel - sigma_s

        def maximum_adm_steel_stress_bot_constraint_neg(x) -> float:
            _, _, _, y_neg, _, sigma_cls_neg = x
            sigma_s = self.compute_steel_stress(y_neg - cop, y_neg, sigma_cls_neg, n)
            return sigma_adm_steel - sigma_s

        # Initial guess
        initial_guess = [
            As_pred / cm_m,
            M_neg / M_pos,
            h / 3,
            h / 3,
            sigma_adm_cls,
            sigma_adm_cls
        ]

        # Bounds
        bounds = [
            [0, None],
            [0, None],
            [0, h],
            [0, h],
            [0, sigma_adm_cls],
            [0, sigma_adm_cls]
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
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        As, As_ratio = result.x[0], result.x[1]
        As *= cm_m  # Convert to m²
        return As, As_ratio


@dataclass
class VerifyRectangularBeam:
    """
    Class for verifying the structural integrity of a rectangular beam section.

    Attributes
    ----------
    h : float
        Height of the rectangular beam section.
    b : float
        Width of the rectangular beam section.
    As_top : float
        Area of the top steel reinforcement.
    As_bot : float
        Area of the bottom steel reinforcement.
    cop : float
        Distance from the top fiber to the centroid of the top reinforcement (cover of concrete).
    """

    h: float
    b: float
    As_top: float
    As_bot: float
    cop: float

    def verify_section(self,
                       sigma_cls_adm: float,
                       sigma_s_adm: float,
                       M: float,
                       n: float = 15) -> tuple[float, float]:
        """
        Verifies the beam section by calculating the maximum concrete and steel stresses.

        Parameters
        ----------
        sigma_adm_cls : float
            Allowable concrete stress.
        sigma_adm_steel : float
            Allowable steel stress.
        M : float
            Applied moment on the section.
        n : float, optional
            Modular ratio, by default 15.

        Returns
        -------
        Tuple[float, float]
            A tuple containing the ratio of computed concrete stress to allowable stress,
            and the maximum ratio of steel stress to allowable stress.
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
        section_geometry: SectionGeometry | None = None
    ) -> RectangularSection:
    """
    Designs a rectangular beam section based on positive and negative bending moments,
    allowable concrete stress, allowable steel stress, and optional section geometry.

    If no section geometry is provided, a default rectangular section is designed,
    reinforcement areas are calculated, and the beam is checked for stress compliance.

    :param M_pos: Positive bending moment in kNm
    :type M_pos: float
    :param M_neg: Negative bending moment in kNm
    :type M_neg: float
    :param sigma_cls_adm: Allowable compressive stress of concrete in MPa
    :type sigma_cls_adm: float
    :param sigma_s_adm: Allowable tensile stress of steel in MPa
    :type sigma_s_adm: float
    :param section_geometry: Optional predefined section geometry, defaults to None
    :type section_geometry: Optional[SectionGeometry]
    :raises ValueError: Raised if no valid reinforcement solution is found for top or bottom reinforcement
    :return: Designed rectangular section with specified reinforcement
    :rtype: RectangularSection
    """

    # Initial estimate of steel area (As) for starting reinforcement calculations
    As_initial: float = 1e-4

    # If no predefined section is given, a new section will be designed
    if section_geometry is None:
        # Default values for section width (b) and cover (cop)
        b_default: float = .3
        cop: float = .03

        # Calculate depth and steel area for both positive and negative moments
        d_pos, As_pos = unconditioned_design_bottom_reinf(
            M=M_pos,
            b=b_default,
            sigma_cls_adm=sigma_cls_adm,
            sigma_s_adm=sigma_s_adm
        )
        d_neg, As_neg = unconditioned_design_bottom_reinf(
            M=M_neg,
            b=b_default,
            sigma_cls_adm=sigma_cls_adm,
            sigma_s_adm=sigma_s_adm
        )

        # Use the larger depth for the final design
        d_design = max(d_pos, d_neg)
        As_initial = max(As_pos, As_neg)

        # Round section height to nearest tolerance
        h_design = round_to_nearest(
            d_design + cop,
            tol=.05
        )

        # Define section geometry based on calculated height and default width
        section_geometry = SectionGeometry(
            h_design,
            b_default,
            cop
        )

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
    As, As_ratio = section_design.compute_minimum_steel_area(
        sigma_adm_cls=sigma_cls_adm,
        sigma_adm_steel=sigma_s_adm,
        n=15,
        As_pred=As_initial
    )

    # Calculate required bottom and top reinforcement areas
    As_bot_design = As * As_ratio
    As_top_design = As

    # Create a reinforcement selector object for finding bar combinations
    reinf_selector = ReinforcementCombination(
        section_width=section_geometry.b,
        min_diameter=12,
        max_count=6
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
