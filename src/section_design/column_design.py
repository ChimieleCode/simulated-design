from dataclasses import dataclass

from scipy.optimize import minimize

from src.section_design.section_geometry import (RectangularSection,
                                                 SectionGeometry)
from src.section_design.section_utils import (ReinforcementCombination,
                                              reinforcement_area)
from src.section_design.stress_functions import (
    compute_maximum_concrete_stress, compute_maximum_steel_stress,
    compute_neutral_axis, compute_static_moment)
from src.sollicitations import MemberSollicitation
from src.units_conversion import mmq_mq


class ColumnSectionDesign:
    """
    This class is responsible for designing the column section given its geometry
    and applied actions. It computes the required minimum steel reinforcement and
    checks if the design satisfies the given stress limits.

    :param section_geometry: The geometry of the column section.
    :type section_geometry: SectionGeometry
    :param design_sollicitations: The forces (axial and moment) applied to the column.
    :type design_sollicitations: MemberSollicitations
    """
    def __init__(self,
                 section_geometry: SectionGeometry,
                 design_sollicitations: MemberSollicitation):
        self.__section_geometry = section_geometry
        self.__design_sollicitations = design_sollicitations

    @staticmethod
    def compute_steel_stress(
            distance: float,
            neutral_axis: float,
            sigma_cls: float,
            n: float = 15.
        ) -> float:
        """
        Computes the stress in steel reinforcement based on its distance from
        the neutral axis, concrete stress, and modular ratio.

        :param distance: Distance between the reinforcement and the neutral axis.
        :type distance: float
        :param neutral_axis: Location of the neutral axis.
        :type neutral_axis: float
        :param sigma_cls: Concrete stress at the top.
        :type sigma_cls: float
        :param n: Modular ratio (default is 15).
        :type n: float
        :return: Calculated steel stress.
        :rtype: float
        """
        return n * sigma_cls * distance / neutral_axis

    def compute_minimum_steel_area(
            self,
            sigma_adm_cls: float,
            sigma_adm_steel: float,
            n: float = 15.,
            As_pred: float = 1e-4
        ) -> float:
        """
        Computes the minimum required steel area (As) for the column section
        based on the concrete and steel stress limitations.

        :param sigma_adm_cls: Allowable compressive stress of concrete in MPa.
        :type sigma_adm_cls: float
        :param sigma_adm_steel: Allowable tensile stress of steel in MPa.
        :type sigma_adm_steel: float
        :param n: Modular ratio, defaults to 15.
        :type n: float, optional
        :param As_pred: Initial estimate of steel reinforcement area in m^2, defaults to 1e-4.
        :type As_pred: float, optional
        :return: The minimum required steel area.
        :rtype: float
        """

        # Retrieve column section geometry
        h = self.__section_geometry.h
        b = self.__section_geometry.b
        cop = self.__section_geometry.cop

        # Get applied loads (Moment and Axial)
        M = self.__design_sollicitations.M
        N = self.__design_sollicitations.N

        # Compute useful parameters
        d = h - cop  # Effective depth
        e = M / N    # Eccentricity of the applied load
        u = e - h/2  # Load eccentricity offset
        d_d = d - cop  # Distance between top and bottom reinforcement

        # Objective function for optimization (minimizes As)
        def obj_function(x) -> float:
            return x[0]  # Minimize steel area (As)

        # Constraints to maintain moment balance at the top reinforcement
        def moment_balance_top_reinforcement_constraint(x) -> float:
            As, y, sigma_cls = x
            sigma_s = self.compute_steel_stress(d - y, y, sigma_cls, n)
            axial_contribution = N * (u + cop)
            steel_contribution = As * sigma_s * d_d
            concrete_contribution = 0.5 * b * y * sigma_cls * (cop - y / 3)
            return steel_contribution + concrete_contribution - axial_contribution

        # Constraints to maintain moment balance at the bottom reinforcement
        def moment_balance_bot_reinforcement_constraint(x) -> float:
            As, y, sigma_cls = x
            sigma_s = self.compute_steel_stress(y - cop, y, sigma_cls, n)
            axial_contribution = N * (u + d)
            steel_contribution = As * sigma_s * d_d
            concrete_contribution = 0.5 * b * y * sigma_cls * (d - y / 3)
            return steel_contribution + concrete_contribution - axial_contribution

        # Constraint ensuring positive steel stress at the top
        def positive_steel_stress_top_constraint(x) -> float:
            _, y, sigma_cls = x
            return self.compute_steel_stress(d - y, y, sigma_cls, n)

        # Constraint ensuring steel stress at the top does not exceed the allowable stress
        def maximum_adm_steel_stress_top_constraint(x) -> float:
            return sigma_adm_steel - positive_steel_stress_top_constraint(x)

        # Constraint ensuring positive steel stress at the bottom
        def positive_steel_stress_bot_constraint(x) -> float:
            _, y, sigma_cls = x
            return self.compute_steel_stress(y - cop, y, sigma_cls, n)

        # Constraint ensuring steel stress at the bottom does not exceed the allowable stress
        def maximum_adm_steel_stress_bot_constraint(x) -> float:
            return sigma_adm_steel - positive_steel_stress_bot_constraint(x)

        # Define the initial guesses for the variables
        initial_guess = [
            As_pred,        # Estimated reinforcement area (As) in m^2
            h / 3,          # Initial estimate for the neutral axis position
            sigma_adm_cls   # Concrete stress
        ]

        # Bounds for the variables (positive values only)
        bounds = [
            [0, None],          # Reinforcement area must be positive
            [0, h],             # Neutral axis must be within the section height
            [0, sigma_adm_cls]  # Concrete stress must be less than the allowable stress
        ]

        # Constraints for optimization
        constraints = (
            {'type': 'eq', 'fun': moment_balance_bot_reinforcement_constraint},
            {'type': 'eq', 'fun': moment_balance_top_reinforcement_constraint},
            {'type': 'ineq', 'fun': positive_steel_stress_bot_constraint},
            {'type': 'ineq', 'fun': positive_steel_stress_top_constraint},
            {'type': 'ineq', 'fun': maximum_adm_steel_stress_bot_constraint},
            {'type': 'ineq', 'fun': maximum_adm_steel_stress_top_constraint}
        )

        # Perform optimization to minimize As subject to constraints
        result = minimize(
            obj_function,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        # Return the optimized steel area (As)
        As, *_ = result.x
        return As


@dataclass
class VerifyRectangularColumn:
    """
    This class is responsible for verifying if a rectangular column section meets
    stress limits based on the applied axial load and moment.

    :param h: Height of the column section in meters.
    :type h: float
    :param b: Width of the column section in meters.
    :type b: float
    :param As_top: Steel area in the top reinforcement.
    :type As_top: float
    :param As_bot: Steel area in the bottom reinforcement.
    :type As_bot: float
    :param cop: Concrete cover in meters.
    :type cop: float
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
            N: float,
            M: float,
            n: float = 15
        ) -> tuple[float, float]:
        """
        Verifies whether the column section meets the allowable stress limits for both
        concrete and steel based on the applied axial load and moment.

        :param sigma_cls_adm: Allowable compressive stress of concrete in MPa.
        :type sigma_cls_adm: float
        :param sigma_s_adm: Allowable tensile stress of steel in MPa.
        :type sigma_s_adm: float
        :param N: Axial load in kN.
        :type N: float
        :param M: Bending moment in kNm.
        :type M: float
        :param n: Modular ratio, defaults to 15.
        :type n: float, optional
        :return: Tuple containing the concrete stress ratio and maximum steel stress ratio.
        :rtype: Tuple[float, float]
        """

        # Calculate eccentricity and neutral axis
        e = M / N
        u = e - self.h / 2

        # Determine section neutral axis
        y = compute_neutral_axis(n, self.b, u, [self.As_top, self.As_bot], [self.cop, self.h - self.cop])

        # Compute static moment for stress calculations
        Sx = compute_static_moment(self.b, y, n, [self.As_top, self.As_bot], [self.cop, self.h - self.cop])

        # Compute the maximum stresses in concrete and steel
        sigma_cls = compute_maximum_concrete_stress(N, Sx, y)
        sigma_steel = [
            compute_maximum_steel_stress(N, Sx, y, n, d)
            for d in [self.cop, self.h - self.cop]
        ]

        # Check against allowable stresses
        check_cls = sigma_cls / sigma_cls_adm
        check_steel = [abs(sigma_s) / sigma_s_adm for sigma_s in sigma_steel]

        # Return the stress ratios for concrete and steel
        return check_cls, max(check_steel)



def design_column_section(
        M: float,
        N: float,
        sigma_cls_adm: float,
        sigma_s_adm: float,
        section_geometry: SectionGeometry
    ) -> RectangularSection:
    """
    Designs a rectangular column section based on axial force (N), bending moment (M),
    allowable compressive stress, and allowable tensile stress. It computes the required
    steel reinforcement and checks if the section meets the stress limits.

    :param M: Applied bending moment in kNm
    :type M: float
    :param N: Applied axial load in kN
    :type N: float
    :param sigma_cls_adm: Allowable compressive stress of concrete in MPa
    :type sigma_cls_adm: float
    :param sigma_s_adm: Allowable tensile stress of steel in MPa
    :type sigma_s_adm: float
    :param section_geometry: Geometry of the rectangular section (height, width, cover)
    :type section_geometry: SectionGeometry
    :raises ValueError: Raised if no valid reinforcement solution is found
    :return: Designed rectangular column section with calculated reinforcement
    :rtype: RectangularSection
    """

    # Initial estimate of steel area (As) for starting reinforcement calculations
    As_initial: float = 1e-4

    # Define the actions (axial load and moment) on the column
    section_actions = MemberSollicitation(
        M=M,
        N=N
    )

    # Initialize column design with given section geometry and actions
    section_design = ColumnSectionDesign(
        section_geometry,
        section_actions
    )

    # Compute the minimum required steel area for the column
    As_design = section_design.compute_minimum_steel_area(
        sigma_adm_cls=sigma_cls_adm,
        sigma_adm_steel=sigma_s_adm,
        n=15,
        As_pred=As_initial
    )

    # Create a reinforcement selector object to find the optimal reinforcement combination
    reinf_selector = ReinforcementCombination(
        section_width=section_geometry.b,
        min_diameter=12,
        max_count=6
    )

    section_check_passed: bool = False
    while not section_check_passed:
        # Find the best bar combination that meets the steel area requirements
        reinf = reinf_selector.find_combination(As_design)

        # Raise error if no valid reinforcement solution is found
        if reinf is None:
            raise ValueError(f"Warning, no solution was found with reinforcement area: {As_design * 10000:.2f} cmq")

        # Calculate the actual steel area of the selected reinforcement
        As = reinforcement_area(*reinf) * mmq_mq

        # Verify the column section for the applied loads and moments
        section_check = VerifyRectangularColumn(
            **section_geometry.__dict__,
            As_top=As,
            As_bot=As
        )
        moment_check = section_check.verify_section(
            sigma_cls_adm=sigma_cls_adm,
            sigma_s_adm=sigma_s_adm,
            M=M,
            N=N
        )

        # Check if the column design passes all verification checks
        section_check_passed = all(check <= 1 for check in (moment_check))

        # If the section fails the check, increase the steel area slightly and try again
        if not section_check_passed:
            As_design = As + 1e-8

    # After passing the checks, retrieve the reinforcement count and diameter
    reinf_count, reinf_d = reinf    # type: ignore[possiblyUnboundVariable] must be initialized in while loop

    # Return the designed rectangular section with top and bottom reinforcement
    return RectangularSection(
        **section_geometry.__dict__,
        top_reinf_count=reinf_count,
        top_reinf_d=reinf_d,
        bot_reinf_count=reinf_count,
        bot_reinf_d=reinf_d
    )
