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

# Unit conversion constants
mmq_mq = 1e-6  # mm^2 to m^2mq
mq_cmq = 1e4  # m^2 to cm^2
mq_dmq = 1e2  # m^2 to dm^2
dmq_mq = 1e-2  # dm^2 to m^2


# Dimensioning Algorithm
class ColumnSectionDesign:
    """
    This class is responsible for designing the column section given its geometry
    and applied actions. It computes the required minimum steel reinforcement and
    checks if the design satisfies the given stress limits.

    :param section_geometry: The geometry of the column section.
    :param design_sollicitations: The forces (axial and moment) applied to the column.
    """
    def __init__(self,
                 section_geometry: SectionGeometry,
                 design_sollicitations: MemberSollicitation):
        self.__section_geometry = section_geometry
        self.__design_sollicitations = design_sollicitations

    @staticmethod
    def _compute_steel_stress(
            distance: float,
            neutral_axis: float,
            sigma_cls: float,
            n: float = 15.
        ) -> float:
        """
        Computes the stress in steel reinforcement based on its distance from
        the neutral axis, concrete stress, and modular ratio.

        :param distance: Distance between the reinforcement and the neutral axis.
        :param neutral_axis: Location of the neutral axis.
        :param sigma_cls: Concrete stress at the top.
        :param n: Modular ratio (default is 15).
        :return: Calculated steel stress.
        """
        return n * sigma_cls * distance / neutral_axis

    def compute_minimum_steel_area(
            self,
            sigma_adm_cls: float,
            sigma_adm_steel: float,
            n: float = 15.,
            As_pred: float = 1e-4
        ) -> tuple[float, bool]:
        """
        Computes the minimum required steel area (As) for the column section
        based on the concrete and steel stress limitations.

        :param sigma_adm_cls: Allowable compressive stress of concrete in kPa.
        :param sigma_adm_steel: Allowable tensile stress of steel in kPa.
        :param n: Modular ratio, defaults to 15.
        :param As_pred: Initial estimate of steel reinforcement area in m^2, defaults to 1e-4.
        :return: The minimum required steel area. If the optimization succeeds
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
            As, *_ = x
            # Objective function to minimize the steel area (As)
            obj = As**2 # The squared is used to have faster convergence

            # Penalty
            p_1 = moment_balance_top_reinforcement_constraint(x)
            p_2 = moment_balance_bot_reinforcement_constraint(x)
            p_3 = min(
                maximum_adm_steel_stress_top_constraint(x),
                maximum_adm_steel_stress_bot_constraint(x),
                maximum_adm_concrete_stress_constraint(x)
            )

            # Penalty for the constraints
            p_w = 1e6

            # Adjusted obj
            return obj + p_w * (p_1**2 + p_2**2 + p_3**2)

        # Constraints to maintain moment balance at the top reinforcement
        def moment_balance_top_reinforcement_constraint(x) -> float:
            # Setup
            As, y = x
            As_mq = As * dmq_mq  # Convert to m^2
            # this is a unitized constraint, must be scalded
            sigma_cls = positive_concrete_stress_constraint(x) * sigma_adm_cls
            # Compute
            sigma_cls = positive_concrete_stress_constraint(x) * sigma_adm_cls
            sigma_s = self._compute_steel_stress(d - y, y, sigma_cls, n)
            axial_contribution = N * (u + cop)
            steel_contribution = As_mq * sigma_s * d_d
            concrete_contribution = 0.5 * b * y * sigma_cls * (cop - y / 3)
            # Division is used to ensure the constraintis proprely scaled
            return (steel_contribution + concrete_contribution - axial_contribution) / axial_contribution

        # Constraints to maintain moment balance at the bottom reinforcement
        def moment_balance_bot_reinforcement_constraint(x) -> float:
            # Setup
            As, y = x
            As_mq = As * dmq_mq  # Convert to m^2
            # this is a unitized constraint, must be scalded
            sigma_cls = positive_concrete_stress_constraint(x) * sigma_adm_cls
            # Compute
            sigma_s = self._compute_steel_stress(y - cop, y, sigma_cls, n)
            axial_contribution = N * (u + d)
            steel_contribution = As_mq * sigma_s * d_d
            concrete_contribution = 0.5 * b * y * sigma_cls * (d - y / 3)
            # Division is used to ensure the constraintis proprely scaled
            return (steel_contribution + concrete_contribution - axial_contribution) / axial_contribution

        # Constraint ensuring steel stress at the bottom does not exceed the allowable stress
        def maximum_adm_steel_stress_bot_constraint(x) -> float:
            return 1 - positive_steel_stress_bot_constraint(x)

        # Constraint ensuring positive steel stress at the top
        def positive_steel_stress_bot_constraint(x) -> float:
            _, y = x
            sigma_cls = positive_concrete_stress_constraint(x) * sigma_adm_cls
            sigma_steel = self._compute_steel_stress(d - y, y, sigma_cls, n)
            # Division is used to ensure the constraintis proprely scaled
            return sigma_steel / sigma_adm_steel

        # Constraint ensuring steel stress at the top does not exceed the allowable stress
        def maximum_adm_steel_stress_top_constraint(x) -> float:
            return 1 - positive_steel_stress_top_constraint(x)

        # Constraint ensuring positive steel stress at the bottom
        def positive_steel_stress_top_constraint(x) -> float:
            _, y = x
            sigma_cls = positive_concrete_stress_constraint(x) * sigma_adm_cls
            sigma_steel = self._compute_steel_stress(y - cop, y, sigma_cls, n)
            # Division is used to ensure the constraintis proprely scaled
            return sigma_steel / sigma_adm_steel


        # Constraint ensuring positive concrete stress
        def positive_concrete_stress_constraint(x) -> float:
            # Setup
            As, y = x
            As_mq = As * dmq_mq  # Convert to m^2
            # Compute
            Sx = compute_static_moment(b, y, n, [As_mq] * 2, [cop, d])
            sigma_cls = compute_maximum_concrete_stress(N, Sx, y)
            # Division is used to ensure the constraintis proprely scaled
            return sigma_cls / sigma_adm_cls

        # Constraint ensuring concrete stress does not exceed the allowable stress
        def maximum_adm_concrete_stress_constraint(x) -> float:
            # Division is used to ensure the constraintis proprely scaled
            return 1 - positive_concrete_stress_constraint(x)

        # Define the initial guesses for the variables
        initial_guess = [
            As_pred * mq_dmq,        # Estimated reinforcement area (As) in dm^2 to have similar order to h/3
            h / 3          # Initial estimate for the neutral axis position
        ]

        # Bounds for the variables (positive values only)
        bounds = [
            [0, None],          # Reinforcement area must be positive
            [0, h],             # Neutral axis must be within the section height
        ]

        # Constraints for optimization
        constraints = (
            {'type': 'eq', 'fun': moment_balance_bot_reinforcement_constraint},
            {'type': 'eq', 'fun': moment_balance_top_reinforcement_constraint},
            {'type': 'ineq', 'fun': positive_steel_stress_bot_constraint},
            {'type': 'ineq', 'fun': positive_steel_stress_top_constraint},
            {'type': 'ineq', 'fun': positive_concrete_stress_constraint},
            {'type': 'ineq', 'fun': maximum_adm_steel_stress_bot_constraint},
            {'type': 'ineq', 'fun': maximum_adm_steel_stress_top_constraint},
            {'type': 'ineq', 'fun': maximum_adm_concrete_stress_constraint}
        )

        # Perform optimization to minimize As subject to constraints
        result = minimize(
            obj_function,
            initial_guess,
            method='trust-constr',  # SLSQP method fails for some reason, even if faster
            bounds=bounds,
            constraints=constraints
        )

        # Return the optimized steel area (As)
        As, *_ = result.x
        return float(As * dmq_mq), bool(result.success)


# Verify the column section
@dataclass
class VerifyRectangularColumn:
    """
    This class is responsible for verifying if a rectangular column section meets
    stress limits based on the applied axial load and moment.

    :param h: Height of the column section in meters.
    :param b: Width of the column section in meters.
    :param As_top: Steel area in the top reinforcement.
    :param As_bot: Steel area in the bottom reinforcement.
    :param cop: Concrete cover in meters.
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

        :param sigma_cls_adm: Allowable compressive stress of concrete in kPa.
        :param sigma_s_adm: Allowable tensile stress of steel in kPa.
        :param N: Axial load in kN.
        :param M: Bending moment in kNm.
        :param n: Modular ratio, defaults to 15.
        :return: Tuple containing the concrete stress ratio and maximum steel stress ratio.
        """

        # Calculate eccentricity and neutral axis
        e = M / N
        u = e - self.h / 2

        # Determine section neutral axis
        y = compute_neutral_axis(n, self.b, u, [self.As_top, self.As_bot], [self.cop, self.h - self.cop])

        if y > self.h:
            # Low eccentricity condition
            return self._verify_low_eccentricity(
                sigma_cls_adm=sigma_cls_adm,
                sigma_s_adm=sigma_s_adm,
                N=N,
                M=M,
                n=n
            )
        # High eccentricity condition
        return self._verify_high_eccentricity(
            sigma_cls_adm=sigma_cls_adm,
            sigma_s_adm=sigma_s_adm,
            N=N,
            y=y,
            n=n
        )

    def _verify_high_eccentricity(
            self,
            sigma_cls_adm: float,
            sigma_s_adm: float,
            N: float,
            y: float,
            n: float = 15.
        ) -> tuple[float, float]:
        """
        Verifies the section for low eccentricity conditions.

        :param sigma_cls_adm: Allowable compressive stress of concrete in kPa.
        :param sigma_s_adm: Allowable tensile stress of steel in kPa.
        :param N: Axial load in kN.
        :param y: Neutral axis depth (assumed < h).
        :return: Tuple containing the concrete stress ratio and maximum steel stress ratio.
        """
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

    def _verify_low_eccentricity(
            self,
            sigma_cls_adm: float,
            sigma_s_adm: float,
            N: float,
            M: float,
            n: float = 15.
        ) -> tuple[float, float]:
        """
        Verifies the section for low eccentricity conditions.

        :param sigma_cls_adm: Allowable compressive stress of concrete in kPa.
        :param sigma_s_adm: Allowable tensile stress of steel in kPa.
        :param N: Axial load in kN.
        :param M: Bending moment in kNm.
        :return: Tuple containing the concrete stress ratio and maximum steel stress ratio.
        """
        # Calculate eccentricity and neutral axis
        e = M / N
        u = e - self.h / 2
        d = self.h - self.cop
        As = self.As_bot    # Low eccentricity method requires equal As_top and As_bot
        assert abs(self.As_top - self.As_bot) < 1e-6, 'Top and bottom reinforcement must be equal for this method'

        # Implicit variables (simplify computation)
        r = (n * As + .5 * self.b * self.h)
        k = n * As * self.h
        g = 1/6 * self.b * self.h**2
        p = 2 * n * As * d * self.cop / self.h

        # Compute stresses
        sigma_cls_bot = -N * (u + (g + p) / r) / (g + k - 2 * p)
        sigma_cls_top = N / r - sigma_cls_bot
        sigma_s_top = n * (sigma_cls_bot + d * (sigma_cls_top - sigma_cls_bot) / self.h)

        # check ratios
        cls_check = sigma_cls_top / sigma_cls_adm
        steel_check = sigma_s_top / sigma_s_adm

        return cls_check, steel_check


# Function for design
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
    :param N: Applied axial load in kN
    :param sigma_cls_adm: Allowable compressive stress of concrete in kPa
    :param sigma_s_adm: Allowable tensile stress of steel in kPa
    :type section_geometry: SectionGeometry
    :return: Designed rectangular column section with calculated reinforcement
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
    As_design, _ = section_design.compute_minimum_steel_area(
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
            raise ValueError(f"Warning, no solution was found with reinforcement area: {As_design * mq_cmq:.2f} cmq")

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
