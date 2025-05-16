import math
from dataclasses import dataclass

from scipy.optimize import minimize

from src.section_design.detailing_minimums import (
    DetailingCode, column_section_detail_checker,
    get_max_stirrup_spacing_column, get_min_longitudinal_bar_area_column,
    min_reinf_diameter_columns)
from src.section_design.section_geometry import (RectangularSection,
                                                 RectangularSectionElement,
                                                 SectionGeometry)
from src.section_design.section_utils import (ReinforcementCombination,
                                              reinforcement_area)
from src.section_design.shear_design import ShearSectionDesignStirrups
from src.section_design.stress_functions import (
    compute_maximum_concrete_stress, compute_maximum_steel_stress,
    compute_neutral_axis, compute_static_moment)
from src.sollicitations import MemberSollicitation
from src.utils import round_to_nearest

# Unit conversion constants
mmq_mq = 1e-6  # mm^2 to m^2mq
mq_cmq = 1e4  # m^2 to cm^2
mq_dmq = 1e2  # m^2 to dm^2
dmq_mq = 1e-2  # dm^2 to m^2


# Compute strictly necessary area based on the reduction of admissible stress for concrete
def compute_min_concrete_area(
        N: float,
        sigma_cls_adm: float,
        reduction_factor: float = 0.7
) -> float:
    """
    Computes the minimum concrete area required to resist an axial load (kN) based on
    the allowable compressive stress of concrete (sigma_cls_adm).

    :param N: Axial load in kN.
    :param sigma_cls_adm: Allowable compressive stress of concrete in kPa.
    :param reduction_factor: Reduction factor for concrete area, defaults to 0.7.
    :return: Minimum concrete area in m².
    """
    return N / (sigma_cls_adm * reduction_factor)


# Section gemetry
def define_section_geometry(
        min_area: float,
        cop: float,
        oversizing_factor: float = 1.2,
        min_width: float = 0.3
) -> SectionGeometry:
    """
    Defines the geometry of a rectangular column section.

    :param min_area: Minimum area of the column section in m².
    :param cop: Concrete cover in meters.
    :param oversizing_factor: Factor to increase the minimum area, defaults to 1.4.
    :return: SectionGeometry object representing the column section.
    """
    target_area = min_area * oversizing_factor

    # Calculate the dimensions of the rectangular section
    b_pre = math.sqrt(target_area / 2.5)
    b = max(min_width, round_to_nearest(b_pre, .05))  # Round to the nearest 5 cm
    h_min = target_area / b
    h = max(min_width, math.ceil(h_min / 0.05) * 0.05)  # Round up to the nearest 5 cm

    return SectionGeometry(
        h=h,
        b=b,
        cop=cop
    )


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
def design_default_column_section(
        M: float,
        N: float,
        sigma_cls_adm: float,
        sigma_s_adm: float,
        section_geometry: SectionGeometry,
        min_reinf_area: float,
        min_bar_diameter: int = 12,
        max_bar_diameter: int = 30,
        max_bar_count: int = 8
    ) -> RectangularSection:
    """
    Designs a rectangular column section based on axial force (N), bending moment (M),
    allowable compressive stress, and allowable tensile stress. It computes the required
    steel reinforcement and checks if the section meets the stress limits.
    :param M: Applied bending moment in kNm
    :param N: Applied axial load in kN
    :param sigma_cls_adm: Allowable compressive stress of concrete in kPa
    :param sigma_s_adm: Allowable tensile stress of steel in kPa
    :param section_geometry: Geometry of the column section
    :param enforce_bar_diameter: Diameter of the reinforcement bars to be enforced
    :return: Designed rectangular column section with calculated reinforcement
    """
    # Create a reinforcement selector object to find the optimal reinforcement combination
    reinf_selector = ReinforcementCombination(
        section_width=section_geometry.b,
        min_diameter=min_bar_diameter,
        max_diameter=max_bar_diameter,
        max_count=max_bar_count
    )

    section_check_passed: bool = False
    As_design: float = min_reinf_area
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


def design_column_section(
        M: float,
        N: float,
        sigma_cls_adm: float,
        sigma_s_adm: float,
        section_geometry: SectionGeometry,
        n: float = 15,
        min_reinf_area: float = 1e-4,
        min_bar_diameter: int = 12,
        max_bar_diameter: int = 30,
        max_bar_count: int = 8
    ) -> RectangularSection:
    """
    Designs a rectangular column section based on axial force (N), bending moment (M),
    allowable compressive stress, and allowable tensile stress. It computes the required
    steel reinforcement and checks if the section meets the stress limits.

    :param M: Applied bending moment in kNm
    :param N: Applied axial load in kN
    :param sigma_cls_adm: Allowable compressive stress of concrete in kPa
    :param sigma_s_adm: Allowable tensile stress of steel in kPa
    :param section_geometry: Geometry of the column section
    :param n: Modular ratio, defaults to 15.
    :param min_reinf_area: Minimum required steel area in m²
    :param min_bar_diameter: Minimum diameter of the reinforcement bars in mm
    :param max_bar_diameter: Maximum diameter of the reinforcement bars in mm
    :param max_bar_count: Maximum number of reinforcement bars
    :return: Designed rectangular column section with calculated reinforcement
    """
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
        n=n,
        As_pred=min_reinf_area
    )
    As_design = max(As_design, min_reinf_area)

    # Create a reinforcement selector object to find the optimal reinforcement combination
    reinf_selector = ReinforcementCombination(
        section_width=section_geometry.b,
        min_diameter=min_bar_diameter,
        max_diameter=max_bar_diameter,
        max_count=max_bar_count
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


def bidirectional_column_design(
        M_main: float,
        V_main: float,
        M_cross: float,
        V_cross: float,
        N: float,
        sigma_cls_adm: float,
        sigma_s_adm: float,
        detailing_code: DetailingCode,
        cop: float = 0.03,
        shear_rebar_diameter: int = 6
) -> tuple[RectangularSectionElement, RectangularSectionElement]:
    """
    Designs a column section based on axial force (N), bending moment (M), and shear force (V).
    It computes the required steel reinforcement and checks if the design satisfies the given stress limits.

    :param M_main: Applied bending moment in kNm
    :param V_main: Applied shear force in kN
    :param M_cross: Applied bending moment in kNm
    :param V_cross: Applied shear force in kN
    :param sigma_cls_adm: Allowable compressive stress of concrete in kPa
    :param sigma_s_adm: Allowable tensile stress of steel in kPa
    :param section_geometry: Geometry of the column section
    :param N: Axial load in kN
    :param cop: Concrete cover in meters
    :param detailing_code: Detailing code to be used for calculations (RD_39 or DM_76)
    :return: Designed rectangular column section with calculated reinforcement
    """
    # Determine a min
    A_min_cls = compute_min_concrete_area(
        N=N,
        sigma_cls_adm=sigma_cls_adm
    )

    # Define section geometry
    section_geometry = define_section_geometry(
        min_area=A_min_cls,
        cop=cop
    )

    # Min_reifnforcement area
    min_reinf_area = get_min_longitudinal_bar_area_column(
        column_area=section_geometry.area,
        column_min_cls_area=A_min_cls,
        detailing_code=detailing_code
    ) / 2 # divided by 2 for top and bottom reinforcement

    min_diameter = min_reinf_diameter_columns.get(
        key=detailing_code,
        default=0
    ) # type: ignore[reportCallIssue]

    # Deifine main orientation
    section_geometry_main = section_geometry
    section_geometry_cross = section_geometry.rotate_90(new_section=True)
    if M_main < M_cross:
        section_geometry_main, section_geometry_cross = section_geometry_cross, section_geometry_main

    # Design the main column section
    if M_main / N <= section_geometry_main.h / 6:
        # Main is low eccentricity
        if M_cross / N > section_geometry_cross.h / 6:
            # Main is low eccentricity
            # Cross is high eccentricity
            section_cross = design_column_section(
                M=M_cross,
                N=N,
                sigma_cls_adm=sigma_cls_adm,
                sigma_s_adm=sigma_s_adm,
                section_geometry=section_geometry_cross,
                min_reinf_area=min_reinf_area,
                min_bar_diameter=min_diameter
            )
            section_main = design_default_column_section(
                M=M_main,
                N=N,
                sigma_cls_adm=sigma_cls_adm,
                sigma_s_adm=sigma_s_adm,
                section_geometry=section_geometry_main,
                min_reinf_area=min_reinf_area,
                min_bar_diameter=min_diameter
            )
        else:
            # Main is low eccentricity
            # Cross is low eccentricity
            section_cross = design_default_column_section(
                M=M_cross,
                N=N,
                sigma_cls_adm=sigma_cls_adm,
                sigma_s_adm=sigma_s_adm,
                section_geometry=section_geometry_cross,
                min_reinf_area=min_reinf_area,
                min_bar_diameter=min_diameter
            )
            section_main = design_default_column_section(
                M=M_main,
                N=N,
                sigma_cls_adm=sigma_cls_adm,
                sigma_s_adm=sigma_s_adm,
                section_geometry=section_geometry_main,
                min_reinf_area=min_reinf_area,
                min_bar_diameter=min_diameter
            )
    else:
        # Main is high eccentricity
        section_main = design_column_section(
            M=M_main,
            N=N,
            sigma_cls_adm=sigma_cls_adm,
            sigma_s_adm=sigma_s_adm,
            section_geometry=section_geometry_main,
            min_reinf_area=min_reinf_area,
            min_bar_diameter=min_diameter
        )
        if M_cross / N > section_geometry_cross.h / 6:
            # Cross is high eccentricity
            section_cross = design_column_section(
                M=M_cross,
                N=N,
                sigma_cls_adm=sigma_cls_adm,
                sigma_s_adm=sigma_s_adm,
                section_geometry=section_geometry_cross,
                min_reinf_area=min_reinf_area,
                min_bar_diameter=min_diameter
            )
        else:
            # Cross is low eccentricity
            section_cross = design_default_column_section(
                M=M_cross,
                N=N,
                sigma_cls_adm=sigma_cls_adm,
                sigma_s_adm=sigma_s_adm,
                section_geometry=section_geometry_cross,
                min_reinf_area=min_reinf_area,
                min_bar_diameter=min_diameter
            )

    # Shear Design
    # Main
    shear_design_main = ShearSectionDesignStirrups(
        **section_geometry_main.__dict__
    )
    min_main_spacing = get_max_stirrup_spacing_column(
        long_bar_d=shear_rebar_diameter,
        min_dim=min(section_geometry_main.b, section_geometry_main.h),
        detailing_code=detailing_code
    )
    max_spacing_stirrups_main = shear_design_main.design_reinforcement(
        V_striups=V_main,
        sigma_s_adm=sigma_s_adm,
        diameter=shear_rebar_diameter,
        max_spacing=min_main_spacing
    )

    # Cross
    shear_design_cross = ShearSectionDesignStirrups(
        **section_geometry_cross.__dict__
    )
    min_cross_spacing = get_max_stirrup_spacing_column(
        long_bar_d=shear_rebar_diameter,
        min_dim=min(section_geometry_cross.b, section_geometry_cross.h),
        detailing_code=detailing_code
    )
    max_spacing_stirrups_cross = shear_design_cross.design_reinforcement(
        V_striups=V_cross,
        sigma_s_adm=sigma_s_adm,
        diameter=shear_rebar_diameter,
        max_spacing=min_cross_spacing
    )

    # Final stirrups
    stirrups_spacing = min(max_spacing_stirrups_cross, max_spacing_stirrups_main)


    # FINAL SECTIONS
    section_main = RectangularSectionElement(
        **section_main.__dict__,
        stirrups_spacing=stirrups_spacing,
        stirrups_reinf_d=shear_rebar_diameter
    )
    section_cross = RectangularSectionElement(
        **section_cross.__dict__,
        stirrups_spacing=stirrups_spacing,
        stirrups_reinf_d=shear_rebar_diameter
    )


    # Check detailing
    check_main = column_section_detail_checker(
        section=section_main,
        detailing_code=detailing_code,
        min_cls_area=A_min_cls
    )
    check_cross = column_section_detail_checker(
        section=section_cross,
        detailing_code=detailing_code,
        min_cls_area=A_min_cls
    )

    assert check_main, 'Main section does not meet detailing requirements.'
    assert check_cross, 'Cross section does not meet detailing requirements.'

    return (
        section_main,
        section_cross
    )
