from dataclasses import dataclass

from src.section_design.section_utils import reinforcement_area

# Unit conversion constants
mmq_mq = 1e-6  # mm^2 to m^2


def compute_transversal_reinf_pitch(
        V: float,
        stirrup_area: float,
        d: float,
        sigma_s_adm: float
    ) -> float:
    """
    Computes the required transverse reinforcement area per unit length (spacing) for a column
    or beam subject to a shear force.

    :param V: Shear force to be resisted (in kN).
    :param stirrup_area: Area of a single stirrup leg (in m²).
    :param d: Effective depth of the section (in meters).
    :param sigma_s_adm: Allowable tensile stress in the stirrup steel (in kPa).
    :return: Required transverse reinforcement spacing (in meters).
    """

    # Compute the required transverse reinforcement area
    return (0.9 * stirrup_area * sigma_s_adm * d) / V


def compute_tangential_stress(
        V: float,
        d: float,
        b: float
    ) -> float:
    """
    Computes the tangential (shear) stress in a rectangular cross-section due to an applied shear force.

    :param V: Applied shear force (in kN).
    :param d: Effective depth of the section (in meters).
    :param b: Width of the section (in meters).
    :return: Tangential (shear) stress in the section (in kPa).
    """
    return V / (0.9 * b * d)


@dataclass
class ShearSectionDesignStirrups:
    h: float
    b: float
    cop: float

    def verify_tau(
            self,
            V: float,
            tau_adm: float
        ) -> float:
        """
        Verifies the tangential (shear) stress in the section against the allowable stress.

        :param V: Applied shear force (in kN).
        :param tau_adm: Allowable tangential (shear) stress (in kPa).
        :return: The ratio of the computed tangential stress to the allowable tangential stress (unitless).
        """
        d = self.h - self.cop
        b = self.b

        # Compute the tangential stress
        tau = compute_tangential_stress(V, d, b)

        # Return the ratio of computed stress to allowable stress
        return tau / tau_adm

    def design_reinforcement(
            self,
            V_striups: float,
            sigma_s_adm: float,
            diameter: int | float = 8,
            min_spacing: float = 0.33
        ) -> float:
        """
        Designs the stirrup reinforcement for shear, based on the applied shear force and material properties.

        :param V_striups: Shear force resisted by the stirrups (in kN).
        :param sigma_s_adm: Allowable tensile stress in the stirrup steel (in kPa).
        :param diameter: Diameter of the stirrups (in mm), defaults to 8.
        :param min_spacing: Minimum allowable spacing between stirrups (in meters), defaults to 0.33.
        :return: The required spacing between stirrups (in meters).
        """
        # Effective depth of the section
        d = self.h - self.cop

        # Area of one stirrup (2 legs, multiplied by the area of one leg)
        Ast = reinforcement_area(2, diameter) * mmq_mq

        # Compute the required stirrup pitch (spacing) based on the applied shear force and stirrup properties
        return min(
            min_spacing,
            compute_transversal_reinf_pitch(
                V=V_striups,
                stirrup_area=Ast,
                d=d,
                sigma_s_adm=sigma_s_adm
            )
        )
