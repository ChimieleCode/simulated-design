import numpy as np


def compute_neutral_axis_ratio(
        sigma_cls_adm: float,
        sigma_s_adm: float,
        n: int = 15
) -> float:
    """
    Computes the neutral axis ratio for a rectangular beam section.

    Parameters
    ----------
    sigma_cls_adm : float
        Allowable stress in the concrete.
    sigma_s_adm : float
        Allowable stress in the steel reinforcement.
    n : int, optional
        Modular ratio (steel to concrete modulus ratio), by default 15.

    Returns
    -------
    float
        The neutral axis ratio of the section.
    """
    return sigma_cls_adm / (sigma_cls_adm + sigma_s_adm/n)


def compute_alpha_bottom_reinf(
        k: float,
        sigma_cls_adm: float
) -> float:
    """
    Computes the alpha coefficient for bottom reinforcement in a rectangular beam section.

    Parameters
    ----------
    k : float
        Neutral axis ratio.
    sigma_cls_adm : float
        Allowable stress in the concrete.

    Returns
    -------
    float
        Alpha coefficient for bottom reinforcement.
    """
    return np.sqrt(2 / (k * (1 - k/3) * sigma_cls_adm))


def compute_beta_bottom_reinf(
    alpha: float,
    k: float,
    sigma_s_adm: float
) -> float:
    """
    Computes the beta coefficient for bottom reinforcement in a rectangular beam section.

    Parameters
    ----------
    alpha : float
        Alpha coefficient for bottom reinforcement.
    k : float
        Neutral axis ratio.
    sigma_s_adm : float
        Allowable stress in the steel reinforcement.

    Returns
    -------
    float
        Beta coefficient for bottom reinforcement.
    """
    return (sigma_s_adm * alpha * (1 - k/3))**-1


def compute_maximum_concrete_stress(N: float, Sx: float, y: float) -> float:
    """
    Compute the maximum stress in concrete.

    This function calculates the maximum stress experienced by concrete based on the applied axial force and the
    section properties. The stress is computed at a distance `y` from the neutral axis.

    :param N: Axial force applied to the section (in Newtons).
    :param Sx: Section modulus of the concrete section (in cubic meters).
    :param y: Distance from the neutral axis to the point where the stress is calculated (in meters).
    :return: Maximum concrete stress at distance `y` from the neutral axis (in Pascals).
    """
    return N / Sx * y


def compute_maximum_steel_stress(N: float, Sx: float, y: float, n: float, d: float) -> float:
    """
    Compute the maximum stress in steel reinforcement.

    This function calculates the maximum stress experienced by steel reinforcement in a structural section.
    The stress is computed at a distance `d` from the neutral axis, adjusted by the distance `y` from the neutral
    axis to the centroid of the steel reinforcement.

    :param N: Axial force applied to the section (in Newtons).
    :param Sx: Section modulus of the concrete section (in cubic meters).
    :param y: Distance from the neutral axis to the centroid of the steel reinforcement (in meters).
    :param n: Modulus of the steel reinforcement material (usually greater than 1).
    :param d: Distance from the neutral axis to the location where the stress in the steel is to be calculated (in meters).
    :return: Maximum steel stress at distance `d` from the neutral axis (in Pascals).
    """
    return n * N / Sx * (d - y)


def compute_static_moment(b, y, n, As_s, d_s):
    """
    Computes the static moment for the section.

    :param b: The width of the beam section.
    :param y: Neutral axis position.
    :param n: Ratio between Ec and Es.
    :param As_s: Sequence of areas of steel reinforcement.
    :param d_s: Sequence of depths of steel reinforcement.

    :return: Static moment value.
    """
    # Calculate the static moment
    Sx = (b * y ** 2 / 2) + n * sum(As * (d - y) for As, d in zip(As_s, d_s))
    return Sx


# Function to compute the neutral axis
def compute_neutral_axis(
    n,
    b,
    u,
    As_s,
    d_s
) -> float:
    """
    Computes the neutral axis for a given beam section using a cubic polynomial.

    The function calculates the neutral axis by solving the cubic equation:

    \[
    a_3 \cdot y^3 + a_2 \cdot y^2 + a_1 \cdot y + a_0 = 0
    \]

    where the coefficients \(a_3\), \(a_2\), \(a_1\), and \(a_0\) are defined based on the input parameters.

    **Cubic Equation Coefficients:**
    - \(a_3 = \frac{b}{6}\)
    - \(a_2 = \frac{b \cdot u}{2}\)
    - \(a_1 = n \cdot \sum (A_s \cdot (u + d))\)
    - \(a_0 = -n \cdot \sum (A_s \cdot d \cdot (u + d))\)

    **Root Selection:**
    The function only considers positive roots and returns the first positive root found.

    :param n: The modular ratio, representing the ratio of elastic moduli of two materials.
    :param b: The width of the beam section.
    :param u: The ultimate stress in the beam.
    :param As_s: Sequence of areas of steel reinforcement.
    :param d_s: Sequence of depths of steel reinforcement from the top of the beam section.

    :raises ValueError: If `As_s` and `d_s` are not of the same length.

    :return: The position of the neutral axis from the top of the beam section.

    **Example:**

    ```python
    n = 8.0
    b = 300.0
    u = 40.0
    As_s = [1200.0, 1000.0]
    d_s = [50.0, 150.0]

    neutral_axis = compute_neutral_axis(n, b, u, As_s, d_s)
    print(neutral_axis)  # Output: <calculated_value>
    ```

    **Notes:**
    - Ensure that the lengths of `As_s` and `d_s` match, otherwise a `ValueError` is raised.
    - The calculation uses numpy's `np.roots` function to solve the cubic equation.

    """
    # Check that the length of steel area and depth sequences are the same length
    if len(As_s) != len(d_s):
        raise ValueError(
            f"As_s and d_s must be of the same size! As_s is size: {len(As_s)} while d_s is size: {len(d_s)}"
        )

    # a3 y^3 + a2 y^2 + a1 y + a0 = 0
    a3 = b / 6
    a2 = b * u / 2
    a1 = n * sum(As * (u + d) for As, d in zip(As_s, d_s))
    a0 = - n * sum(As * d * (u + d) for As, d in zip(As_s, d_s))

    roots = np.roots(
        [
            a3,
            a2,
            a1,
            a0
        ]
    )
    # Keep only positive roots
    root = roots[roots > 0]
    # Just one root should be positive
    return root[0]
