from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.sollicitations import MemberSollicitation


def compute_floors_shear(forces: Sequence[float]) -> list[float]:
    """
    Computes the shear forces for each floor based on the provided forces.

    The shear force for each floor is calculated as the cumulative sum of forces
    from the top floor to the bottom floor. The result is a list where each
    element represents the cumulative shear force at each floor, starting from
    the bottom.

    :param forces:
        A sequence of forces at each floor, where each element represents the
        force at a specific floor. The sequence is expected to be ordered from
        the top floor to the bottom floor.

    :return:
        A list of shear forces for each floor, starting from the bottom floor.
        The list is derived by calculating the cumulative sum of the reversed
        forces sequence and then reversing it back.

    :Example:

    >>> compute_floors_shear([10.0, 20.0, 30.0])
    [60.0, 50.0, 30.0]

    This output indicates:
    - For the bottom floor: shear = 10.0 + 20.0 + 30.0 = 60.0
    - For the middle floor: shear = 20.0 + 30.0 = 50.0
    - For the top floor: shear = 30.0 = 30.0
    """
    return list(np.cumsum(forces[::-1])[::-1])


def compute_beam_shears(beam_moments: Sequence[float], span_length: float) -> list[float]:
    """
    Computes the shear forces for beams based on their moments and span length.

    For each beam, the shear force is calculated as twice the beam moment divided
    by the span length.

    :param beam_moments:
        A sequence of moments for each beam. Each element represents the moment
        at a specific beam.

    :param span_length:
        The length of the span of the beam. This value is used to normalize the
        shear force calculation.

    :return:
        A list of shear forces for each beam, where each shear force is computed
        as 2 * moment / span_length.

    :raises ValueError:
        If `span_length` is zero or negative, a ValueError is raised to indicate
        that the span length must be positive.

    :Example:

    >>> compute_beam_shears([10.0, 20.0, 30.0], 5.0)
    [4.0, 8.0, 12.0]

    This output indicates that:
    - For the first beam: shear = 2 * 10.0 / 5.0 = 4.0
    - For the second beam: shear = 2 * 20.0 / 5.0 = 8.0
    - For the third beam: shear = 2 * 30.0 / 5.0 = 12.0
    """
    if span_length <= 0:
        raise ValueError(f"Span length must be positive, but got {span_length}.")

    return [2 * moment / span_length for moment in beam_moments]


def compute_beam_moments(columns_moment: Sequence[float]) -> list[float]:
    """
    Computes the beam moments based on column moments.

    For each floor (except the last one), the moment is computed as the average
    of the column moments of the current floor and the next floor. For the last
    floor, the moment is calculated as half of the moment from the previous floor.

    :param columns_moment:
        A sequence of moments for each column. Each element represents the moment
        for a specific floor.

    :return:
        A list of beam moments. Each moment is computed as the average of the moments
        from the current and the next floor, with the last floor's moment being half
        of the previous floor's moment.

    :raises ValueError:
        If `columns_moment` is empty, a ValueError is raised.
    """
    if not columns_moment:
        raise ValueError('`columns_moment` cannot be empty.')

    moments = list()
    if len(columns_moment) > 1:
        # Compute moments for all floors except the last
        moments = [(prev + next_) / 2 for prev, next_ in zip(columns_moment, columns_moment[1:])]

    # Compute moment for the last floor
    moments.append(columns_moment[-1] / 2)

    return moments


def compute_column_moments(columns_shear: Sequence[float], heights: Sequence[float]) -> list[float]:
    """
    Computes the moments for each column based on shear forces and heights.

    For each column, the moment is calculated as the product of the shear force
    and the height of the column. The function assumes that each shear force
    corresponds to a height in the same order.

    :param columns_shear:
        A sequence of shear forces for each column. Each element represents
        the shear force applied to a specific column.

    :param heights:
        A sequence of heights corresponding to each column. Each element represents
        the height of a column.

    :return:
        A list of moments for each column, where each moment is the product
        of the shear force and the height for that column.

    :raises ValueError:
        If the lengths of `columns_shear` and `heights` do not match, a ValueError
        is raised with a message indicating the mismatched lengths.

    :Example:

    >>> compute_column_moments([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
    [10.0, 40.0, 90.0]

    This output indicates that:
    - For the first column: moment = 10.0 * 1.0 = 10.0
    - For the second column: moment = 20.0 * 2.0 = 40.0
    - For the third column: moment = 30.0 * 3.0 = 90.0
    """
    if len(columns_shear) != len(heights):
        raise ValueError(f"Length mismatch: columns_shear has {len(columns_shear)} elements, but heights has {len(heights)} elements.")

    return [shear * h for shear, h in zip(columns_shear, heights)]


def compute_columns_shear(floor_shears: Sequence[float], column_count: int) -> list[float]:
    """
    Distributes shear forces per floor across columns.

    Calculates the shear force per column by dividing each floor's total shear
    force by the number of columns minus one.

    :param floor_shears:
        Shear forces for each floor.

    :param column_count:
        Number of columns on each floor. Must be greater than one.

    :return:
        list of shear forces per column for each floor.

    :Example:

    >>> compute_columns_shear([500.0, 600.0, 700.0], 4)
    [166.67, 200.0, 233.33]

    :Note:
        Ensure `column_count` > 1 to avoid division errors.
    """
    return [shear / (column_count - 1) for shear in floor_shears]


def compute_seismic_axial_load(beam_shears: Sequence[float]) -> list[float]:
    """
    Computes the cumulative seismic axial load for beams.

    The axial load for each beam is calculated as the cumulative sum of the beam
    shears from the top floor to the bottom floor. The result is a list where each
    element represents the cumulative axial load at each beam, starting from the
    bottom.

    :param beam_shears:
        A sequence of shear forces for each beam, where each element represents
        the shear force at a specific beam. The sequence is expected to be ordered
        from the top floor to the bottom floor.

    :return:
        A list of cumulative seismic axial loads for each beam, starting from the
        bottom floor.

    :Example:

    >>> compute_seismic_axial_load([10.0, 20.0, 30.0])
    [60.0, 50.0, 30.0]

    This output indicates:
    - For the bottom beam: axial load = 10.0 + 20.0 + 30.0 = 60.0
    - For the middle beam: axial load = 20.0 + 30.0 = 50.0
    - For the top beam: axial load = 30.0 = 30.0
    """
    return list(np.cumsum(beam_shears[::-1])[::-1])


# Hadlers for the data relative to Seismic sollicitations
@dataclass
class RegularSpanFrameMoments:
    """
    Represents the moments in a regular span frame structure.

    :ivar beams_moment:
        A list of moments for beams at each floor.
    :ivar columns_moment:
        A list of moments for columns at each floor.
    """
    beams_moment: list[float]
    columns_moment: list[float]

    def get_beam_moment(self, floor: int) -> float | None:
        """
        Retrieves the beam moment for a specified floor.

        :param floor:
            The index of the floor for which the beam moment is requested.

        :return:
            The beam moment at the specified floor if it exists; otherwise, None.
        """
        try:
            return self.beams_moment[floor]
        except IndexError:
            return None

    def get_internal_column_moment(self, floor: int) -> float | None:
        """
        Retrieves the moment for an internal column at a specified floor.

        :param floor:
            The index of the floor for which the internal column moment is requested.

        :return:
            The internal column moment at the specified floor if it exists; otherwise, None.
        """
        try:
            return self.columns_moment[floor]
        except IndexError:
            return None

    def get_external_column_moment(self, floor: int) -> float | None:
        """
        Retrieves the moment for an external column at a specified floor, halved.

        :param floor:
            The index of the floor for which the external column moment is requested.

        :return:
            Half of the external column moment at the specified floor if it exists; otherwise, None.
        """
        try:
            return self.columns_moment[floor] / 2
        except IndexError:
            return None


@dataclass
class RegularSpanFrameShears:
    """
    Represents the shear forces in a regular span frame structure.

    :ivar beams_shear:
        A list of shear forces for beams at each floor.
    :ivar columns_shear:
        A list of shear forces for columns at each floor.
    """
    beams_shear: list[float]
    columns_shear: list[float]

    def get_beam_shear(self, floor: int) -> float | None:
        """
        Retrieves the shear force for a beam at a specified floor.

        :param floor:
            The index of the floor for which the beam shear is requested.

        :return:
            The beam shear at the specified floor if it exists; otherwise, None.
        """
        try:
            return self.beams_shear[floor]
        except IndexError:
            return None

    def get_internal_column_shear(self, floor: int) -> float | None:
        """
        Retrieves the shear force for an internal column at a specified floor.

        :param floor:
            The index of the floor for which the internal column shear is requested.

        :return:
            The internal column shear at the specified floor if it exists; otherwise, None.
        """
        try:
            return self.columns_shear[floor]
        except IndexError:
            return None

    def get_external_column_shear(self, floor: int) -> float | None:
        """
        Retrieves the shear force for an external column at a specified floor, halved.

        :param floor:
            The index of the floor for which the external column shear is requested.

        :return:
            Half of the external column shear at the specified floor if it exists; otherwise, None.
        """
        try:
            return self.columns_shear[floor] / 2
        except IndexError:
            return None

@dataclass
class RegularSpanFrameSollicitations(
    RegularSpanFrameMoments,
    RegularSpanFrameShears
):
    """
    Combines moment and shear solicitations for a regular span frame structure.

    :ivar beams_moment:
        A list of moments for beams at each floor.
    :ivar columns_moment:
        A list of moments for columns at each floor.
    :ivar beams_shear:
        A list of shear forces for beams at each floor.
    :ivar columns_shear:
        A list of shear forces for columns at each floor.
    """

    def __init__(self,
                 beams_moment: list[float],
                 columns_moment: list[float],
                 beams_shear: list[float],
                 columns_shear: list[float],
                 axial_loads: list[float]):
        """
        Initializes a `RegularSpanFrameSollicitations` instance.

        :param beams_moment:
            A list of moments for beams.
        :param columns_moment:
            A list of moments for columns.
        :param beams_shear:
            A list of shear forces for beams.
        :param columns_shear:
            A list of shear forces for columns.
        :param axial_loads:
            A list of axial loads for beams.
        """
        RegularSpanFrameMoments.__init__(self, beams_moment, columns_moment)
        RegularSpanFrameShears.__init__(self, beams_shear, columns_shear)
        self.axial_loads = axial_loads

    def get_axial_loads(self, floor: int, is_downwind: bool = True) -> float | None:
        """
        Retrieves the axial load for a specified floor.

        :param floor:
            The index of the floor for which the axial load is requested.

        :param is_downwind:
            A boolean indicating whether the axial load is downwind or not.

        :return:
            The axial load at the specified floor if it exists; otherwise, None.
        """
        try:
            axial = self.axial_loads[floor]
            if not is_downwind:
                return axial * -1
            return axial
        except IndexError:
            return None

    def get_internal_column_sollicitations(self, floor: int) -> MemberSollicitation:
        """
        Retrieves the internal column solicitations for a specified floor.

        :param floor:
            The index of the floor for which the internal column solicitations are requested.

        :return:
            A `MemberSollicitation` object containing the internal column moment and shear
            at the specified floor.
        """
        moment = self.get_internal_column_moment(floor)
        shear = self.get_internal_column_shear(floor)

        if moment is None or shear is None:
            raise ValueError(f"Invalid floor index: {floor}.")

        return MemberSollicitation(
            M=moment,
            V=shear
        )

    def get_external_column_sollicitations(self,
                                           floor: int,
                                           include_axial: bool = False,
                                           is_downwind: bool = True) -> MemberSollicitation:
        """
        Retrieves the solicitations for an external column at a specified floor.

        :param floor:
            The index of the floor for which the external column solicitations are requested.

        :param include_axial:
            A boolean indicating whether to include the axial load in the solicitations.
            Defaults to False.

        :param is_downwind:
            A boolean indicating whether the axial load is downwind or not.
            Defaults to True.

        :raises ValueError:
            If the floor index is invalid or the required data is missing.

        :return:
            A `MemberSollicitation` object containing the moment, shear, and optionally
            the axial load for the external column at the specified floor.
        """
        moment = self.get_external_column_moment(floor)
        shear = self.get_external_column_shear(floor)
        axial = 0.
        if include_axial:
            axial = self.get_axial_loads(floor, is_downwind)

        if (moment is None) or (shear is None) or (axial is None):
            raise ValueError(f"Invalid floor index: {floor}.")

        return MemberSollicitation(
            M=moment,
            V=shear,
            N=axial
        )

    def get_beam_sollicitations(self, floor: int) -> MemberSollicitation:
        """
        Retrieves the solicitations for a beam at a specified floor.

        :param floor:
            The index of the floor for which the beam solicitations are requested.

        :raises ValueError:
            If the floor index is invalid or the required data is missing.

        :return:
            A `MemberSollicitation` object containing the moment and shear for the beam
            at the specified floor.
        """
        moment = self.get_beam_moment(floor)
        shear = self.get_beam_shear(floor)

        if (moment is None) or (shear is None):
            raise ValueError(f"Invalid floor index: {floor}.")

        return MemberSollicitation(
            M=moment,
            V=shear
        )


def get_portal_frame_method_sollicitations(
    forces: Sequence[float],
    heights: Sequence[float],
    span_length: float,
    column_count: int) -> RegularSpanFrameSollicitations:
    """
    Computes the solicitations (moments and shears) for a regular span frame structure
    using the portal frame method.

    :param forces:
        A sequence of forces applied at each floor, ordered from the top floor to the bottom floor.
    :param heights:
        A sequence of heights for each floor, where each element represents the height of a column.
    :param span_length:
        The length of the span of the beams.
    :param column_count:
        The number of columns per floor. Must be greater than one.

    :return:
        An instance of `RegularSpanFrameSollicitations` containing the computed moments and shears
        for beams and columns.
    """
    # Compute the shear forces for each floor
    floor_shears = compute_floors_shear(forces)

    # Distribute the floor shear forces across the columns
    columns_shear = compute_columns_shear(floor_shears, column_count)

    # Compute the moments for each column based on shear forces and heights
    columns_moment = compute_column_moments(columns_shear, heights)

    # Compute the moments for beams based on column moments
    beams_moment = compute_beam_moments(columns_moment)

    # Compute the shear forces for beams based on their moments and span length
    beams_shear = compute_beam_shears(beams_moment, span_length)

    # Axial loads for beams are computed as the cumulative sum of beam shears
    axial_loads = compute_seismic_axial_load(beams_shear)

    # Return the computed moments and shears as a RegularSpanFrameSollicitations instance
    return RegularSpanFrameSollicitations(
        beams_moment=beams_moment,
        columns_moment=columns_moment,
        beams_shear=beams_shear,
        columns_shear=columns_shear,
        axial_loads=axial_loads
    )
