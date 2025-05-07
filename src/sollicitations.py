from dataclasses import dataclass


@dataclass
class MemberSollicitation:
    """
    Represents the solicitation forces and moments on a structural member.

    :ivar M:
        The bending moment on the member.
    :ivar V:
        The shear force on the member.
    :ivar N:
        The axial force on the member.
    """
    M: float = 0.0
    V: float = 0.0
    N: float = 0.0

    def __add__(self, other):
        """
        Adds the solicitation forces and moments of two members.

        :param other:
            Another `MemberSollicitation` instance to add.

        :return:
            A new `MemberSollicitation` instance with the summed forces and moments.
        """
        return MemberSollicitation(
            self.M + other.M,
            self.V + other.V,
            self.N + other.N
        )
