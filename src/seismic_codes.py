from collections.abc import Callable, Sequence
from enum import Enum


class BuildingCode(Enum):
    RDL_573 = 'RDL 573'
    RDL_431 = 'RDL 431'
    RDL_640 = 'RDL 640'
    RDL_2105 = 'RDL 2105'
    Law_1684 = 'Law 1684'
    DM_40_75 = 'DM 40 75'
    DM_515_81 = 'DM 515 81'
    DM_1984 = 'DM 1984'
    NewCode = 'New Code'


class SeismicCat(Enum):
    CatI = 'Category I'
    CatII = 'Category II'
    CatIII = 'Category III'


class SeismicForces:
    def __init__(self):
        # Map building codes to force calculation methods that do not use the heights as a param
        self.forces_methods: dict[BuildingCode, Callable[[Sequence[float], SeismicCat], list[float]]] = {
            BuildingCode.RDL_573: self._RDL_573_forces,
            BuildingCode.RDL_431: self._RDL_431_forces,
            BuildingCode.RDL_640: self._RDL_640_forces,
            BuildingCode.RDL_2105: self._RDL_640_forces,  # No change in formulation
            BuildingCode.Law_1684: self._RDL_640_forces,  # No change in formulation
        }

        # Map building codes to force calculation methods that use the heights as a param
        self.forces_methods_with_height: dict[BuildingCode, Callable[[Sequence[float], Sequence[float], SeismicCat], list[float]]] = {
            BuildingCode.DM_40_75: self._DM_40_75_forces,
            BuildingCode.DM_515_81: self._DM_40_75_forces,  # No change in formulation
            BuildingCode.DM_1984: self._DM_1984_forces,
        }

    # Dispatch method
    def compute_forces(
        self,
        building_code: BuildingCode,
        weights: Sequence[float],
        seismic_cat: SeismicCat = SeismicCat.CatI,
        floor_heights: Sequence[float] | None= None
    ) -> list[float]:
        """
        Compute the seismic forces based on the building code and category.

        :param building_code: The building code to use for calculation.
        :param weights: Sequence of weights for each floor.
        :param seismic_cat: Optional seismic category, required for certain codes.
        :param floor_heights: Optional sequence of floor heights, required for certain codes.
        :return: Calculated seismic forces as a list of floats.
        :raises ValueError: If the building code is not supported or required arguments are missing.
        """
        # Determine the appropriate method based on the building code
        if building_code in self.forces_methods:
            method = self.forces_methods[building_code]
            return method(weights, seismic_cat)

        elif building_code in self.forces_methods_with_height:
            if floor_heights is None:
                raise ValueError('Floor heights are required for this building code.')
            method = self.forces_methods_with_height[building_code]
            return method(weights, floor_heights, seismic_cat)

        else:
            raise ValueError(f"Building code {building_code} is not supported.")

    @staticmethod
    def _RDL_573_forces(weights: Sequence[float], seismic_cat: SeismicCat | None = None) -> list[float]:
        """
        Calculate forces for RDL 573 building code.
        """
        forces_coeff = [1 / 8]

        floors = len(weights)
        # Floors above 1
        if floors > 1:
            forces_coeff += [1 / 6] * (floors - 1)

        return [c * w for c, w in zip(forces_coeff, weights)]

    def _RDL_431_forces(self, weights: Sequence[float], seismic_cat: SeismicCat) -> list[float]:
        """
        Calculate forces for RDL 431 building code.
        """
        floors = len(weights)

        # Case CatIII no CatIII
        if seismic_cat is SeismicCat.CatIII:
            return [0] * floors

        # Case CatII, all floors equal
        if seismic_cat is SeismicCat.CatII:
            return [1 / 10 * w for w in weights]

        # Case CatI equal to older
        return self._RDL_573_forces(weights)

    @staticmethod
    def _RDL_640_forces(weights: Sequence[float], category: SeismicCat) -> list[float]:
        """
        Calculate forces for RDL 640 building code with amplification coefficients.
        """
        seismic_acc: dict[SeismicCat, float] = {
            SeismicCat.CatI: 0.1,
            SeismicCat.CatII: 0.07,
            SeismicCat.CatIII: 0  # No CatIII in this building code
        }
        return [seismic_acc[category] * w for w in weights]

    @staticmethod
    def _DM_40_75_forces(
        weights: Sequence[float],
        floor_heights: Sequence[float],
        category: SeismicCat,
        amp_coeff: float | None = None
    ) -> list[float]:
        """
        Calculate forces for DM 40 75 building code.
        """
        if amp_coeff is None:
            # Assuming residential (beta = 1), No dynamic analysis (T0 unknown -> R = 1)
            # Assuming normal soil (epsilon = 1)
            seismic_acc: dict[SeismicCat, float] = {
                SeismicCat.CatI: 0.1,
                SeismicCat.CatII: 0.07,
                SeismicCat.CatIII: 0  # No CatIII in this building code
            }
            amp_coeff = seismic_acc[category]

        total_weight = sum(weights)
        total_weight_height = sum(w * h for w, h in zip(weights, floor_heights))
        gamma = [h * total_weight / total_weight_height for h in floor_heights]

        return [w * g * amp_coeff for w, g in zip(weights, gamma)]

    def _DM_1984_forces(self, weights: Sequence[float], floor_heights: Sequence[float], category: SeismicCat) -> list[float]:
        """
        Calculate forces for DM 1984 building code.
        """
        # Assuming residential (beta = 1), No dynamic analysis (T0 unknown -> R = 1)
        # Assuming normal soil (epsilon = 1)
        if category is SeismicCat.CatIII:
            catIII_coeff = 0.04
            return self._DM_40_75_forces(weights, floor_heights, category, catIII_coeff)

        return self._DM_40_75_forces(weights, floor_heights, category)


class SeismicWeight:
    """
    A class to calculate the seismic weight for a single floor based on
    permanent loads (G), live loads (Q), and an optional seismic category.
    This class provides methods to compute seismic weights according to
    various building codes. Each building code has its own formula for
    calculating the seismic weight, and the appropriate method is selected
    based on the provided building code.
    Attributes:
        weight_methods (dict[BuildingCode, Callable[..., float]]):
            A mapping of building codes to their respective weight calculation methods.
    Methods:
        compute_weight(building_code: BuildingCode, G: float, Q: float, seismic_cat: Optional[SeismicCat] = None) -> float:
            Compute the seismic weight based on the building code, permanent loads (G),
            live loads (Q), and an optional seismic category.
    """
    def __init__(self):
        # Map building codes to weight calculation methods
        self.weight_methods: dict[BuildingCode, Callable[..., float]] = {
            BuildingCode.RDL_573: self._RDL_573_weights,
            BuildingCode.RDL_431: self._RDL_431_weights,
            BuildingCode.RDL_640: self._RDL_640_weights,
            BuildingCode.RDL_2105: self._RDL_2105_weights,
            BuildingCode.Law_1684: self._RDL_2105_weights,  # No change in formulation
            BuildingCode.DM_40_75: self._DM_40_75_weights,
            BuildingCode.DM_515_81: self._DM_40_75_weights,  # No change in formulation
            BuildingCode.DM_1984: self._DM_40_75_weights,    # No change in formulation
            BuildingCode.NewCode: self._raise_not_implemented
        }

    def compute_weight(
        self,
        building_code: BuildingCode,
        G: float,
        Q: float,
        seismic_cat: SeismicCat | None = None
    ) -> float:
        """
        Compute the seismic weight based on the building code and category.

        :param building_code: The building code to use for calculation.
        :param G: Dead load.
        :param Q: Live load.
        :param seismic_cat: Optional seismic category, required for certain codes.
        :return: Calculated seismic weight.
        :raises ValueError: If the building code is not supported.
        """
        if building_code not in self.weight_methods:
            raise ValueError(f"Building code {building_code} is not supported.")

        method = self.weight_methods[building_code]

        # Call the method with the appropriate arguments
        if building_code == BuildingCode.RDL_640:
            if seismic_cat is None:
                raise ValueError('Seismic category is required for RDL 640.')
            return method(G, Q, seismic_cat)
        else:
            return method(G, Q)

    @staticmethod
    def _RDL_573_weights(G: float, Q: float) -> float:
        """
        Calculate weights for RDL 573 building code.
        """
        return 1.5 * (G + Q)

    @staticmethod
    def _RDL_431_weights(G: float, Q: float) -> float:
        """
        Calculate weights for RDL 431 building code.
        """
        return 4/3 * (G + Q)

    @staticmethod
    def _RDL_640_weights(G: float, Q: float, cathegory: SeismicCat) -> float:
        """
        Calculate weights for RDL 640 building code with amplification coefficients.
        """
        ampl_coefficient: dict[SeismicCat, float] = {
            SeismicCat.CatI: 1.4,
            SeismicCat.CatII: 1.25,
            SeismicCat.CatIII: 0  # No CatIII in this building code
        }
        return max(1/3 * Q + G, 2/3 * (Q + G)) * ampl_coefficient[cathegory]

    @staticmethod
    def _RDL_2105_weights(G: float, Q: float) -> float:
        """
        Calculate weights for RDL 2105 building code.
        """
        return (1/3 * Q + G)

    @staticmethod
    def _DM_40_75_weights(G: float, Q: float) -> float:
        """
        Calculate weights for DM 40 75 building code.
        """
        return (1/3 * Q + G)

    @staticmethod
    def _raise_not_implemented(*args) -> float:
        """
        Raise NotImplementedError for unsupported methods.
        """
        raise NotImplementedError(f'The provided combination of args: {args} is not implemented')
