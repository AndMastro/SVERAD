import numpy as np
import random
import math

######################
from bidict import bidict
from collections import defaultdict
import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import scipy.sparse as sparse
from typing import *
######################


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def inv_muiltinom_coeff(number_of_players: int, coalition_size: int) -> float:
    """Factor to weight coalitions ins the Shapley formalism.

    Parameters
    ----------
    number_of_players: int
        total number of available players according to the Shapley formalism
    coalition_size
        number of players selected for a coalition
    Returns
    -------
        float
        weight for contribution of coalition
    """
    n_total_permutations = math.factorial(number_of_players)
    n_permutations_coalition = math.factorial(coalition_size)
    n_permutations_remaining_players = math.factorial(number_of_players - 1 - coalition_size)

    return n_permutations_remaining_players * n_permutations_coalition / n_total_permutations


####utils for the definition of the dataset


def construct_check_mol_list(smiles_list: List[str]) -> List[Chem.Mol]:
    mol_obj_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    if None in mol_obj_list:
        invalid_smiles = []
        for smiles, mol_obj in zip(smiles_list, mol_obj_list):
            if not mol_obj:
                invalid_smiles.append(smiles)
        invalid_smiles = "\n".join(invalid_smiles)
        raise ValueError(f"Following smiles are not valid:\n {invalid_smiles}")
    return mol_obj_list


class AtomEnvironment:
    """"A Class to store environment-information for fingerprint features"""
    def __init__(self, environment_atoms: Set[int]):
        self.environment_atoms = environment_atoms  # set of all atoms within radius


class CircularAtomEnvironment(AtomEnvironment):
    """"A Class to store environment-information for morgan-fingerprint features"""

    def __init__(self, central_atom: int, radius: int, environment_atoms: Set[int]):
        super().__init__(environment_atoms)
        self.central_atom = central_atom
        self.radius = radius


class UnfoldedMorganFingerprint:
    """Transforms smiles-strings or molecular objects into unfolded bit-vectors based on Morgan-fingerprints [1].
    Features are mapped to bits based on the amount of molecules they occur in.

    Long version:
        Circular fingerprints do not have a unique mapping to a bit-vector, therefore the features are mapped to the
        vector according to the number of molecules they occur in. The most occurring feature is mapped to bit 0, the
        second most feature to bit 1 and so on...

        Weak-point: features not seen in the fit method are not mappable to the bit-vector and therefore cause an error.
            This behaviour can be deactivated using ignore_unknown=True where these are simply ignored.

    References:
            [1] http://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
    """

    def __init__(self, counted: bool = False, radius: int = 2, use_features: bool = False, ignore_unknown=False):
        """ Initializes the class

        Parameters
        ----------
        counted: bool
            if False, bits are binary: on if present in molecule, off if not present
            if True, bits are positive integers and give the occurrence of their respective features in the molecule
        radius: int
            radius of the circular fingerprint [1]. Radius of 2 corresponds to ECFP4 (radius 2 -> diameter 4)
        use_features: bool
            instead of atoms, features are encoded in the fingerprint. [2]

        ignore_unknown: bool
            if true features not occurring in fitting are ignored for transformation. Otherwise, an error is raised.

        References
        ----------
        [1] http://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
        [2] http://rdkit.org/docs/GettingStartedInPython.html#feature-definitions-used-in-the-morgan-fingerprints
        """

        self._n_bits = None
        self._use_features = use_features
        self._bit_mapping: Optional[bidict] = None

        if not isinstance(counted, bool):
            raise TypeError("The argument 'counted' must be a bool!")
        self._counted = counted

        if not isinstance(ignore_unknown, bool):
            raise TypeError("The argument 'ignore_unknown' must be a bool!")
        self.ignore_unknown = ignore_unknown

        if isinstance(radius, int) and radius >= 0:
            self._radius = radius
        else:
            raise ValueError(f"Number of bits has to be a positive integer! (Received: {radius})")

    def __len__(self):
        return self.n_bits

    @property
    def n_bits(self) -> int:
        """Number of unique features determined after fitting."""
        if self._n_bits is None:
            raise ValueError("Number of bits is undetermined!")
        return self._n_bits

    @property
    def radius(self):
        """Radius for the Morgan algorithm"""
        return self._radius

    @property
    def use_features(self) -> bool:
        """Returns if atoms are hashed according to more abstract features.[2]"""
        return self._use_features

    @property
    def counted(self) -> bool:
        """Returns the bool value for enabling counted fingerprint."""
        return self._counted

    @property
    def bit_mapping(self) -> bidict:
        return self._bit_mapping.copy()

    def fit(self, mol_obj_list: List[Chem.Mol]) -> None:
        mol_iterator = (self._gen_features(mol_obj) for mol_obj in mol_obj_list)
        self._create_mapping(mol_iterator)

    def _gen_features(self, mol_obj: Chem.Mol) -> Dict[int, int]:
        """returns the a dict, where the key is the feature-hash and the value is the count."""
        return AllChem.GetMorganFingerprint(mol_obj, self.radius, useFeatures=self.use_features).GetNonzeroElements()

    def explain_rdmol(self, mol_obj: Chem.Mol) -> dict:
        bi = {}
        _ = AllChem.GetMorganFingerprint(mol_obj, self.radius, useFeatures=self.use_features, bitInfo=bi)
        bit_info = {self.bit_mapping[k]: v for k, v in bi.items()}
        return bit_info

    def explain_smiles(self, smiles: str) -> dict:
        return self.explain_rdmol(Chem.MolFromSmiles(smiles))

    def fit_transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        mol_fp_list = [self._gen_features(mol_obj) for mol_obj in mol_obj_list]
        self._create_mapping(mol_fp_list)
        return self._transform(mol_fp_list)

    def transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        mol_iterator = (self._gen_features(mol_obj) for mol_obj in mol_obj_list)
        return self._transform(mol_iterator)

    def _map_features(self, mol_fp) -> List[int]:
        if self.ignore_unknown:
            return [self._bit_mapping[feature] for feature in mol_fp.keys() if feature in self._bit_mapping[feature]]
        else:
            return [self._bit_mapping[feature] for feature in mol_fp.keys()]
            
    def fit_smiles(self, smiles_list: List[str]):
        mol_obj_list = construct_check_mol_list(smiles_list)
        self.fit(mol_obj_list)

    def fit_transform_smiles(self, smiles_list: List[str]):
        mol_obj_list = construct_check_mol_list(smiles_list)
        return self.fit_transform(mol_obj_list)

    def transform_smiles(self, smiles_list: List[str]):
        mol_obj_list = construct_check_mol_list(smiles_list)
        return self.transform(mol_obj_list)

    def _transform(self, mol_fp_list: Union[Iterator[Dict[int, int]], List[Dict[int, int]]]) -> sparse.csr_matrix:
        data = []
        rows = []
        cols = []
        n_col = 0
        if self._counted:
            for i, mol_fp in enumerate(mol_fp_list):
                features, counts = zip(*mol_fp.items())
                data.append(counts)
                rows.append(self._map_features(features))
                cols.append(i)
                n_col += 1
        else:
            for i, mol_fp in enumerate(mol_fp_list):
                data.extend([1] * len(mol_fp))
                rows.extend(self._map_features(mol_fp))
                cols.extend([i] * len(mol_fp))
                n_col += 1
        return sparse.csr_matrix((data, (cols, rows)), shape=(n_col, self.n_bits))

    def _create_mapping(self, molecule_features: Union[Iterator[Dict[int, int]], List[Dict[int, int]]]):
        unraveled_features = [f for f_list in molecule_features for f in f_list.keys()]
        feature_hash, count = np.unique(unraveled_features, return_counts=True)
        feature_hash_dict = dict(zip(feature_hash, count))
        unique_features = set(unraveled_features)
        feature_order = sorted(unique_features, key=lambda f: (feature_hash_dict[f], f), reverse=True)
        self._bit_mapping = bidict(zip(feature_order, range(len(feature_order))))
        self._n_bits = len(self._bit_mapping)

    def bit2atom_mapping(self, mol_obj: Chem.Mol) -> Dict[int, List[CircularAtomEnvironment]]:
        bit2atom_dict = self.explain_rdmol(mol_obj)
        result_dict = defaultdict(list)

        # Iterating over all present bits and respective matches
        for bit, matches in bit2atom_dict.items():  # type: int, tuple
            for central_atom, radius in matches:  # type: int, int
                if radius == 0:
                    result_dict[bit].append(CircularAtomEnvironment(central_atom, radius, {central_atom}))
                    continue
                env = Chem.FindAtomEnvironmentOfRadiusN(mol_obj, radius, central_atom)
                amap = {}
                _ = Chem.PathToSubmol(mol_obj, env, atomMap=amap)
                env_atoms = amap.keys()
                assert central_atom in env_atoms
                result_dict[bit].append(CircularAtomEnvironment(central_atom, radius, set(env_atoms)))

        # Transforming defaultdict to dict
        return {k: v for k, v in result_dict.items()}


FeatureMatrix = Union[np.ndarray, sparse.csr.csr_matrix]


class DataSet:
    """ Object to contain paired data such das features and label. Supports adding other attributes such as groups.
    """
    def __init__(self, label: np.ndarray, feature_matrix: FeatureMatrix):

        if not isinstance(label, np.ndarray):
            label = np.array(label).reshape(-1)

        if label.shape[0] != feature_matrix.shape[0]:
            raise IndexError

        self.label = label
        self.feature_matrix = feature_matrix
        self._additional_attributes = set()

    def add_attribute(self, attribute_name, attribute_values: np.ndarray):
        if not isinstance(attribute_values, np.ndarray):
            attribute_values = np.array(attribute_values).reshape(-1)

        if attribute_values.shape[0] != len(self):
            raise IndexError("Size does not match!")

        self._additional_attributes.add(attribute_name)
        self.__dict__[attribute_name] = attribute_values

    @property
    def columns(self) -> dict:
        r_dict = {k: v for k, v in self.__dict__.items() if k in self._additional_attributes}
        r_dict["label"] = self.label
        r_dict["feature_matrix"] = self.feature_matrix
        return r_dict

    def __len__(self):
        return self.label.shape[0]

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, idx) -> Union[dict, 'DataSet']:
        if isinstance(idx, int):
            return {col: values[idx] for col, values in self.columns.items()}

        data_slice = DataSet(self.label[idx], self.feature_matrix[idx])
        for additional_attribute in self._additional_attributes:
            data_slice.add_attribute(additional_attribute, self.__dict__[additional_attribute][idx])

        return data_slice


# if __name__ == "__main__":
#     # noinspection SpellCheckingInspection
#     test_smiles_list = ["c1ccccc1",
#                         "CC(=O)C1CCC2C1(CCC3C2CCC4=CC(=O)CCC34C)C",
#                         "c1cc(ccc1C2CCNCC2COc3ccc4c(c3)OCO4)F",
#                         "c1c(c2c(ncnc2n1C3C(C(C(O3)CO)O)O)N)C(=O)N",
#                         "Cc1cccc(c1NC(=O)c2cnc(s2)Nc3cc(nc(n3)C)N4CCN(CC4)CCO)Cl",
#                         "CN(C)c1c2c(ncn1)n(cn2)C3C(C(C(O3)CO)NC(=O)C(Cc4ccc(cc4)OC)N)O",
#                         "CC12CCC(CC1CCC3C2CC(C4(C3(CCC4C5=CC(=O)OC5)O)C)O)O",

#                         ]
#     test_mol_obj_list = construct_check_mol_list(test_smiles_list)

#     ecfp2_1 = UnfoldedMorganFingerprint()
#     fp1 = ecfp2_1.fit_transform(test_mol_obj_list)
#     print(fp1.shape)