from typing import Any, Dict, List, Optional, Tuple
from abc import abstractmethod

import numpy as np

from panqec.bpauli import brank
from ._stabilizer_code import StabilizerCode

Operator = Dict[Tuple, str]  # Coordinate to pauli ('X', 'Y' or 'Z')
SubOperator = Dict[Any, str]  # Subcode qubit location to pauli
Location = Tuple[int, Any]  # (subcode index, location in the subcode)
Weld = Tuple[int, int, Dict[Any, Any]]  # (code a, code b, qubit a -> b)


class WeldedCode(StabilizerCode):
    """Abstract class for stabilizer codes obtained by welding CSS codes

    Implements the welding operation of Michnicki [arXiv:1208.3496].
    Two CSS codes are welded along a set of pairs of qubits, each pair
    containing one qubit of each code. Every pair is contracted to a single
    qubit, and the generators of the welded code are:

    - all the stabilizers of Pauli type opposite to ``weld_pauli``, adopted
      as they are (duplicates obtained after contraction are only kept once),
    - the stabilizers of Pauli type ``weld_pauli`` not acting on the weld,
    - for every stabilizer ``g_a`` of the first code and ``g_b`` of the
      second code of Pauli type ``weld_pauli`` with the same restriction to
      the weld, the welded stabilizer acting as ``g_a`` on the first code,
      as ``g_b`` on the second code, and as their common restriction on the
      contracted qubits.

    ``weld_pauli = 'Z'`` corresponds to a Z-type weld (⊞) and
    ``weld_pauli = 'X'`` to an X-type weld (⊠) in the notation of the paper.
    The weld must be well matched (every weld-acting generator has a partner
    with the same restriction in the other code) and the generators must be
    linearly independent on the weld, in which case the generators above
    generate the welded code (Theorem 3 of the paper).

    Any subclass should override the following methods:
    - get_subcodes() to define the CSS codes being welded
    - get_welds() to define which pairs of qubits are contracted
    - get_logicals_x() and get_logicals_z() (welded_operator() can be used to
      transport logical operators of the subcodes to the welded code)

    Qubits of the welded code are located at ``(i, q)``, where ``q`` is a qubit
    location of the i-th subcode. Contracted qubits are located at the
    representative of lowest subcode index. Similarly, stabilizers are located
    at ``(i, s)``, where ``s`` is a stabilizer location of the i-th subcode.
    All the welds are performed at once.
    """

    weld_pauli: str = 'Z'

    def __init__(
        self,
        L_x: int,
        L_y: Optional[int] = None,
        L_z: Optional[int] = None,
    ):
        self._welding_built = False
        self._subcodes: List[StabilizerCode] = []
        self._welds: List[Weld] = []
        self._qubit_representative: Dict[Location, Location] = {}
        self._welded_qubit_coordinates: List[Location] = []
        self._welded_stabilizers: Dict[Location, Operator] = {}
        self._stabilizer_members: Dict[Location, List[Location]] = {}

        super().__init__(L_x, L_y, L_z)

    @abstractmethod
    def get_subcodes(self) -> List[StabilizerCode]:
        """List of the CSS codes being welded together"""

    @abstractmethod
    def get_welds(self) -> List[Weld]:
        """List of welds, formatted as tuples ``(a, b, qubit_map)``, where
        ``a`` and ``b`` are indices of subcodes and ``qubit_map`` maps the
        location of each qubit of subcode ``a`` on the weld to the location
        of the qubit of subcode ``b`` it is contracted with.
        """

    @property
    def subcodes(self) -> List[StabilizerCode]:
        if not self._subcodes:
            self._subcodes = self.get_subcodes()
        return self._subcodes

    @property
    def welds(self) -> List[Weld]:
        self._ensure_welding()
        return self._welds

    def qubit_representative(self, i_code: int, location: Any) -> Location:
        """Location in the welded code of the qubit ``location`` of the
        subcode ``i_code``"""
        self._ensure_welding()
        return self._qubit_representative[(i_code, location)]

    def stabilizer_members(self, location: Location) -> List[Location]:
        """Stabilizers of the subcodes that form the given welded stabilizer,
        as a list of ``(i, s)`` with ``s`` a stabilizer of the i-th subcode.
        Adopted stabilizers have a single member.
        """
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        return list(self._stabilizer_members[location])

    def welded_operator(self, operators: Dict[int, SubOperator]) -> Operator:
        """Operator of the welded code obtained by applying
        ``operators[i]`` on the i-th subcode.

        The operators must agree on contracted qubits, as is the case for the
        logical operators of the welded code constructed in the paper.
        """
        self._ensure_welding()
        return self._contract_operator(operators)

    def _contract_operator(
        self, operators: Dict[int, SubOperator]
    ) -> Operator:
        welded: Operator = {}

        for i_code, operator in operators.items():
            for qubit, pauli in operator.items():
                representative = self._qubit_representative[(i_code, qubit)]
                if welded.get(representative, pauli) != pauli:
                    raise ValueError(
                        f"Operators disagree on the contracted qubit "
                        f"{representative}"
                    )
                welded[representative] = pauli

        return welded

    def get_qubit_coordinates(self) -> List[Tuple]:
        self._ensure_welding()
        return list(self._welded_qubit_coordinates)

    def get_stabilizer_coordinates(self) -> List[Tuple]:
        self._ensure_welding()
        return list(self._welded_stabilizers.keys())

    def get_stabilizer(self, location: Tuple) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        return dict(self._welded_stabilizers[location])

    def stabilizer_type(self, location: Tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        i_code, sub_location = location
        return self.subcodes[i_code].stabilizer_type(sub_location)

    def qubit_axis(self, location: Tuple) -> str:
        i_code, sub_location = location
        return self.subcodes[i_code].qubit_axis(sub_location)

    def _ensure_welding(self) -> None:
        if self._welding_built:
            return

        if self.weld_pauli not in ('X', 'Z'):
            raise ValueError(
                f"Invalid weld Pauli {self.weld_pauli}, must be 'X' or 'Z'"
            )

        self._welds = self.get_welds()

        self._weld_qubits()
        self._weld_stabilizers()
        self._welding_built = True

    def _weld_qubits(self) -> None:
        """Contract every pair of qubits of the welds into a single qubit"""
        def order(qubit: Location) -> Tuple[int, int]:
            i_code, location = qubit
            return (i_code, self.subcodes[i_code].qubit_index[location])

        parent = {
            (i_code, location): (i_code, location)
            for i_code, code in enumerate(self.subcodes)
            for location in code.qubit_coordinates
        }

        def find(qubit: Location) -> Location:
            while parent[qubit] != qubit:
                parent[qubit] = parent[parent[qubit]]
                qubit = parent[qubit]
            return qubit

        for i_a, i_b, qubit_map in self._welds:
            if i_a == i_b:
                raise ValueError("A subcode cannot be welded to itself")
            if len(set(qubit_map.values())) != len(qubit_map):
                raise ValueError(
                    f"Weld between subcodes {i_a} and {i_b} is not one-to-one"
                )

            for qubit_a, qubit_b in qubit_map.items():
                for i_code, qubit in [(i_a, qubit_a), (i_b, qubit_b)]:
                    if (i_code, qubit) not in parent:
                        raise ValueError(
                            f"Invalid qubit {qubit} for subcode {i_code}"
                        )

                root_a = find((i_a, qubit_a))
                root_b = find((i_b, qubit_b))
                if root_a != root_b:
                    root, child = sorted([root_a, root_b], key=order)
                    parent[child] = root

        self._qubit_representative = {
            qubit: find(qubit) for qubit in parent
        }

        codes_by_class: Dict[Location, set] = {}
        for (i_code, _), representative in self._qubit_representative.items():
            codes_in_class = codes_by_class.setdefault(representative, set())
            if i_code in codes_in_class:
                raise ValueError(
                    f"Welding contracts two qubits of subcode {i_code}"
                )
            codes_in_class.add(i_code)

        self._welded_qubit_coordinates = [
            qubit for qubit in parent
            if self._qubit_representative[qubit] == qubit
        ]

    def _weld_stabilizers(self) -> None:
        def order(stabilizer: Location) -> Tuple[int, int]:
            i_code, location = stabilizer
            return (i_code, self.subcodes[i_code].stabilizer_index[location])

        operators: Dict[Location, SubOperator] = {}
        for i_code, code in enumerate(self.subcodes):
            for location in code.stabilizer_coordinates:
                operator = code.get_stabilizer(location)
                paulis = set(operator.values())
                if len(paulis) != 1 or not paulis.issubset({'X', 'Z'}):
                    raise ValueError(
                        f"Welding requires CSS codes, but stabilizer "
                        f"{location} of subcode {i_code} is {operator}"
                    )
                operators[(i_code, location)] = operator

        welded_type = {
            stabilizer for stabilizer, operator in operators.items()
            if next(iter(operator.values())) == self.weld_pauli
        }

        parent = {stabilizer: stabilizer for stabilizer in welded_type}

        def find(stabilizer: Location) -> Location:
            while parent[stabilizer] != stabilizer:
                parent[stabilizer] = parent[parent[stabilizer]]
                stabilizer = parent[stabilizer]
            return stabilizer

        for i_a, i_b, qubit_map in self._welds:
            restrictions_a = self._weld_restrictions(
                i_a, list(qubit_map.keys()), operators, welded_type
            )
            restrictions_b = self._weld_restrictions(
                i_b, list(qubit_map.values()), operators, welded_type
            )

            if restrictions_a.keys() != restrictions_b.keys():
                raise ValueError(
                    f"Weld between subcodes {i_a} and {i_b} is not well "
                    f"matched"
                )

            for restriction, stabilizer_a in restrictions_a.items():
                root_a = find(stabilizer_a)
                root_b = find(restrictions_b[restriction])
                if root_a != root_b:
                    root, child = sorted([root_a, root_b], key=order)
                    parent[child] = root

        members: Dict[Location, List[Location]] = {}
        for stabilizer in sorted(welded_type, key=order):
            members.setdefault(find(stabilizer), []).append(stabilizer)

        welded_stabilizers: Dict[Location, Operator] = {}
        stabilizer_members: Dict[Location, List[Location]] = {}
        adopted_supports = set()

        for stabilizer in sorted(operators.keys(), key=order):
            i_code, _ = stabilizer
            if stabilizer in welded_type:
                if find(stabilizer) != stabilizer:
                    continue

                codes = [member[0] for member in members[stabilizer]]
                if len(set(codes)) != len(codes):
                    raise ValueError(
                        f"Welding joins several stabilizers of the same "
                        f"subcode into {stabilizer}"
                    )

                welded_stabilizers[stabilizer] = self._contract_operator({
                    member[0]: operators[member]
                    for member in members[stabilizer]
                })
                stabilizer_members[stabilizer] = members[stabilizer]
            else:
                operator = self._contract_operator(
                    {i_code: operators[stabilizer]}
                )
                support = frozenset(operator.items())
                if support in adopted_supports:
                    continue

                adopted_supports.add(support)
                welded_stabilizers[stabilizer] = operator
                stabilizer_members[stabilizer] = [stabilizer]

        self._welded_stabilizers = welded_stabilizers
        self._stabilizer_members = stabilizer_members

    def _weld_restrictions(
        self,
        i_code: int,
        weld_qubits: List[Any],
        operators: Dict[Location, SubOperator],
        welded_type: set,
    ) -> Dict[frozenset, Location]:
        """Restriction to the weld of every generator of Pauli type
        ``weld_pauli`` of the subcode ``i_code`` acting on the weld, written
        as the set of indices of the pairs of the weld it acts on.
        """
        pair_index = {qubit: index for index, qubit in enumerate(weld_qubits)}
        restrictions: Dict[frozenset, Location] = {}

        for stabilizer in welded_type:
            if stabilizer[0] != i_code:
                continue

            restriction = frozenset(
                pair_index[qubit] for qubit in operators[stabilizer]
                if qubit in pair_index
            )
            if restriction:
                if restriction in restrictions:
                    raise ValueError(
                        f"Generators of subcode {i_code} are not linearly "
                        f"independent on the weld"
                    )
                restrictions[restriction] = stabilizer

        matrix = np.zeros((len(restrictions), len(weld_qubits)), dtype='uint8')
        for row, restriction in enumerate(restrictions):
            matrix[row, list(restriction)] = 1

        if len(restrictions) > 0 and brank(matrix) != len(restrictions):
            raise ValueError(
                f"Generators of subcode {i_code} are not linearly "
                f"independent on the weld"
            )

        return restrictions
