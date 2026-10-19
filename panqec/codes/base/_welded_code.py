from typing import Any, Dict, List, Optional, Tuple
from abc import abstractmethod
import json
import os

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
    with the same restriction in the other code). When the generators are
    linearly independent on the weld, the generators above generate the
    welded code (Theorem 3 of the paper). Otherwise, the welded stabilizers of
    Pauli type ``weld_pauli`` are all the products of generators with the
    same restriction on both sides of every weld, and a generating set is
    built from the most local of them.

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

    In the GUI, contracted qubits and welded stabilizers (made of stabilizers
    of several subcodes) can be highlighted, with the style given by the
    ``weld`` entry of the code in ``gui-config.json``.
    """

    weld_pauli: str = 'Z'

    _DEFAULT_WELD_STYLE = {
        'qubit': {'color': 'orange', 'opacity': 1},
        'stabilizer': {'opacity': 0.6},
    }

    def __init__(
        self,
        L_x: int,
        L_y: Optional[int] = None,
        L_z: Optional[int] = None,
        weld_pauli: Optional[str] = None,
    ):
        if weld_pauli is not None:
            self.weld_pauli = weld_pauli

        self._welding_built = False
        self._subcodes: List[StabilizerCode] = []
        self._welds: List[Weld] = []
        self._qubit_representative: Dict[Location, Location] = {}
        self._contracted_qubits: set = set()
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

    def is_contracted(self, location: Location) -> bool:
        """Whether the qubit of the welded code results from the contraction
        of qubits of several subcodes"""
        self._ensure_welding()
        return location in self._contracted_qubits

    def is_welded(self, location: Location) -> bool:
        """Whether the stabilizer of the welded code is made of stabilizers
        of several subcodes"""
        members = self.stabilizer_members(location)
        return len({i_code for i_code, _ in members}) > 1

    def qubit_representation(
        self, location: Tuple, rotated_picture=False, json_file=None
    ) -> dict:
        rep = super().qubit_representation(
            location, rotated_picture, json_file=json_file
        )

        if self.is_contracted(location):
            style = dict(self._weld_style(json_file)['qubit'])
            style['color'] = self.colormap[style['color']]
            rep['weld'] = style

        return rep

    def stabilizer_representation(
        self, location: Tuple, rotated_picture=False, json_file=None
    ) -> dict:
        rep = super().stabilizer_representation(
            location, rotated_picture, json_file=json_file
        )

        if self.is_welded(location):
            rep['weld'] = dict(self._weld_style(json_file)['stabilizer'])

        return rep

    def _weld_style(self, json_file=None) -> dict:
        if json_file is None:
            json_file = os.path.join(
                os.environ['PANQEC_ROOT_DIR'], 'codes', 'gui-config.json'
            )

        with open(json_file, 'r') as f:
            data = json.load(f)

        return data.get(self.id, {}).get('weld', self._DEFAULT_WELD_STYLE)

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

        self._contracted_qubits = {
            representative
            for representative, codes_in_class in codes_by_class.items()
            if len(codes_in_class) > 1
        }

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

        welded_type = sorted(
            (
                stabilizer for stabilizer, operator in operators.items()
                if next(iter(operator.values())) == self.weld_pauli
            ),
            key=order
        )

        restrictions = self._weld_restrictions(operators, welded_type)
        combinations = self._welded_combinations(welded_type, restrictions)

        qubit_index = {
            qubit: index
            for index, qubit in enumerate(self._welded_qubit_coordinates)
        }

        def contract(members: List[Location]) -> Operator:
            sub_operators: Dict[int, SubOperator] = {}
            for i_code, location in members:
                sub_operator = sub_operators.setdefault(i_code, {})
                for qubit, pauli in operators[(i_code, location)].items():
                    if qubit in sub_operator:
                        del sub_operator[qubit]
                    else:
                        sub_operator[qubit] = pauli

            return self._contract_operator(sub_operators)

        # Keep a basis of the welded stabilizers, starting from the most local
        # combinations, and index each of them by one of its members
        basis: Dict[int, int] = {}
        welded: Dict[Location, List[Location]] = {}

        for members in combinations:
            operator = contract(members)
            vector = sum(1 << qubit_index[qubit] for qubit in operator)

            while vector:
                pivot = vector.bit_length() - 1
                if pivot not in basis:
                    break
                vector ^= basis[pivot]

            if not vector:
                continue

            key = next(
                (member for member in members if member not in welded), None
            )
            if key is None:
                raise ValueError(
                    "Could not find a location for every welded stabilizer"
                )

            basis[vector.bit_length() - 1] = vector
            welded[key] = members

        welded_stabilizers: Dict[Location, Operator] = {}
        stabilizer_members: Dict[Location, List[Location]] = {}
        adopted_supports = set()
        welded_set = set(welded_type)

        for stabilizer in sorted(operators.keys(), key=order):
            i_code, _ = stabilizer
            if stabilizer in welded_set:
                if stabilizer not in welded:
                    continue

                members = welded[stabilizer]
                welded_stabilizers[stabilizer] = contract(members)
                stabilizer_members[stabilizer] = members
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
        operators: Dict[Location, SubOperator],
        welded_type: List[Location],
    ) -> Dict[Tuple[Location, int], frozenset]:
        """Restriction to each weld of every generator of Pauli type
        ``weld_pauli``, written as the set of indices of the pairs of the weld
        it acts on, indexed by ``(generator, weld index)``.
        Only non-empty restrictions are kept.
        """
        restrictions: Dict[Tuple[Location, int], frozenset] = {}

        for i_weld, (i_a, i_b, qubit_map) in enumerate(self._welds):
            sides = [
                (i_a, {q: k for k, q in enumerate(qubit_map.keys())}),
                (i_b, {q: k for k, q in enumerate(qubit_map.values())}),
            ]
            side_restrictions = []

            for i_code, pair_index in sides:
                restrictions_code = set()
                for stabilizer in welded_type:
                    if stabilizer[0] != i_code:
                        continue
                    restriction = frozenset(
                        pair_index[qubit] for qubit in operators[stabilizer]
                        if qubit in pair_index
                    )
                    if restriction:
                        restrictions[(stabilizer, i_weld)] = restriction
                        restrictions_code.add(restriction)
                side_restrictions.append(restrictions_code)

            if side_restrictions[0] != side_restrictions[1]:
                raise ValueError(
                    f"Weld between subcodes {i_a} and {i_b} is not well "
                    f"matched"
                )

        return restrictions

    def _welded_combinations(
        self,
        welded_type: List[Location],
        restrictions: Dict[Tuple[Location, int], frozenset],
    ) -> List[List[Location]]:
        """Products of generators of Pauli type ``weld_pauli`` that have the
        same restriction on both sides of every weld, i.e. that give a
        stabilizer of the welded code, as lists of generators.

        Local combinations come first: every generator followed by the
        generators of the other side of the welds with the same restriction
        (the welded stabilizers of the paper when the generators are linearly
        independent on the weld), then a basis of all the valid combinations.
        """
        position = {stabilizer: k for k, stabilizer in enumerate(welded_type)}

        by_restriction: Dict[Tuple[int, int, frozenset], List[Location]] = {}
        for (stabilizer, i_weld), restriction in restrictions.items():
            key = (stabilizer[0], i_weld, restriction)
            by_restriction.setdefault(key, []).append(stabilizer)
        for stabilizers in by_restriction.values():
            stabilizers.sort(key=position.__getitem__)

        def mismatch(
            members: List[Location]
        ) -> Optional[Tuple[int, frozenset]]:
            for i_weld, (i_a, i_b, _) in enumerate(self._welds):
                difference: frozenset = frozenset()
                for stabilizer in members:
                    if stabilizer[0] in (i_a, i_b):
                        difference ^= restrictions.get(
                            (stabilizer, i_weld), frozenset()
                        )
                if difference:
                    return i_weld, difference
            return None

        def close(seed: Location) -> Optional[List[Location]]:
            members = [seed]
            for _ in range(len(welded_type)):
                broken = mismatch(members)
                if broken is None:
                    return sorted(members, key=position.__getitem__)

                i_weld, difference = broken
                i_a, i_b, _ = self._welds[i_weld]
                codes = [stabilizer[0] for stabilizer in members]
                sides = [i_b, i_a] if codes.count(i_b) <= codes.count(i_a) \
                    else [i_a, i_b]

                partner = next((
                    stabilizer
                    for i_code in sides
                    for stabilizer in by_restriction.get(
                        (i_code, i_weld, difference), []
                    )
                    if stabilizer not in members
                ), None)
                if partner is None:
                    return None
                members.append(partner)

            return None

        combinations = []
        for seed in welded_type:
            members = close(seed)
            if members is not None:
                combinations.append(members)

        # Basis of all the valid combinations, in case the local ones do not
        # generate the welded stabilizers
        constraint_index: Dict[Tuple[int, int], int] = {}
        for i_weld, (_, _, qubit_map) in enumerate(self._welds):
            for k in range(len(qubit_map)):
                constraint_index[(i_weld, k)] = len(constraint_index)

        reduced: Dict[int, Tuple[int, int]] = {}
        for stabilizer in welded_type:
            mask = 0
            for i_weld in range(len(self._welds)):
                for k in restrictions.get((stabilizer, i_weld), frozenset()):
                    mask ^= 1 << constraint_index[(i_weld, k)]
            combination = 1 << position[stabilizer]

            while mask:
                pivot = mask.bit_length() - 1
                if pivot not in reduced:
                    break
                pivot_mask, pivot_combination = reduced[pivot]
                mask ^= pivot_mask
                combination ^= pivot_combination

            if mask:
                reduced[mask.bit_length() - 1] = (mask, combination)
            else:
                combinations.append([
                    welded_type[k] for k in range(len(welded_type))
                    if combination >> k & 1
                ])

        return combinations
