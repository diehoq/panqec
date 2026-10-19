from itertools import combinations

import numpy as np
import pytest

from panqec.bpauli import brank, bs_prod
from panqec.bsparse import to_array
from panqec.codes import TetrahedralColor3DCode, TetrahelixCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


def expected_n(d, n_tetrahedra):
    return (
        n_tetrahedra*(d**3 + d)//2 - (n_tetrahedra - 1)*(3*d**2 + 1)//4
    )


class TestTetrahelixCode(StabilizerCodeTest):

    @pytest.fixture(params=[
        (3, 1, 'Z'), (3, 4, 'Z'), (5, 3, 'Z'), (3, 4, 'X'), (5, 3, 'X')
    ])
    def code(self, request):
        d, n_tetrahedra, weld_pauli = request.param
        return TetrahelixCode(d, n_tetrahedra, weld_pauli=weld_pauli)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        TetrahelixCode(4, 2)

    with pytest.raises(ValueError):
        TetrahelixCode(3, 0)

    with pytest.raises(ValueError):
        TetrahelixCode(3, 2, 5)

    with pytest.raises(ValueError):
        TetrahelixCode(3, 2, weld_pauli='Y')


def test_default_number_of_tetrahedra_is_distance():
    code = TetrahelixCode(3)

    assert code.n_tetrahedra == 3
    assert code.params == {'L_x': 3, 'L_y': 3, 'L_z': 3, 'weld_pauli': 'Z'}
    assert TetrahelixCode(**code.params).n == code.n

    code = TetrahelixCode(3, weld_pauli='X')
    assert TetrahelixCode(**code.params).weld_pauli == 'X'


def test_single_tetrahedron_is_tetrahedral_code():
    code = TetrahelixCode(3, 1)
    tetrahedron = TetrahedralColor3DCode(3)

    assert code.n == tetrahedron.n == 15
    assert code.n_stabilizers == tetrahedron.n_stabilizers
    assert np.array_equal(
        code.stabilizer_matrix.toarray(),
        tetrahedron.stabilizer_matrix.toarray(),
    )


@pytest.mark.parametrize("d, n_tetrahedra", [
    (3, 2), (3, 4), (3, 6), (5, 2), (5, 5), (7, 3)
])
def test_code_parameters(d, n_tetrahedra):
    code = TetrahelixCode(d, n_tetrahedra)

    assert code.n == expected_n(d, n_tetrahedra)
    assert code.k == 1
    assert brank(code.stabilizer_matrix) == code.n - 1
    assert code.d == d

    commutators = bs_prod(code.stabilizer_matrix, code.stabilizer_matrix)
    assert np.all(to_array(commutators) == 0)


@pytest.mark.parametrize("d, n_tetrahedra", [
    (3, 2), (3, 4), (3, 6), (5, 2), (5, 5), (7, 3)
])
def test_x_weld_code_parameters(d, n_tetrahedra):
    code = TetrahelixCode(d, n_tetrahedra, weld_pauli='X')

    assert code.n == expected_n(d, n_tetrahedra)
    assert code.k == 1
    assert brank(code.stabilizer_matrix) == code.n - 1

    commutators = bs_prod(code.stabilizer_matrix, code.stabilizer_matrix)
    assert np.all(to_array(commutators) == 0)


@pytest.mark.parametrize("n_tetrahedra, distance", [(2, 3), (3, 3), (4, 5)])
def test_x_weld_distance(n_tetrahedra, distance):
    """The X logical is an edge contracted in three consecutive tetrahedra,
    so the distance grows with the number of tetrahedra"""
    code = TetrahelixCode(3, n_tetrahedra, weld_pauli='X')
    n = code.n
    Hz = to_array(code.Hz)
    logical_z = code.logicals_z[:, n:]

    assert code.d == distance

    for weight in range(1, distance):
        for support in combinations(range(n), weight):
            error = np.zeros(n, dtype='uint8')
            error[list(support)] = 1
            assert (
                np.any(Hz.dot(error) % 2)
                or not np.any(logical_z.dot(error) % 2)
            )


def test_x_weld_welds_faces():
    code = TetrahelixCode(3, 4, weld_pauli='X')

    welded_paulis = {
        location[1][0] for location in code.stabilizer_coordinates
        if code.is_welded(location)
    }
    assert welded_paulis == {'X'}

    for location in code.stabilizer_coordinates:
        if location[1][0] == 'Z':
            assert len(code.stabilizer_members(location)) == 1


@pytest.mark.parametrize("weld_pauli", ['X', 'Z'])
def test_weld_is_highlighted_in_representations(weld_pauli):
    code = TetrahelixCode(3, 3, weld_pauli=weld_pauli)

    for qubit in code.qubit_coordinates:
        rep = code.qubit_representation(qubit)
        assert ('weld' in rep) == code.is_contracted(qubit)

    n_contracted = sum(map(code.is_contracted, code.qubit_coordinates))
    # Two shared facets of 7 qubits meeting on an edge of 3 qubits
    assert n_contracted == 2*7 - 3

    n_welded = 0
    for location in code.stabilizer_coordinates:
        rep = code.stabilizer_representation(location)
        assert ('weld' in rep) == code.is_welded(location)
        if code.is_welded(location):
            assert location[1][0] == weld_pauli
            n_welded += 1

    assert n_welded > 0


@pytest.mark.parametrize("n_tetrahedra", [2, 3, 4, 5])
def test_no_logical_of_weight_below_three(n_tetrahedra):
    code = TetrahelixCode(3, n_tetrahedra)
    n = code.n
    Hx = to_array(code.Hx)
    Hz = to_array(code.Hz)
    logical_x = code.logicals_x[:, :n]
    logical_z = code.logicals_z[:, n:]

    for weight in [1, 2]:
        for support in combinations(range(n), weight):
            error = np.zeros(n, dtype='uint8')
            error[list(support)] = 1

            is_x_logical = (
                not np.any(Hz.dot(error) % 2)
                and np.any(logical_z.dot(error) % 2)
            )
            is_z_logical = (
                not np.any(Hx.dot(error) % 2)
                and np.any(logical_x.dot(error) % 2)
            )
            assert not is_x_logical and not is_z_logical


def test_shared_facets_are_contracted():
    code = TetrahelixCode(3, 5)
    tetrahedron = code.tetrahedron

    for i in range(code.n_tetrahedra - 1):
        facet = code.facet(tetrahedron, code.shared_color(i))
        assert len(facet) == 7

        for qubit in facet:
            assert (
                code.qubit_representative(i + 1, qubit)
                == code.qubit_representative(i, qubit)
            )

    # Consecutive shared facets have different colors, so that every corner
    # of the helix is shared by at most four tetrahedra
    colors = [code.shared_color(i) for i in range(4)]
    assert len(set(colors)) == 4

    sharing = {}
    for i in range(code.n_tetrahedra):
        for qubit in tetrahedron.qubit_coordinates:
            representative = code.qubit_representative(i, qubit)
            sharing.setdefault(representative, set()).add(i)

    assert max(len(tetrahedra) for tetrahedra in sharing.values()) == 4


def test_cells_touching_shared_facets_are_welded():
    code = TetrahelixCode(5, 3)
    tetrahedron = code.tetrahedron

    for location in code.stabilizer_coordinates:
        members = code.stabilizer_members(location)
        operator = code.get_stabilizer(location)

        if location[1][0] == 'X':
            assert len(members) == 1
            continue

        # The welded cell is the union of cells of the same color with the
        # same restriction to each shared facet
        assert len({tetrahedron.stabilizer_type(m[1]) for m in members}) == 1
        expected_support = {
            code.qubit_representative(i, qubit)
            for i, sub_location in members
            for qubit in tetrahedron.get_stabilizer(sub_location)
        }
        assert set(operator.keys()) == expected_support

        for (i, cell), (j, other_cell) in zip(members, members[1:]):
            assert j == i + 1 and cell == other_cell
            shared_boundary = tetrahedron.boundary_vertices[
                code.shared_color(i)
            ]
            assert tetrahedron._edge(cell[1], shared_boundary) in \
                tetrahedron.edge_to_tetras


def test_welded_code_is_paper_merge_with_contracted_pairs():
    """Rebuild the tetrahelix of Paletta et al., where qubits of shared
    facets are kept as pairs with a weight-two stabilizer, and check that it
    encodes one qubit like its contracted version"""
    code = TetrahelixCode(3, 4)
    tetrahedron = code.tetrahedron
    n_sub = tetrahedron.n
    n = n_sub*code.n_tetrahedra

    def index(i, qubit):
        return i*n_sub + tetrahedron.qubit_index[qubit]

    rows_x, rows_z = [], []
    for i in range(code.n_tetrahedra - 1):
        for qubit in code.facet(tetrahedron, code.shared_color(i)):
            row = np.zeros(n, dtype='uint8')
            row[[index(i, qubit), index(i + 1, qubit)]] = 1
            rows_x.append(row)

    for location in code.stabilizer_coordinates:
        row = np.zeros(n, dtype='uint8')
        members = code.stabilizer_members(location)
        for i, sub_location in members:
            for qubit in tetrahedron.get_stabilizer(sub_location):
                row[index(i, qubit)] = 1
        (rows_x if location[1][0] == 'X' else rows_z).append(row)

    # Faces in a shared facet are only kept once by the welding, add the
    # copies of the other side
    for i in range(1, code.n_tetrahedra):
        for location in tetrahedron.stabilizer_coordinates:
            if location[0] != 'X':
                continue
            if (i, location) not in code.stabilizer_index:
                row = np.zeros(n, dtype='uint8')
                for qubit in tetrahedron.get_stabilizer(location):
                    row[index(i, qubit)] = 1
                rows_x.append(row)

    Hx = np.array(rows_x)
    Hz = np.array(rows_z)

    assert np.all(Hx.dot(Hz.T) % 2 == 0)
    assert n - brank(Hx) - brank(Hz) == 1


@pytest.mark.parametrize("d, n_tetrahedra", [(3, 4), (5, 2)])
def test_visual_representations_follow_helix(d, n_tetrahedra):
    code = TetrahelixCode(d, n_tetrahedra)

    edge_lengths = {
        round(float(np.linalg.norm(
            code.helix_vertex(j) - code.helix_vertex(j + shift)
        )), 8)
        for j in range(n_tetrahedra)
        for shift in [1, 2, 3]
    }
    assert len(edge_lengths) == 1

    positions = []
    for qubit in code.qubit_coordinates:
        rep = code.qubit_representation(qubit)
        assert rep['object'] == 'sphere'
        assert len(rep['location']) == 3
        positions.append(tuple(np.round(rep['location'], 8)))

    assert len(set(positions)) == code.n

    for location in code.stabilizer_coordinates:
        rep = code.stabilizer_representation(location)
        n_vertices = len(rep['params']['vertices']) // 3

        assert rep['object'] == 'polyhedron'
        assert rep['type'] == code.stabilizer_type(location)
        assert len(rep['location']) == 3
        assert max(rep['params']['triangles']) < n_vertices
        assert max(rep['params']['edges']) < n_vertices


class BrokenWeldTetrahelixCode(TetrahelixCode):
    """Tetrahelix where the qubits of the shared facet are permuted, so that
    the cells on each side do not match anymore"""

    def get_welds(self):
        i_a, i_b, qubit_map = super().get_welds()[0]
        qubits = list(qubit_map)

        return [(i_a, i_b, dict(zip(qubits, qubits[1:] + qubits[:1])))]


class SelfWeldTetrahelixCode(TetrahelixCode):

    def get_welds(self):
        return [(0, 0, {})]


class NonInjectiveWeldTetrahelixCode(TetrahelixCode):

    def get_welds(self):
        qubit_a, qubit_b = self.tetrahedron.qubit_coordinates[:2]

        return [(0, 1, {qubit_a: qubit_a, qubit_b: qubit_a})]


@pytest.mark.parametrize("code_class", [
    BrokenWeldTetrahelixCode,
    SelfWeldTetrahelixCode,
    NonInjectiveWeldTetrahelixCode,
])
def test_invalid_welds_are_rejected(code_class):
    with pytest.raises(ValueError):
        code_class(3, 2)
