from itertools import combinations

import numpy as np
import pytest

from panqec.bpauli import brank, bs_prod
from panqec.bsparse import to_array
from panqec.codes import TetrahedralColor3DCode
from tests.codes.stabilizer_code_test import StabilizerCodeTest


def expected_n_dual_vertices(d):
    return ((d - 1)*(d + 1)*(d + 3))//12 + 4


def expected_n_dual_edges(d):
    return ((d - 1)*(7*d*d + 10*d + 15))//12 + 6


def barycentric_weights(point, corners):
    colors = TetrahedralColor3DCode._COLORS
    matrix = np.vstack([
        np.column_stack([corners[color] for color in colors]),
        np.ones(len(colors)),
    ])
    rhs = np.append(point, 1)

    return np.linalg.solve(matrix, rhs)


def qubits_by_position(code):
    return {
        tuple(np.round(code.qubit_representation(qubit)['location'], 8)): qubit
        for qubit in code.qubit_coordinates
    }


@pytest.fixture(params=[3, 5, 7, 9])
def code(request):
    return TetrahedralColor3DCode(request.param)


class TestTetrahedralColor3DCode(StabilizerCodeTest):

    @pytest.fixture(params=[3, 5])
    def code(self, request):
        return TetrahedralColor3DCode(request.param)


def test_invalid_distances_are_rejected():
    with pytest.raises(ValueError):
        TetrahedralColor3DCode(2)

    with pytest.raises(ValueError):
        TetrahedralColor3DCode(4)

    with pytest.raises(ValueError):
        TetrahedralColor3DCode(5, 5, 7)


def test_dual_lattice_counts_match_paper(code):
    d = code.distance
    n_vertices = expected_n_dual_vertices(d)
    n_edges = expected_n_dual_edges(d)

    assert len(code.dual_vertices) == n_vertices
    assert len(code.dual_edges) == n_edges
    assert len(code.dual_faces) == d**3 + d + 2
    assert len(code.dual_tetrahedra) == (d**3 + d)//2
    assert code.n == (d**3 + d)//2
    assert code.n_stabilizers == (n_edges - 6) + (n_vertices - 4)
    assert code.n_gauges == code.n_stabilizers


def test_stabilizer_boundary_rules(code):
    boundaries = list(code.boundary_vertices.values())

    for boundary in boundaries:
        assert ('Z', boundary) not in code.stabilizer_index

    for boundary_a, boundary_b in combinations(boundaries, 2):
        edge = code._edge(boundary_a, boundary_b)
        assert ('X', edge) not in code.stabilizer_index

    boundary_adjacent_edges = [
        edge for edge in code.dual_edges
        if sum(vertex in boundaries for vertex in edge) == 1
    ]
    assert boundary_adjacent_edges
    for edge in boundary_adjacent_edges:
        assert ('X', edge) in code.stabilizer_index


def test_stabilizer_locations_are_faces_and_cells(code):
    x_locations = [
        simplex for pauli, simplex in code.stabilizer_coordinates
        if pauli == 'X'
    ]
    z_locations = [
        simplex for pauli, simplex in code.stabilizer_coordinates
        if pauli == 'Z'
    ]

    assert len(x_locations) == len(code.dual_edges) - 6
    assert all(not code._is_border_edge(edge) for edge in x_locations)
    assert len(z_locations) == len(code.dual_vertices) - 4
    assert all(not code._is_boundary_vertex(vertex) for vertex in z_locations)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_stabilizer_algebra(d):
    code = TetrahedralColor3DCode(d)

    assert code.gauge_matrix.shape == (code.n_gauges, 2*code.n)
    assert code.stabilizer_matrix.shape == (code.n_stabilizers, 2*code.n)
    assert np.array_equal(
        code.gauge_matrix.toarray(),
        code.stabilizer_matrix.toarray(),
    )
    assert brank(code.stabilizer_matrix) == code.n - 1

    stabilizer_commutators = bs_prod(
        code.stabilizer_matrix, code.stabilizer_matrix
    )

    assert np.all(to_array(stabilizer_commutators) == 0)


@pytest.mark.parametrize("d", [3, 5])
def test_stabilizers_match_primal_face_and_cell_supports(d):
    code = TetrahedralColor3DCode(d)

    for location in code.stabilizer_coordinates:
        pauli, simplex = location
        if pauli == 'X':
            expected_support = code.edge_to_tetras[simplex]
        else:
            expected_support = code.vertex_to_tetras[simplex]

        expected_operator = {tetra: pauli for tetra in expected_support}
        assert code.get_stabilizer(location) == expected_operator

        direct_bsf = code.to_bsf(expected_operator)
        row = code.stabilizer_matrix[code.stabilizer_index[location]]
        assert np.array_equal(row.toarray()[0], direct_bsf)


def test_logicals_have_expected_boundary_supports(code):
    d = code.distance
    m = (d - 1)//2
    boundaries = code.boundary_vertices
    logical_x = code.get_logicals_x()[0]
    logical_z = code.get_logicals_z()[0]

    # X on the edge between the blue and yellow facets, Z on the red facet
    assert len(logical_x) == d
    for tetra in logical_x:
        assert boundaries['blue'] in tetra.vertices
        assert boundaries['yellow'] in tetra.vertices

    assert len(logical_z) == 3*m*(m + 1) + 1
    for tetra in logical_z:
        assert boundaries['red'] in tetra.vertices

    logical_commutator = bs_prod(code.logicals_x, code.logicals_z)
    assert np.array_equal(to_array(logical_commutator), np.eye(1))

    x_stabilizer_commutators = bs_prod(
        code.logicals_x, code.stabilizer_matrix
    )
    z_stabilizer_commutators = bs_prod(
        code.logicals_z, code.stabilizer_matrix
    )
    assert np.all(to_array(x_stabilizer_commutators) == 0)
    assert np.all(to_array(z_stabilizer_commutators) == 0)


def test_distance_is_d(code):
    """code.d only looks at the logicals given, so the X logical must be one
    of the lightest (an edge of the tetrahedron, not a facet)"""
    assert code.d == code.distance


def test_no_logical_of_weight_below_three():
    code = TetrahedralColor3DCode(3)
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


def test_primal_qubit_positions_form_tetrahedron(code):
    colors = TetrahedralColor3DCode._COLORS
    corners = code._regular_tetrahedron_corners()
    positions = []

    for tetra in code.qubit_coordinates:
        point = np.array(code.qubit_representation(tetra)['location'])
        positions.append(tuple(np.round(point, 8)))
        weights = barycentric_weights(point, corners)
        boundary_colors = {
            vertex[1]
            for vertex in tetra.vertices
            if len(vertex) == 2 and vertex[0] == 'boundary'
        }

        assert np.isclose(np.sum(weights), 1)
        assert np.all(weights >= -1e-9)

        for color in boundary_colors:
            color_index = colors.index(color)
            assert weights[color_index] == pytest.approx(0)

        if len(boundary_colors) == 3:
            missing_color = next(
                color for color in colors if color not in boundary_colors
            )
            assert point == pytest.approx(corners[missing_color])

    assert len(set(positions)) == code.n


def test_distance_three_primal_layout_is_reed_muller_tetrahedron():
    code = TetrahedralColor3DCode(3)
    corners = code._regular_tetrahedron_corners()
    actual_weights = {
        tuple(np.round(
            barycentric_weights(
                code.qubit_representation(tetra)['location'],
                corners,
            ),
            8,
        ))
        for tetra in code.qubit_coordinates
    }
    expected_weights = set()

    n_colors = len(TetrahedralColor3DCode._COLORS)
    for subset_size in range(1, n_colors + 1):
        for subset in combinations(range(n_colors), subset_size):
            weights = [0.0]*len(TetrahedralColor3DCode._COLORS)
            for index in subset:
                weights[index] = 1/subset_size
            expected_weights.add(tuple(np.round(weights, 8)))

    assert actual_weights == expected_weights


@pytest.mark.parametrize("d", [3, 5])
def test_primal_visual_representations_are_generated(d):
    code = TetrahedralColor3DCode(d)

    qubit_rep = code.qubit_representation(code.qubit_coordinates[0])
    assert qubit_rep['object'] == 'sphere'
    assert len(qubit_rep['location']) == 3

    seen_cell_colors = set()
    expected_cell_colors = set(TetrahedralColor3DCode._COLORS)

    for location in code.stabilizer_coordinates:
        pauli, simplex = location
        rep = code.stabilizer_representation(location)
        assert rep['object'] == 'polyhedron'
        assert len(rep['location']) == 3

        if pauli == 'X':
            assert rep['type'] == 'face-x'
            assert rep['color']['activated'] == code.colormap['pink']
            assert len(rep['params']['vertices']) >= 3*3
            assert len(rep['params']['triangles']) >= 1*3
            assert len(rep['params']['edges']) > 0
        else:
            color = simplex[0]
            seen_cell_colors.add(color)
            assert rep['type'] == f'cell-{color}-z'
            color_name = 'gold' if color == 'yellow' else color
            assert rep['color']['activated'] == code.colormap[color_name]
            assert rep['color']['deactivated'] == code.colormap[color_name]
            assert len(rep['params']['vertices']) >= 8*3
            assert len(rep['params']['triangles']) >= 12*3
            assert len(rep['params']['edges']) > 0

    assert seen_cell_colors == expected_cell_colors


@pytest.mark.parametrize("d", [3, 5, 7, 9])
def test_stabilizer_meshes_follow_primal_edges(d):
    """The edges of every mesh are the edges of the primal face or cell, i.e.
    they join qubits whose dual tetrahedra share a triangle, so that no face
    is drawn as a self-intersecting polygon"""
    code = TetrahedralColor3DCode(d)
    qubit_at = qubits_by_position(code)

    for location in code.stabilizer_coordinates:
        params = code.stabilizer_representation(location)['params']
        qubits = [
            qubit_at[tuple(np.round(vertex, 8))]
            for vertex in np.reshape(params['vertices'], (-1, 3))
        ]
        support = code.get_stabilizer(location)
        assert sorted(qubits) == sorted(support)

        mesh_edges = {
            frozenset((qubits[a], qubits[b]))
            for a, b in np.reshape(params['edges'], (-1, 2))
        }
        primal_edges = {
            frozenset((tetra_a, tetra_b))
            for tetra_a, tetra_b in combinations(support, 2)
            if len(set(tetra_a.vertices) & set(tetra_b.vertices)) == 3
        }
        assert mesh_edges == primal_edges


@pytest.mark.parametrize("d", [3, 5, 9, 13])
def test_cell_meshes_are_closed_and_oriented_outwards(d):
    """Every cell is drawn as a closed surface with all its triangles facing
    outwards, including faces whose first vertices are collinear and cells
    that are far from convex"""
    code = TetrahedralColor3DCode(d)

    for location in code.stabilizer_coordinates:
        if location[0] != 'Z':
            continue

        params = code.stabilizer_representation(location)['params']
        vertices = np.reshape(params['vertices'], (-1, 3))
        triangles = np.reshape(params['triangles'], (-1, 3)).tolist()

        # Every edge is crossed once in each direction
        directed_edges = [
            (triangle[i], triangle[(i + 1) % 3])
            for triangle in triangles
            for i in range(3)
        ]
        assert len(set(directed_edges)) == len(directed_edges)
        assert set(directed_edges) == {(b, a) for a, b in directed_edges}

        volume = sum(
            np.dot(vertices[a], np.cross(vertices[b], vertices[c]))
            for a, b, c in triangles
        ) / 6
        assert volume > 0


def test_stabilizer_locations_are_centers_of_their_supports(code):
    """Stabilizers are located at the center of their qubits, in the same
    frame as the qubits and the meshes"""
    positions = {
        qubit: code.qubit_representation(qubit)['location']
        for qubit in code.qubit_coordinates
    }

    for location in code.stabilizer_coordinates:
        center = np.mean(
            [positions[qubit] for qubit in code.get_stabilizer(location)],
            axis=0,
        )
        rep = code.stabilizer_representation(location)
        assert rep['location'] == pytest.approx(center.tolist())
