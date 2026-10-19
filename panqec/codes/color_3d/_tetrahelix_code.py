from math import acos, sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np

from panqec.codes import StabilizerCode
from panqec.codes.base._welded_code import WeldedCode, Weld
from ._tetrahedral_color_3d_code import TetrahedralColor3DCode, Tetra

Operator = Dict[Tuple, str]  # Location to pauli ('X','Y','Z')


class TetrahelixCode(WeldedCode):
    """Tetrahelix code of Paletta et al. [arXiv:2307.10729].

    A chain of ``L_y`` tetrahedral color codes of distance ``L_x`` (the
    [[15,1,3]] Reed-Muller code for ``L_x = 3``) packed into a
    Boerdijk-Coxeter helix, where consecutive tetrahedra are welded along
    their common facet using the welding operation of Michnicki
    [arXiv:1208.3496].

    Tetrahedron ``i`` has its corners on the vertices ``v_i, ..., v_{i+3}``
    of the helix, and the corner ``v_j`` has color ``COLORS[j % 4]``.
    Tetrahedra ``i`` and ``i+1`` therefore share the facet opposite to
    ``v_i``, which has color ``COLORS[i % 4]`` in both, and every vertex of
    the helix is shared by at most four tetrahedra.

    With the convention of :class:`TetrahedralColor3DCode` (X stabilizers on
    faces, Z stabilizers on cells), each pair of qubits facing each other on a
    shared facet is contracted to a single qubit, every cell touching the
    facet is welded with the cell of the same color on the other side, and
    all the faces are kept. This is the merge of the tetrahelix paper, where
    the weight-two stabilizers of each pair of facet qubits are used to
    contract the pair into a single qubit.

    With ``weld_pauli = 'X'``, the faces touching each shared facet are
    welded instead and the cells are kept. Since the faces are not linearly
    independent on a facet, the welded faces are the products of faces with
    the same restriction on both sides of every shared facet.

    The code encodes one logical qubit in
    ``L_y (d^3 + d)/2 - (L_y - 1)(3 d^2 + 1)/4`` physical qubits. The logical
    of Pauli type ``weld_pauli`` is spread over every tetrahedron (a facet for
    Z, the edge between the first two shared facets for X, which makes the X
    distance grow with ``L_y``), and the other one is supported on the first
    tetrahedron only (an edge for X, a facet for Z).

    Parameters
    ----------
    L_x : int
        Odd distance ``d >= 3`` of each tetrahedral code
    L_y : int, optional
        Number of tetrahedra in the helix (``L_x`` by default)
    L_z : int, optional
        Must be equal to ``L_x`` if given
    weld_pauli : str, optional
        Type of the welds, ``'Z'`` (cells are welded, default) or ``'X'``
        (faces are welded)
    """

    dimension = 3
    deformation_names: List[str] = []

    _COLORS = TetrahedralColor3DCode._COLORS

    # Boerdijk-Coxeter helix with unit edges: v_j = (r cos jθ, r sin jθ, j h)
    _HELIX_ANGLE = acos(-2/3)
    _HELIX_RADIUS = 3*sqrt(3)/10
    _HELIX_RISE = 1/sqrt(10)

    def __init__(
        self,
        L_x: int,
        L_y: Optional[int] = None,
        L_z: Optional[int] = None,
        weld_pauli: str = 'Z',
    ):
        if L_x < 3 or L_x % 2 == 0:
            raise ValueError(
                "TetrahelixCode expects an odd distance d >= 3 as L_x"
            )
        if L_y is None:
            L_y = L_x
        if L_y < 1:
            raise ValueError(
                "TetrahelixCode expects at least one tetrahedron as L_y"
            )
        if L_z is not None and L_z != L_x:
            raise ValueError("TetrahelixCode expects L_z to be equal to L_x")
        if weld_pauli not in ('X', 'Z'):
            raise ValueError("TetrahelixCode expects weld_pauli to be X or Z")

        self._helix_maps: List[np.ndarray] = []

        super().__init__(L_x, L_y, L_x, weld_pauli=weld_pauli)

    @property
    def label(self) -> str:
        return (
            f'Tetrahelix d={self.L_x} ({self.L_y} tetrahedra, '
            f'{self.weld_pauli} weld)'
        )

    @property
    def params(self) -> dict:
        return {**super().params, 'weld_pauli': self.weld_pauli}

    @property
    def distance(self) -> int:
        return self.L_x

    @property
    def n_tetrahedra(self) -> int:
        return self.L_y

    @property
    def tetrahedron(self) -> TetrahedralColor3DCode:
        """Tetrahedral code of which every tetrahedron is a copy"""
        return self.subcodes[0]  # type: ignore[return-value]

    def get_subcodes(self) -> List[StabilizerCode]:
        tetrahedron = TetrahedralColor3DCode(self.distance)

        return [tetrahedron] * self.n_tetrahedra

    def get_welds(self) -> List[Weld]:
        # Every tetrahedron is a copy of the same code, so the reflection
        # through a shared facet leaves the qubit locations unchanged
        return [
            (i, i + 1, {
                qubit: qubit
                for qubit in self.facet(self.tetrahedron, self.shared_color(i))
            })
            for i in range(self.n_tetrahedra - 1)
        ]

    def shared_color(self, i_tetra: int) -> str:
        """Color of the facet shared by tetrahedra ``i_tetra`` and
        ``i_tetra + 1``"""
        return self._COLORS[i_tetra % len(self._COLORS)]

    @staticmethod
    def facet(code: TetrahedralColor3DCode, color: str) -> List[Tetra]:
        """Qubits of the facet of the given color of a tetrahedral code"""
        return code.vertex_to_tetras[code.boundary_vertices[color]]

    @staticmethod
    def edge(
        code: TetrahedralColor3DCode, color_a: str, color_b: str
    ) -> List[Tetra]:
        """Qubits of the edge between two facets of a tetrahedral code"""
        facet_b = set(TetrahelixCode.facet(code, color_b))

        return [
            qubit for qubit in TetrahelixCode.facet(code, color_a)
            if qubit in facet_b
        ]

    def get_logicals_x(self) -> List[Operator]:
        # For X welds, the edge between the first two shared facets is
        # contracted in three consecutive tetrahedra, which gives the lightest
        # representative
        colors = self._COLORS[2:] if self.weld_pauli == 'Z' \
            else self._COLORS[:2]
        edge = self.edge(self.tetrahedron, *colors)

        return [self._logical(edge, 'X')]

    def get_logicals_z(self) -> List[Operator]:
        facet = self.facet(self.tetrahedron, self._COLORS[0])

        return [self._logical(facet, 'Z')]

    def _logical(self, support: List[Tetra], pauli: str) -> Operator:
        """Logical acting on the given support of every tetrahedron if it has
        the Pauli type of the welds, and of the first tetrahedron otherwise"""
        n_copies = self.n_tetrahedra if pauli == self.weld_pauli else 1

        return self.welded_operator({
            i: {qubit: pauli for qubit in support} for i in range(n_copies)
        })

    def qubit_representation(
        self, location: Tuple, rotated_picture=False, json_file=None
    ) -> dict:
        rep = super().qubit_representation(
            location, rotated_picture, json_file=json_file
        )

        i_tetra, qubit = location
        sub_rep = self.tetrahedron.qubit_representation(qubit)
        rep['location'] = self._helix_points(
            i_tetra, [sub_rep['location']]
        )[0].tolist()

        return rep

    def stabilizer_representation(
        self, location: Tuple, rotated_picture=False, json_file=None
    ) -> dict:
        rep = super().stabilizer_representation(
            location, rotated_picture, json_file=json_file
        )

        vertices: List[float] = []
        triangles: List[int] = []
        edges: List[int] = []
        centers = []

        for i_tetra, sub_location in self.stabilizer_members(location):
            sub_rep = self.tetrahedron.stabilizer_representation(sub_location)
            offset = len(vertices) // 3
            points = np.reshape(sub_rep['params']['vertices'], (-1, 3))

            vertices.extend(
                self._helix_points(i_tetra, points).ravel().tolist()
            )
            triangles.extend(
                offset + index for index in sub_rep['params']['triangles']
            )
            edges.extend(
                offset + index for index in sub_rep['params']['edges']
            )
            centers.append(
                self._helix_points(i_tetra, [sub_rep['location']])[0]
            )

        rep['location'] = np.mean(centers, axis=0).tolist()
        rep['params']['vertices'] = vertices
        rep['params']['triangles'] = triangles
        rep['params']['edges'] = edges

        return rep

    def helix_vertex(self, j: int) -> np.ndarray:
        """Position of the j-th vertex of the Boerdijk-Coxeter helix, scaled
        to the size of the tetrahedral code and centered at the origin"""
        corners = self.tetrahedron._regular_tetrahedron_corners()
        edge_length = np.linalg.norm(
            corners[self._COLORS[0]] - corners[self._COLORS[1]]
        )
        center = (self.n_tetrahedra + 2) / 2

        return edge_length * np.array([
            self._HELIX_RADIUS * np.cos(j * self._HELIX_ANGLE),
            self._HELIX_RADIUS * np.sin(j * self._HELIX_ANGLE),
            (j - center) * self._HELIX_RISE,
        ])

    def _helix_points(self, i_tetra: int, points) -> np.ndarray:
        """Map points of the tetrahedral code to the i-th tetrahedron of the
        helix, using the affine map sending each colored corner of the
        tetrahedral code to the helix vertex of the same color"""
        if not self._helix_maps:
            self._helix_maps = [
                self._helix_map(i) for i in range(self.n_tetrahedra)
            ]

        points = np.asarray(points, dtype=float)
        homogeneous = np.hstack([points, np.ones((len(points), 1))])

        return homogeneous @ self._helix_maps[i_tetra].T

    def _helix_map(self, i_tetra: int) -> np.ndarray:
        corners = self.tetrahedron._regular_tetrahedron_corners()
        n_colors = len(self._COLORS)

        source = np.zeros((n_colors, n_colors))
        target = np.zeros((3, n_colors))
        for j in range(i_tetra, i_tetra + n_colors):
            color = self._COLORS[j % n_colors]
            source[:, j - i_tetra] = np.append(corners[color], 1)
            target[:, j - i_tetra] = self.helix_vertex(j)

        return target @ np.linalg.inv(source)
