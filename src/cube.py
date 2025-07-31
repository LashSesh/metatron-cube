"""
cube.py
-------

High‑level API wrapper around the Metatron Cube graph, symmetry operators
and serialization.  The :class:`MetatronCube` class exposes methods to
query nodes and edges, list elements by type, apply permutations,
enumerate symmetry groups, and export/validate configurations.  It
implements many of the features outlined in Section 5 of the blueprint
【265925364547942†L1549-L1594】.

This class is designed as a convenience layer over the underlying
data structures (`MetatronCubeGraph`, `Node`, etc.) and can be used
directly in applications or as a basis for a REST API.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from .geometry import Node, canonical_nodes, canonical_edges, complete_canonical_edges, full_edge_list, find_node
from .graph import MetatronCubeGraph
from .symmetries import (
    generate_s7_permutations,
    generate_c6_subgroup,
    generate_d6_subgroup,
    generate_symmetric_group,
    generate_alternating_group,
    permutation_matrix,
)
from .quantum import QuantumState, QuantumOperator
from .api import export_nodes_json, export_edges_json, export_adjacency_json, export_group_json, export_matrices_json


class MetatronCube:
    """High‑level API class representing a Metatron Cube instance.

    Parameters
    ----------
    nodes : Iterable[Node], optional
        List of nodes to initialise the cube with.  Defaults to the
        canonical node list.
    edges : Iterable[Tuple[int, int]], optional
        Edge list to use.  If ``None``, uses the canonical edge list.
        Pass ``full_edge_list()`` for a fully connected graph.
    operators : Dict[str, Tuple[int, ...]], optional
        Initial mapping of operator IDs to permutations.  This allows
        preloading of symmetry groups.
    """

    def __init__(self,
                 nodes: Optional[Iterable[Node]] = None,
                 edges: Optional[Iterable[Tuple[int, int]]] = None,
                 operators: Optional[Dict[str, Tuple[int, ...]]] = None,
                 full_edges: bool = False) -> None:
        """Create a new Metatron Cube instance.

        Parameters
        ----------
        nodes : Iterable[Node], optional
            Node list to initialise with.  If omitted, the canonical node list
            is used.
        edges : Iterable[Tuple[int, int]], optional
            Explicit edge list to use.  If omitted and ``full_edges`` is False,
            the canonical partial edge list is used.  If ``full_edges`` is
            True, the complete canonical edge list (78 edges) is used.
        operators : Dict[str, Tuple[int, ...]], optional
            Predefined operators mapping IDs to 13‑length permutations.
        full_edges : bool, optional
            If True and no custom ``edges`` are provided, initialise the graph
            with the full 78‑edge connectivity (complete Metatron Cube) via
            :func:`geometry.complete_canonical_edges`.  Defaults to False.
        """
        self.nodes: List[Node] = list(nodes) if nodes is not None else canonical_nodes()
        # Determine edge list: explicit edges override full_edges switch
        if edges is not None:
            self.edges: List[Tuple[int, int]] = list(edges)
        else:
            if full_edges:
                # import lazily to avoid circular import at module top level
                from .geometry import complete_canonical_edges
                self.edges = list(complete_canonical_edges())
            else:
                self.edges = list(canonical_edges())
        self.graph = MetatronCubeGraph(nodes=self.nodes, edges=self.edges)
        # Operator registry: maps string IDs to permutation tuples (length 13)
        self.operators: Dict[str, Tuple[int, ...]] = operators or {}
        # Prepopulate with basic groups if empty
        if not self.operators:
            self._register_basic_groups()

        # Build solid membership mapping for nodes
        self._init_solid_membership()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _register_basic_groups(self) -> None:
        """Register a set of basic symmetry operators into the registry.

        Includes the six rotations of C6 and the six reflections of D6, each
        given a descriptive key.  Permutations are extended to length 13
        by fixing nodes 8–13.
        """
        # C6 rotations: six elements
        for k, perm7 in enumerate(generate_c6_subgroup()):
            # Extend to full 13 permutation
            sigma = tuple(list(perm7) + list(range(8, 14)))
            self.operators[f"C6_rot_{k*60}"] = sigma
        # D6 reflections: indices 0..5 correspond to axis through H2..H7
        for idx, perm7 in enumerate(generate_d6_subgroup()[6:]):  # first 6 are rotations
            sigma = tuple(list(perm7) + list(range(8, 14)))
            self.operators[f"D6_ref_H{idx+2}"] = sigma

    # ------------------------------------------------------------------
    # Solid definitions and membership
    # ------------------------------------------------------------------
    # Define default node subsets for embedded Platonic solids.  These are
    # heuristic selections of node indices forming the vertices of each solid.
    # If the geometry is redefined, adjust these tuples accordingly.
    _SOLID_SETS: Dict[str, List[Tuple[int, ...]]] = {
        # Tetrahedra: two sample sets of four nodes (mixed hex/cube) that
        # approximate tetrahedra within the Metatron structure.  Users can
        # override or extend this via API.
        "tetrahedron": [
            (2, 4, 6, 8),  # three hex vertices and one cube corner
            (3, 5, 7, 9),  # alternate pattern
        ],
        # Cube: the six cube corner nodes.  The canonical cube would have 8
        # vertices; since only six are present, this represents a projected
        # hexahedron within the Metatron Cube.
        "cube": [
            (8, 9, 10, 11, 12, 13),
        ],
        # Octahedron: the six hexagon nodes.  These naturally form the
        # vertices of an octahedron around the centre.
        "octahedron": [
            (2, 3, 4, 5, 6, 7),
        ],
        # Icosahedron: the 12 non‑centre nodes.  In the Metatron Cube, the
        # icosahedron can be approximated by combining hexagon and cube
        # vertices.  This yields 12 vertices; the centre is excluded.
        "icosahedron": [
            tuple(range(2, 14)),
        ],
        # Dodecahedron: a heuristic set including the centre and 11 outer
        # nodes.  A true dodecahedron has 20 vertices; with only 13 nodes
        # available, we approximate it by omitting one cube corner.  Users
        # can define their own sets for strict modelling.
        "dodecahedron": [
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13),
        ],
    }

    def _init_solid_membership(self) -> None:
        """Initialise node‑to‑solid membership mapping.

        Builds a dictionary mapping each node index to a list of solid names it
        belongs to.  This includes the basic types (center, hexagon, cube) as
        well as the heuristic platonic solids defined in ``_SOLID_SETS``.  The
        result is stored in ``self.node_membership`` and used by
        :func:`get_node` and other methods.
        """
        membership: Dict[int, List[str]] = {}
        # basic memberships by node type
        for node in self.nodes:
            if node.type == "center":
                membership.setdefault(node.index, []).extend(["center"])
            elif node.type == "hexagon":
                membership.setdefault(node.index, []).extend(["hexagon"])
            elif node.type == "cube":
                membership.setdefault(node.index, []).extend(["cube"])
        # assign platonic solids
        for solid_name, subsets in self._SOLID_SETS.items():
            for subset in subsets:
                for idx in subset:
                    membership.setdefault(idx, []).append(solid_name)
        # deduplicate memberships
        for idx in membership:
            membership[idx] = sorted(set(membership[idx]))
        self.node_membership = membership

    # ------------------------------------------------------------------
    # Solid API methods
    # ------------------------------------------------------------------
    def list_solids(self) -> List[str]:
        """Return the names of all predefined platonic solids.

        Users can inspect these names and query further details via
        :func:`get_solid_nodes` and :func:`get_solid_edges`.
        """
        return list(self._SOLID_SETS.keys())

    def get_solid_nodes(self, name: str) -> Optional[List[List[int]]]:
        """Get the node index sets defining a solid.

        Parameters
        ----------
        name : str
            Name of the solid (e.g. "tetrahedron", "cube").  Case
            insensitive.

        Returns
        -------
        list of list of int or None
            A list where each entry is a list of node indices forming one
            instance of the solid.  Returns ``None`` if the name is not
            recognised.
        """
        key = name.lower()
        subsets = self._SOLID_SETS.get(key)
        if subsets is None:
            return None
        return [list(s) for s in subsets]

    def get_solid_edges(self, name: str) -> Optional[List[List[Tuple[int, int]]]]:
        """Get the edge lists for each instance of a solid.

        For each node subset defined in the solid, all possible edges among
        those nodes are returned.  This uses complete connectivity within the
        subset.  If the solid is unknown, ``None`` is returned.
        """
        key = name.lower()
        subsets = self._SOLID_SETS.get(key)
        if subsets is None:
            return None
        edge_sets: List[List[Tuple[int, int]]] = []
        for subset in subsets:
            edges: List[Tuple[int, int]] = []
            sorted_subset = sorted(subset)
            for i_idx, i in enumerate(sorted_subset):
                for j in sorted_subset[i_idx + 1:]:
                    edges.append((i, j))
            edge_sets.append(edges)
        return edge_sets

    def enumerate_solid_group(self, name: str, even_only: bool = False) -> Optional[List[Dict[str, Any]]]:
        """Enumerate the symmetry group of a given solid.

        Parameters
        ----------
        name : str
            Name of the solid (e.g. "tetrahedron", "cube", "octahedron").
        even_only : bool, optional
            If True, generate only even permutations (alternating group).  If
            False, generate the full symmetric group on the solid’s node set.

        Returns
        -------
        list of operator dicts or None
            A list of operator objects as returned by :func:`get_operator`, or
            ``None`` if the solid is not known.
        """
        key = name.lower()
        subsets = self._SOLID_SETS.get(key)
        if not subsets:
            return None
        # For simplicity, we take the first subset to define the symmetry group
        subset = subsets[0]
        if even_only:
            perms = generate_alternating_group(subset, total_n=13)
            group_name = f"A{len(subset)}"
        else:
            perms = generate_symmetric_group(subset, total_n=13)
            group_name = f"S{len(subset)}"
        # Register and return operators
        result: List[Dict[str, Any]] = []
        for idx, perm in enumerate(perms):
            op_id = f"{name.lower()}_{group_name}_elem_{idx}"
            self.operators[op_id] = perm
            op_info = self.get_operator(op_id)
            # Override group name with specific group
            if op_info:
                op_info["group"] = group_name
            result.append(op_info)
        return result

    # ------------------------------------------------------------------
    # Node and edge accessors
    # ------------------------------------------------------------------
    def get_node(self, id_or_label: Union[int, str]) -> Optional[Dict[str, Any]]:
        """Return a node object by index or label.

        The returned dictionary follows the schema suggested in the blueprint
        (id, label, type, coordinates, membership)【265925364547942†L1549-L1564】.  Membership is
        inferred from the node type (e.g. hexagon, cube) and can be
        extended manually.
        """
        node = None
        if isinstance(id_or_label, int):
            node = find_node(self.nodes, index=id_or_label)
        elif isinstance(id_or_label, str):
            node = find_node(self.nodes, label=id_or_label)
        if node is None:
            return None
        # membership list: node may belong to multiple solids (hexagon, cube, tetrahedron, etc.)
        membership = self.node_membership.get(node.index, [])
        return {
            "id": node.index,
            "label": node.label,
            "type": node.type,
            "coordinates": list(node.coords),
            "membership": membership,
        }

    def list_nodes(self, type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all nodes or only those of a given type.

        Parameters
        ----------
        type : str, optional
            Filter by node type ("center", "hexagon", "cube").
        """
        result = []
        for node in self.nodes:
            if type is None or node.type == type:
                result.append(self.get_node(node.index))
        return result

    def get_edge(self, id_or_pair: Union[int, Tuple[int, int]]) -> Optional[Dict[str, Any]]:
        """Return an edge object by index or by node pair.

        The edge dictionary contains id, from, to, label, type and solids.
        """
        edge_pair: Optional[Tuple[int, int]] = None
        if isinstance(id_or_pair, int):
            # index is 1‑based in blueprint; convert to 0‑based list index
            idx = id_or_pair - 1
            if idx < 0 or idx >= len(self.edges):
                return None
            edge_pair = self.edges[idx]
        elif isinstance(id_or_pair, tuple) and len(id_or_pair) == 2:
            # normalise ordering
            pair = tuple(sorted(id_or_pair))
            if pair in [tuple(sorted(e)) for e in self.edges]:
                edge_pair = pair
        if edge_pair is None:
            return None
        i, j = edge_pair
        # Generate a simple label and type information
        n1 = find_node(self.nodes, index=i)
        n2 = find_node(self.nodes, index=j)
        if n1 is None or n2 is None:
            return None
        label = f"{n1.label}--{n2.label}"
        # Determine edge type heuristically based on node types
        if {n1.type, n2.type} == {"hexagon"}:
            edge_type = "hex"
        elif {n1.type, n2.type} == {"cube"}:
            edge_type = "cube"
        elif {n1.type, n2.type} == {"hexagon", "cube"}:
            edge_type = "cross"
        elif {n1.type, n2.type} == {"center", "hexagon"}:
            edge_type = "center-hex"
        elif {n1.type, n2.type} == {"center", "cube"}:
            edge_type = "center-cube"
        else:
            edge_type = "other"
        solids = list(set(self.get_node(i)["membership"]).union(self.get_node(j)["membership"]))
        return {
            "id": self.edges.index(edge_pair) + 1,
            "from": i,
            "to": j,
            "label": label,
            "type": edge_type,
            "solids": solids,
        }

    def list_edges(self, type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all edges or filter by edge type.

        Parameters
        ----------
        type : str, optional
            Filter by edge type ("hex", "cube", "cross", "center-hex", etc.).
        """
        result = []
        for idx, (i, j) in enumerate(self.edges, start=1):
            edge_info = self.get_edge((i, j))
            if edge_info is None:
                continue
            if type is None or edge_info["type"] == type:
                result.append(edge_info)
        return result

    # ------------------------------------------------------------------
    # Operator management and application
    # ------------------------------------------------------------------
    def add_operator(self, operator_id: str, permutation: Tuple[int, ...]) -> None:
        """Add a custom operator to the registry.

        The permutation must be a tuple of length 13 representing a
        permutation of 1..13.
        """
        if len(permutation) != 13 or set(permutation) != set(range(1, 14)):
            raise ValueError("Operator permutation must be a permutation of 1..13")
        self.operators[operator_id] = permutation

    def get_operator(self, operator_id: str) -> Optional[Dict[str, Any]]:
        """Return an operator object by ID."""
        perm = self.operators.get(operator_id)
        if perm is None:
            return None
        mat = permutation_matrix(perm, size=13)
        # Determine group name heuristically
        group = None
        if operator_id.startswith("C6"):
            group = "C6"
        elif operator_id.startswith("D6"):
            group = "D6"
        elif operator_id.startswith("S7"):
            group = "S7"
        return {
            "id": operator_id,
            "group": group,
            "permutation": list(perm),
            "matrix": mat.tolist(),
        }

    def apply_operator(self, operator_id: str, target: Union[str, np.ndarray]) -> Any:
        """Apply an operator to the given target.

        The target can be:
        * ``"adjacency"`` – returns the permuted adjacency matrix;
        * a NumPy vector of length 13 – returns the permuted vector;
        * ``"nodes"`` – returns a list of permuted nodes (maintaining labels).
        """
        perm = self.operators.get(operator_id)
        if perm is None:
            raise KeyError(f"Operator '{operator_id}' not found")
        P = permutation_matrix(perm, size=13)
        if isinstance(target, str) and target == "adjacency":
            A = self.graph.get_adjacency_matrix()
            return P @ A @ P.T
        elif isinstance(target, np.ndarray):
            if target.shape[0] != 13:
                raise ValueError("Vector length must be 13 to match operator dimension")
            return P @ target
        elif isinstance(target, str) and target == "nodes":
            # Return new order of node dicts (labels remain with nodes)
            new_order = [self.get_node(i) for i in perm]
            return new_order
        else:
            raise ValueError("Invalid target for operator application")

    # ------------------------------------------------------------------
    # Quantum state operations
    # ------------------------------------------------------------------
    def get_quantum_operator(self, operator_id: str) -> QuantumOperator:
        """Return a :class:`QuantumOperator` corresponding to a registered operator.

        This helper wraps :class:`~metatron_cube.src.quantum.QuantumOperator` to
        turn a stored permutation into a unitary operator acting on 13‑dimensional
        quantum states.  If the requested operator is not found, a ``KeyError``
        is raised.

        Parameters
        ----------
        operator_id : str
            The key of the operator in the internal registry.

        Returns
        -------
        QuantumOperator
            A quantum operator constructed from the stored permutation.
        """
        perm = self.operators[operator_id]  # may raise KeyError
        return QuantumOperator.from_permutation(perm)

    def apply_operator_to_state(self, operator_id: str, state: QuantumState) -> QuantumState:
        """Apply a registered permutation operator to a quantum state.

        This convenience method retrieves the stored permutation, constructs
        the corresponding :class:`QuantumOperator` and applies it to the
        given :class:`QuantumState`.  A new state is returned; the
        original is left unchanged.

        Parameters
        ----------
        operator_id : str
            The ID of the registered operator.
        state : QuantumState
            The quantum state to transform.

        Returns
        -------
        QuantumState
            The transformed quantum state.
        """
        qop = self.get_quantum_operator(operator_id)
        return state.apply(qop)

    def enumerate_group(self, group_name: str, subset: Optional[Iterable[int]] = None) -> List[Dict[str, Any]]:
        """Enumerate all operators in a given group.

        Supported groups: "C6", "D6", "S7", "S4", "A4", "A5".  For S4/A4/A5 a
        ``subset`` of node indices must be provided.  The returned list
        contains operator objects (id, group, permutation, matrix).
        """
        group_name = group_name.upper()
        perms: List[Tuple[int, ...]] = []
        if group_name == "C6":
            perms7 = generate_c6_subgroup()
            perms = [tuple(list(p) + list(range(8, 14))) for p in perms7]
        elif group_name == "D6":
            perms7 = generate_d6_subgroup()
            perms = [tuple(list(p) + list(range(8, 14))) for p in perms7]
        elif group_name == "S7":
            perms7 = generate_s7_permutations()
            perms = [tuple(list(p) + list(range(8, 14))) for p in perms7]
        elif group_name in {"S4", "A4", "A5"}:
            if subset is None:
                raise ValueError(f"subset must be provided for group {group_name}")
            subset = tuple(subset)
            # Basic validation of subset length
            expected_len = {"S4": 4, "A4": 4, "A5": 5}[group_name]
            if len(subset) != expected_len:
                raise ValueError(f"Group {group_name} requires a subset of length {expected_len}, got {len(subset)}")
            if group_name == "S4":
                perms = generate_symmetric_group(subset, total_n=13)
            elif group_name == "A4":
                perms = generate_alternating_group(subset, total_n=13)
            elif group_name == "A5":
                perms = generate_alternating_group(subset, total_n=13)
        else:
            raise ValueError(f"Unsupported group name {group_name}")
        result = []
        for idx, perm in enumerate(perms):
            op_id = f"{group_name}_elem_{idx}"
            self.operators[op_id] = perm  # register for reuse
            result.append(self.get_operator(op_id))
        return result

    # ------------------------------------------------------------------
    # Serialization and validation
    # ------------------------------------------------------------------
    def serialize(self, format: str = "json", path: Optional[str] = None) -> str:
        """Serialize the full cube (nodes, edges, adjacency, operators).

        Currently only JSON is supported.
        """
        if format.lower() != "json":
            raise ValueError("Only JSON serialization is supported")
        data = {
            "nodes": json.loads(export_nodes_json(self.graph)),
            "edges": json.loads(export_edges_json(self.graph)),
            "adjacency": json.loads(export_adjacency_json(self.graph)),
            "operators": {op_id: list(perm) for op_id, perm in self.operators.items()},
        }
        js = json.dumps(data, indent=2)
        if path:
            with open(path, "w") as f:
                f.write(js)
        return js

    def validate(self, config: Any) -> bool:
        """Validate a user-supplied configuration.

        Accepted configurations:
        * A permutation (tuple of length 13) – returns True if it is a valid
          permutation of 1..13.
        * An adjacency matrix (13×13 NumPy array) – returns True if it is
          symmetric, binary and has zero diagonal.
        """
        if isinstance(config, tuple) and len(config) == 13:
            return set(config) == set(range(1, 14))
        if isinstance(config, np.ndarray) and config.shape == (13, 13):
            if not np.array_equal(config, config.T):
                return False
            if not np.all((config == 0) | (config == 1)):
                return False
            if np.any(np.diag(config) != 0):
                return False
            return True
        return False