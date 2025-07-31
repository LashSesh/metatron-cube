"""
Public API for the :mod:`metatron_cube.src` package.

This package provides a modular, extensible implementation of the
Metatron Cube.  The core concepts exposed here are:

* :class:`~metatron_cube.src.geometry.Node` – dataclass representing a
  single node with index, label, type and coordinates.
* :func:`~metatron_cube.src.geometry.canonical_nodes` – canonical list of
  the 13 nodes as defined in the blueprint【256449862750268†L940-L1034】.
* :func:`~metatron_cube.src.geometry.canonical_edges` – partial edge list
  capturing the fundamental connections【256449862750268†L1183-L1194】.
* :class:`~metatron_cube.src.graph.MetatronCubeGraph` – graph
  representation with adjacency matrix generation, permutation and
  simple graph operations.
* Higher‑level modules such as :mod:`~metatron_cube.src.master_agent`,
  :mod:`~metatron_cube.src.qlogic` and :mod:`~metatron_cube.src.monolith`
  build on these structures to implement resonant AI architectures.

The package is designed to be imported as a namespace; end users
should access specific submodules directly.
"""

from .geometry import (
    Node,
    canonical_nodes,
    canonical_edges,
    complete_canonical_edges,
    find_node,
)
from .geometry import full_edge_list
from .graph import MetatronCubeGraph
from .field_vector import FieldVector
from .gabriel_cell import GabrielCell
from .mandorla import MandorlaField
from .seraphic_feedback import SeraphicFeedbackModule
from .spiralmemory import SpiralMemory
from .history import HistoryLogger
from .qlogic import QLogicEngine
from .monolith import TripolarOperator, OphanKernel, MonolithDecision
from .master_agent import MasterAgent
from .symmetries import (
    generate_s7_permutations,
    permutation_matrix,
    generate_c6_subgroup,
    generate_d6_subgroup,
    generate_symmetric_group,
    generate_alternating_group,
)

# Quantum state and operator classes for Hilbert‑space functionality
from .quantum import QuantumState, QuantumOperator

__all__ = [
    "Node", "canonical_nodes", "canonical_edges", "find_node",
    "complete_canonical_edges",
    "full_edge_list",
    "MetatronCubeGraph",
    "FieldVector",
    "GabrielCell",
    "MandorlaField",
    "SeraphicFeedbackModule",
    "SpiralMemory",
    "HistoryLogger",
    "QLogicEngine",
    "TripolarOperator", "OphanKernel", "MonolithDecision",
    "MasterAgent",
    "generate_s7_permutations", "permutation_matrix",
    "generate_c6_subgroup", "generate_d6_subgroup",
    "generate_symmetric_group", "generate_alternating_group",

    # Quantum exports
    "QuantumState", "QuantumOperator",
]
