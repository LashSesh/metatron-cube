
"""
FastAPI‑basierte REST‑API für den Metatron Cube und die Resonanz‑Module.

Dieser Server kombiniert den MasterAgent (SpiralMemory, GabrielCells, MandorlaField)
mit dem hohen Metatron‑Cube‑API.  Es stellt Endpunkte bereit zum
Injizieren von Eingaben, zum Abrufen des Agent‑Zustands und zur
expliziten Abfrage der Metatron‑Cube‑Strukturen (Knoten, Kanten,
Platonsche Solids, Symmetriegruppen).

Die API folgt den Prinzipien aus dem Blueprint: alle Datenstrukturen
sind transparent zugänglich, und Operatoren können auf Adjazenz oder
Vektoren angewendet werden.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Any, Optional
import numpy as np

# Use explicit relative imports so that the FastAPI app can locate the
# package modules when the server is run via ``uvicorn metatron_cube.main:app``.
from .src.master_agent import MasterAgent
from .src.cube import MetatronCube
from .src.qdash_agent import QDASHAgent
from .src.resonance_tensor import ResonanceTensorField
from .src.meta_interpreter import MetaInterpreter
from .src.tensor_network import TensorNetwork
from .src.quantum import QuantumState

class AgentInput(BaseModel):
    data: List[Any]

class OperatorRequest(BaseModel):
    operator_id: str
    target: str  # 'adjacency', 'nodes' or 'vector'
    vector: Optional[List[float]] = None

class GroupRequest(BaseModel):
    group: str
    subset: Optional[List[int]] = None

# Data model for quantum operator application
class QuantumApplyRequest(BaseModel):
    operator_id: str
    amplitudes: List[List[float]]  # list of [real, imag] pairs, length 13

# Data model for quantum state measurement
class QuantumStateRequest(BaseModel):
    amplitudes: List[List[float]]

# Request model for QDASH decision cycle
class QDASHRequest(BaseModel):
    input_vector: List[float]
    max_iter: Optional[int] = 3
    dt: Optional[float] = 1.0

# Request model for resonance tensor update
class ResonanceUpdateRequest(BaseModel):
    dt: float
    input_modulation: Optional[List[List[List[float]]]] = None

app = FastAPI()

# Instantiate core components
# Set Mandorla dynamic threshold coefficients (α and β) to 0.5 by default.
agent = MasterAgent(n_cells=4, mandorla_alpha=0.5, mandorla_beta=0.5)
cube = MetatronCube(full_edges=False)

# Create a global QDASH agent and resonance tensor field for simulation
qdash = QDASHAgent(n_cells=4, alpha=0.5, beta=0.5)
res_field = ResonanceTensorField(shape=(3, 3, 3))
# Create meta interpreter for adaptive control
meta = MetaInterpreter(qdash)

# Initialise a tensor network with the default resonance field
tensor_net = TensorNetwork(fields=[res_field])

@app.post("/inject/agent_state")
def inject_agent_state(input: AgentInput):
    """Inject raw inputs into the MasterAgent and return its new state."""
    result = agent.process(input.data)
    return {"result": result}

@app.get("/get/spiralmemory")
def get_spiral_memory():
    """Return the SpiralMemory history and current memory."""
    memory = agent.spiral_memory.memory
    history = agent.spiral_memory.history
    return {"spiralmemory": str(memory), "history": history}

@app.get("/get/gabrielcells")
def get_gabrielcells():
    """Return current state of all GabrielCells."""
    cells = [
        {"psi": c.psi, "rho": c.rho, "omega": c.omega, "output": c.output}
        for c in agent.gabriel_cells
    ]
    return {"gabriel_cells": cells}

@app.get("/get/mandorla")
def get_mandorla():
    """Return MandorlaField inputs, resonance and decision."""
    return {
        "inputs": [list(v) for v in agent.mandorla.inputs],
        "resonance": agent.mandorla.resonance,
        "decision": agent.last_decision,
        "history": agent.mandorla.history
    }

@app.get("/compute/decision")
def compute_decision():
    """Return the latest decision outcome of the MandorlaField."""
    res = agent.last_decision
    return {"decision": res}

@app.get("/get/full_state")
def get_full_state():
    """Return the full state of the MasterAgent (SpiralMemory, GabrielCells, Mandorla)."""
    return {
        "spiralmemory": [tuple([float(x) for x in pts[0][i]]) if isinstance(pts, tuple) else pts for pts in agent.spiral_memory.memory],
        "gabriel_cells": [
            {"psi": c.psi, "rho": c.rho, "omega": c.omega, "output": c.output}
            for c in agent.gabriel_cells
        ],
        "mandorla": {
            "inputs": [list(v) for v in agent.mandorla.inputs],
            "resonance": agent.mandorla.resonance,
            "decision": agent.last_decision
        }
    }

@app.get("/cube/nodes")
def api_list_nodes(type: Optional[str] = Query(None, description="Filter by type: center, hexagon, cube")):
    """Return all nodes (optionally filtered by type)."""
    return cube.list_nodes(type=type)

@app.get("/cube/edges")
def api_list_edges(type: Optional[str] = Query(None, description="Filter by edge type")):
    """Return all edges (optionally filtered by type)."""
    return cube.list_edges(type=type)

@app.get("/cube/nodes/{identifier}")
def api_get_node(identifier: str):
    """Return a node by id or label."""
    # interpret identifier: if digits → index else label
    try:
        idx = int(identifier)
        node = cube.get_node(idx)
    except ValueError:
        node = cube.get_node(identifier)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return node

@app.get("/cube/edges/{edge_id}")
def api_get_edge(edge_id: int):
    """Return an edge by its 1‑based ID."""
    edge = cube.get_edge(edge_id)
    if edge is None:
        raise HTTPException(status_code=404, detail="Edge not found")
    return edge

@app.get("/cube/solids")
def api_list_solids():
    """Return the names of all predefined platonic solids in the cube."""
    return cube.list_solids()

@app.get("/cube/solids/{solid_name}/nodes")
def api_get_solid_nodes(solid_name: str):
    """Return the node index lists for the specified solid."""
    nodes = cube.get_solid_nodes(solid_name)
    if nodes is None:
        raise HTTPException(status_code=404, detail="Solid not found")
    return nodes

@app.get("/cube/solids/{solid_name}/edges")
def api_get_solid_edges(solid_name: str):
    """Return the edge lists for the specified solid."""
    edges = cube.get_solid_edges(solid_name)
    if edges is None:
        raise HTTPException(status_code=404, detail="Solid not found")
    return edges

@app.post("/cube/operators/apply")
def api_apply_operator(req: OperatorRequest):
    """Apply a registered operator to a target (adjacency, nodes or vector)."""
    op_id = req.operator_id
    target = req.target
    # If vector target, ensure vector is provided
    if target == "vector":
        if req.vector is None:
            raise HTTPException(status_code=400, detail="Vector must be provided for vector target")
        vec = np.array(req.vector, dtype=float)
        if len(vec) != 13:
            raise HTTPException(status_code=400, detail="Vector must have length 13")
        try:
            res = cube.apply_operator(op_id, vec)
        except KeyError:
            raise HTTPException(status_code=404, detail="Operator not found")
        return {"result": res.tolist()}
    else:
        try:
            res = cube.apply_operator(op_id, target)
        except KeyError:
            raise HTTPException(status_code=404, detail="Operator not found")
        return {"result": res if isinstance(res, list) else (res.tolist() if hasattr(res, 'tolist') else res)}

@app.post("/cube/operators/enumerate")
def api_enumerate_group(req: GroupRequest):
    """Enumerate a permutation group on a subset of nodes."""
    try:
        ops = cube.enumerate_group(req.group, subset=req.subset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ops

@app.get("/cube/operators/{op_id}")
def api_get_operator(op_id: str):
    """Return a registered operator by ID."""
    op = cube.get_operator(op_id)
    if op is None:
        raise HTTPException(status_code=404, detail="Operator not found")
    return op

# ---------------------------------------------------------------------------
# Quantum endpoints
# ---------------------------------------------------------------------------
@app.post("/cube/quantum/apply")
def api_apply_quantum_operator(req: QuantumApplyRequest):
    """Apply a registered permutation operator to a quantum state.

    The request body must include ``operator_id`` (the key of a registered
    operator) and ``amplitudes`` – a list of 13 pairs ``[real, imag]``
    describing the initial quantum state.  The state is normalised on
    creation.  The endpoint returns the transformed state as a list of
    ``[real, imag]`` pairs.
    """
    # convert amplitude pairs into complex numbers
    if len(req.amplitudes) != 13:
        raise HTTPException(status_code=400, detail="Quantum amplitudes must have length 13")
    try:
        amps = [complex(a[0], a[1]) for a in req.amplitudes]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid amplitude format; expected [real, imag] pairs")
    state = QuantumState(amps)
    try:
        new_state = cube.apply_operator_to_state(req.operator_id, state)
    except KeyError:
        raise HTTPException(status_code=404, detail="Operator not found")
    result = [[complex_am.real, complex_am.imag] for complex_am in new_state.as_array()]
    return {"state": result}


@app.post("/cube/quantum/measure")
def api_measure_quantum_state(req: QuantumStateRequest):
    """Measure a quantum state in the computational basis.

    The request must include ``amplitudes`` – a list of 13 pairs ``[real, imag]``.
    The state will be normalised before measurement.  The endpoint
    returns the (1‑based) index of the measured node and the collapsed
    state vector.
    """
    if len(req.amplitudes) != 13:
        raise HTTPException(status_code=400, detail="Quantum amplitudes must have length 13")
    try:
        amps = [complex(a[0], a[1]) for a in req.amplitudes]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid amplitude format; expected [real, imag] pairs")
    state = QuantumState(amps)
    outcome = state.measure()
    collapsed = [[c.real, c.imag] for c in state.as_array()]
    return {"outcome": outcome, "state": collapsed}

@app.get("/")
def root():
    return {"message": "Metatron Cube & Resonance API"}

# ---------------------------------------------------------------------------
# QDASH endpoints
# ---------------------------------------------------------------------------
@app.post("/qdash/decision_cycle")
def api_qdash_decision(req: QDASHRequest):
    """Run the QDASH decision cycle on an input vector.

    The request body must include ``input_vector`` as a list of floats.  Optional
    fields ``max_iter`` and ``dt`` control the iteration count and time
    increment.  The endpoint returns the oscillator signal, resonance,
    adaptive threshold, decision flag, spiral embedding and Gabriel cell
    outputs.
    """
    try:
        result = qdash.decision_cycle(req.input_vector, max_iter=req.max_iter, dt=req.dt)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/qdash/state")
def api_qdash_state():
    """Return the latest QDASH decision state and resonance metrics."""
    return {
        "decision": qdash.last_decision,
        "resonance": float(qdash.mandorla.resonance),
        "threshold": float(getattr(qdash.mandorla, 'current_theta', qdash.mandorla.threshold)),
        "history": qdash.mandorla.history
    }

# ---------------------------------------------------------------------------
# Resonance tensor endpoints
# ---------------------------------------------------------------------------
@app.post("/resonance/step")
def api_resonance_step(req: ResonanceUpdateRequest):
    """Step the global resonance tensor field by ``dt`` and optionally modulate it.

    The request must include ``dt`` (time increment).  If
    ``input_modulation`` is provided, it should be a 3D list matching the
    field shape (default 3×3×3) whose values will be added to the phase
    offsets.  The endpoint returns the new resonance state as a nested
    list, the coherence, the gradient norm and whether a singularity has
    been detected.
    """
    mod = None
    if req.input_modulation is not None:
        mod = np.array(req.input_modulation, dtype=float)
    try:
        new_state = res_field.step(req.dt, input_modulation=mod)
        coherence = res_field.coherence()
        grad = res_field.gradient_norm()
        singular = res_field.detect_singularity()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "state": new_state.tolist(),
        "coherence": coherence,
        "gradient_norm": grad,
        "singularity": singular
    }

# ---------------------------------------------------------------------------
# Meta interpreter endpoints
# ---------------------------------------------------------------------------
@app.post("/meta/record_decision")
def api_meta_record():
    """Record the latest decision of the QDASH agent for meta analysis."""
    meta.record_decision()
    return {"message": "decision recorded", "history": list(meta.window)}

@app.post("/meta/adjust")
def api_meta_adjust():
    """Adapt the Mandorla threshold coefficients based on recent decisions."""
    meta.adjust_parameters()
    return {
        "alpha": meta.agent.mandorla.alpha,
        "beta": meta.agent.mandorla.beta,
        "history": list(meta.window)
    }

@app.post("/meta/modulate")
def api_meta_modulate(factor: float):
    """Modulate the oscillator frequency by a given factor."""
    meta.modulate_oscillator_frequency(factor)
    return {"num_nodes": meta.agent.qlogic.osc_core.num_nodes}

# ---------------------------------------------------------------------------
# Tensor network endpoints
# ---------------------------------------------------------------------------
@app.post("/tensor/add_field")
def api_tensor_add_field(shape: List[int], amplitude: float = 1.0, frequency: float = 1.0, phase: float = 0.0):
    """Add a new resonance field to the global tensor network.

    Parameters
    ----------
    shape : List[int]
        Dimensions of the new field (e.g. [3,3,3]).
    amplitude, frequency, phase : float, optional
        Initial parameters for the new field.
    """
    try:
        shape_tuple = tuple(int(x) for x in shape)
        field = ResonanceTensorField(shape_tuple, initial_amplitude=amplitude, initial_frequency=frequency, initial_phase=phase)
        tensor_net.add_field(field)
        return {"message": "field added", "num_fields": len(tensor_net.fields)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/tensor/step")
def api_tensor_step(dt: float, modulations: Optional[List[List[List[List[float]]]]] = None):
    """Step the tensor network and return global metrics.

    Parameters
    ----------
    dt : float
        Time increment for all fields.
    modulations : optional
        A list of 3D arrays (nested lists) for phase modulation of each field.
    """
    mods = None
    if modulations is not None:
        mods = [np.array(m, dtype=float) for m in modulations]
    try:
        states = tensor_net.step(dt, input_modulations=mods)
        coherence = tensor_net.coherence()
        cross = tensor_net.cross_coherence()
        singular = tensor_net.detect_singularities()
        return {
            "states": [s.tolist() for s in states],
            "coherence": coherence,
            "cross_coherence": cross,
            "singularity": singular,
            "num_fields": len(tensor_net.fields)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
