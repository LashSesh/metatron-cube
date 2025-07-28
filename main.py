
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
import numpy as np

from src.master_agent import MasterAgent

class AgentInput(BaseModel):
    data: List[Any]

app = FastAPI()
agent = MasterAgent(n_cells=4)

@app.post("/inject/agent_state")
def inject_agent_state(input: AgentInput):
    result = agent.process(input.data)
    return {"result": result}

@app.get("/get/spiralmemory")
def get_spiral_memory():
    memory = agent.spiral_memory.memory
    history = agent.spiral_memory.history
    return {"spiralmemory": str(memory), "history": history}

@app.get("/get/gabrielcells")
def get_gabrielcells():
    cells = [
        {"psi": c.psi, "rho": c.rho, "omega": c.omega, "output": c.output}
        for c in agent.gabriel_cells
    ]
    return {"gabriel_cells": cells}

@app.get("/get/mandorla")
def get_mandorla():
    return {
        "inputs": [list(v) for v in agent.mandorla.inputs],
        "resonance": agent.mandorla.resonance,
        "decision": agent.last_decision,
        "history": agent.mandorla.history
    }

@app.get("/compute/decision")
def compute_decision():
    res = agent.last_decision
    return {"decision": res}

@app.get("/get/full_state")
def get_full_state():
    return {
        "spiralmemory": agent.spiral_memory.memory,
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

@app.get("/")
def root():
    return {"message": "Fusion API for MasterAgent (5D Resonance System)"}
