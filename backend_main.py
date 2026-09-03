"""
Q-Circuit Lab — Backend
FastAPI + Qiskit Aer + Claude API

This is the real backend described in Section 5 of the master plan.
The in-browser prototype (quantum_circuit_lab.html) reimplements this same
gate logic in JavaScript so it can run instantly with no server for the demo;
this file is the production version that actually calls Qiskit Aer, which is
what you show a judge who asks "is this really running quantum simulation?"

Run locally:
    pip install -r requirements.txt
    uvicorn backend_main:app --reload
    -> POST http://localhost:8000/run-circuit
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Optional
import os

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer import AerSimulator
import anthropic

app = FastAPI(title="Q-Circuit Lab API")

# Frontend (React/Vercel) needs CORS to call this from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten to your deployed frontend URL before submission
    allow_methods=["*"],
    allow_headers=["*"],
)

claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------
# Request/response schema — mirrors the gate objects the frontend sends
# ---------------------------------------------------------------------
class Gate(BaseModel):
    type: Literal["H", "X", "Z", "CX"]
    qubits: List[int]          # [q] for H/X/Z, [control, target] for CX
    col: int


class CircuitRequest(BaseModel):
    num_qubits: int
    gates: List[Gate]
    shots: int = 1000


class RunResult(BaseModel):
    counts: dict                     # {"00": 512, "11": 488, ...}
    probabilities: dict              # same keys, as fractions
    bloch_vectors: List[dict]        # [{"x":.., "y":.., "z":..}, ...] per qubit


# ---------------------------------------------------------------------
# Step 1: build the real Qiskit circuit from the gate list the UI sent
# ---------------------------------------------------------------------
def build_circuit(req: CircuitRequest) -> QuantumCircuit:
    qc = QuantumCircuit(req.num_qubits, req.num_qubits)
    for g in sorted(req.gates, key=lambda g: g.col):
        if g.type == "H":
            qc.h(g.qubits[0])
        elif g.type == "X":
            qc.x(g.qubits[0])
        elif g.type == "Z":
            qc.z(g.qubits[0])
        elif g.type == "CX":
            qc.cx(g.qubits[0], g.qubits[1])
    return qc


# ---------------------------------------------------------------------
# Step 2: the core endpoint — this is the bridge between frontend and
# the actual quantum simulator (Section 5, step 3 of the plan)
# ---------------------------------------------------------------------
@app.post("/run-circuit", response_model=RunResult)
def run_circuit(req: CircuitRequest):
    if req.num_qubits not in (2, 3):
        raise HTTPException(400, "This MVP supports 2 or 3 qubits.")

    qc = build_circuit(req)

    # Exact statevector — used for probabilities and Bloch vectors
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities_dict()

    # Bloch vector per qubit via partial trace of the density matrix
    dm = DensityMatrix(sv)
    bloch_vectors = []
    for q in range(req.num_qubits):
        trace_out = [i for i in range(req.num_qubits) if i != q]
        reduced = dm if not trace_out else __import__("qiskit.quantum_info", fromlist=["partial_trace"]).partial_trace(dm, trace_out)
        paulis = {"x": [[0, 1], [1, 0]], "y": [[0, -1j], [1j, 0]], "z": [[1, 0], [0, -1]]}
        import numpy as np
        m = reduced.data
        bx = float(np.real(np.trace(m @ np.array(paulis["x"]))))
        by = float(np.real(np.trace(m @ np.array(paulis["y"]))))
        bz = float(np.real(np.trace(m @ np.array(paulis["z"]))))
        bloch_vectors.append({"x": bx, "y": by, "z": bz})

    # Shot-based counts — this is the literal "run 1000 shots" from the plan
    qc.measure(range(req.num_qubits), range(req.num_qubits))
    backend = AerSimulator()
    job = backend.run(qc, shots=req.shots)
    counts = job.result().get_counts()

    return RunResult(counts=counts, probabilities=probs, bloch_vectors=bloch_vectors)


# ---------------------------------------------------------------------
# Step 3: AI tutor endpoint — grounds Claude in the real circuit + results
# so it explains truth, not a guess (Section 9's answer to "hallucination?")
# ---------------------------------------------------------------------
class ExplainRequest(BaseModel):
    gates: List[Gate]
    counts: dict


@app.post("/explain")
def explain_circuit(req: ExplainRequest):
    gate_desc = ", ".join(
        f"CNOT(control=q{g.qubits[0]}, target=q{g.qubits[1]})" if g.type == "CX"
        else f"{g.type}(q{g.qubits[0]})"
        for g in sorted(req.gates, key=lambda g: g.col)
    ) or "no gates (identity circuit)"

    prompt = (
        f"A student built this quantum circuit: {gate_desc}. "
        f"Measured counts over 1000 shots: {req.counts}. "
        "In under 120 words, plain English, explain step by step what happened "
        "to the qubits and why the results look the way they do."
    )

    msg = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"explanation": msg.content[0].text}


@app.get("/health")
def health():
    return {"status": "ok"}
