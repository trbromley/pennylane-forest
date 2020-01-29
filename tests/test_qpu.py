"""
Unit tests for the QPU device.
"""
import logging

import pytest
import pyquil
import pennylane as qml
from pennylane import numpy as np
from conftest import BaseTest, QVM_SHOTS

from flaky import flaky

log = logging.getLogger(__name__)

VALID_QPU_LATTICES = [qc for qc in pyquil.list_quantum_computers() if "qvm" not in qc]


class TestQPUIntegration(BaseTest):
    """Test the wavefunction simulator works correctly from the PennyLane frontend."""

    # pylint: disable=no-self-use

    def test_load_qpu_device(self):
        """Test that the QPU device loads correctly"""
        device = [qpu for qpu in VALID_QPU_LATTICES if '-2Q' in qpu][0]
        dev = qml.device("forest.qpu", device=device, load_qc=False)
        self.assertEqual(dev.num_wires, 2)
        self.assertEqual(dev.shots, 1024)
        self.assertEqual(dev.short_name, "forest.qpu")

    def test_load_virtual_qpu_device(self):
        """Test that the QPU simulators load correctly"""
        device = np.random.choice(VALID_QPU_LATTICES)
        qml.device("forest.qpu", device=device, load_qc=False)

    def test_qpu_args(self):
        """Test that the QPU plugin requires correct arguments"""
        device = np.random.choice(VALID_QPU_LATTICES)

        with pytest.raises(ValueError, match="QPU device does not support a wires parameter"):
            qml.device("forest.qpu", device=device, wires=2)

        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qml.device("forest.qpu")

        with pytest.raises(ValueError, match="Number of shots must be a positive integer"):
            qml.device("forest.qpu", device=device, shots=0)

        with pytest.raises(ValueError, match="Readout error cannot be set on the physical QPU"):
            qml.device("forest.qpu", device=device, load_qc=True, readout_error=[0.9, 0.75])


class TestQPUBasic(BaseTest):
    """Unit tests for the QPU (as a QVM)."""

    # pylint: disable=protected-access

    def test_no_readout_correction(self):
        device = np.random.choice(VALID_QPU_LATTICES)
        dev_qpu = qml.device('forest.qpu', device=device, load_qc=False, readout_error=[0.9, 0.75],
                            symmetrize_readout=None, calibrate_readout=None)
        qubit = 0   # just run program on the first qubit

        @qml.qnode(dev_qpu)
        def circuit_Xpl():
            qml.RY(np.pi/2, wires=qubit)
            return qml.expval(qml.PauliX(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Xmi():
            qml.RY(-np.pi/2, wires=qubit)
            return qml.expval(qml.PauliX(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Ypl():
            qml.RX(-np.pi/2, wires=qubit)
            return qml.expval(qml.PauliY(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Ymi():
            qml.RX(np.pi/2, wires=qubit)
            return qml.expval(qml.PauliY(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Zpl():
            qml.RX(0.0, wires=qubit)
            return qml.expval(qml.PauliZ(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Zmi():
            qml.RX(np.pi, wires=qubit)
            return qml.expval(qml.PauliZ(qubit))

        num_expts = 10
        results_unavged = np.zeros((num_expts, 6))

        for i in range(num_expts):
            results_unavged[i, :] = [circuit_Xpl(), circuit_Ypl(), circuit_Zpl(),
                                     circuit_Xmi(), circuit_Ymi(), circuit_Zmi()]

        results = np.mean(results_unavged, axis=0)

        assert np.allclose(results[:3], 0.8, atol=2e-2)
        assert np.allclose(results[3:], -0.5, atol=2e-2)

    def test_readout_correction(self):
        device = np.random.choice(VALID_QPU_LATTICES)
        dev_qpu = qml.device('forest.qpu', device=device, load_qc=False, readout_error=[0.9, 0.75],
                            symmetrize_readout="exhaustive", calibrate_readout="plus-eig")
        qubit = 0   # just run program on the first qubit

        @qml.qnode(dev_qpu)
        def circuit_Xpl():
            qml.RY(np.pi/2, wires=qubit)
            return qml.expval(qml.PauliX(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Xmi():
            qml.RY(-np.pi/2, wires=qubit)
            return qml.expval(qml.PauliX(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Ypl():
            qml.RX(-np.pi/2, wires=qubit)
            return qml.expval(qml.PauliY(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Ymi():
            qml.RX(np.pi/2, wires=qubit)
            return qml.expval(qml.PauliY(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Zpl():
            qml.RX(0.0, wires=qubit)
            return qml.expval(qml.PauliZ(qubit))

        @qml.qnode(dev_qpu)
        def circuit_Zmi():
            qml.RX(np.pi, wires=qubit)
            return qml.expval(qml.PauliZ(qubit))

        num_expts = 10
        results_unavged = np.zeros((num_expts, 6))

        for i in range(num_expts):
            results_unavged[i, :] = [circuit_Xpl(), circuit_Ypl(), circuit_Zpl(),
                                     circuit_Xmi(), circuit_Ymi(), circuit_Zmi()]

        results = np.mean(results_unavged, axis=0)

        assert np.allclose(results[:3], 1.0, atol=2e-2)
        assert np.allclose(results[3:], -1.0, atol=2e-2)

    @flaky(max_runs=10, min_passes=1)
    def test_2q_gate(self):
        device = np.random.choice(VALID_QPU_LATTICES)
        dev_qpu = qml.device('forest.qpu', device=device, load_qc=False, readout_error=[0.9, 0.75],
                            symmetrize_readout="exhaustive", calibrate_readout="plus-eig", shots=QVM_SHOTS)

        @qml.qnode(dev_qpu)
        def circuit():
            qml.RY(np.pi/2, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(), 0.0, atol=2e-2)

    @flaky(max_runs=10, min_passes=1)
    def test_2q_gate_pauliz_identity_tensor(self):
        device = np.random.choice(VALID_QPU_LATTICES)
        dev_qpu = qml.device('forest.qpu', device=device, load_qc=False, readout_error=[0.9, 0.75],
                            symmetrize_readout="exhaustive", calibrate_readout="plus-eig", shots=QVM_SHOTS)

        @qml.qnode(dev_qpu)
        def circuit():
            qml.RY(np.pi/2, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.Identity(1))

        assert np.allclose(circuit(), 0.0, atol=2e-2)

    @flaky(max_runs=10, min_passes=1)
    def test_2q_gate_pauliz_pauliz_tensor(self):
        device = np.random.choice(VALID_QPU_LATTICES)
        dev_qpu = qml.device('forest.qpu', device=device, load_qc=False, readout_error=[0.9, 0.75],
                            symmetrize_readout="exhaustive", calibrate_readout="plus-eig", shots=QVM_SHOTS)

        @qml.qnode(dev_qpu)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert np.allclose(circuit(), 1.0, atol=2e-2)
