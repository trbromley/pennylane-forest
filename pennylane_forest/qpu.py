"""
QPU Device
==========

**Module name:** :mod:`pennylane_forest.qpu`

.. currentmodule:: pennylane_forest.qpu

This module contains the :class:`~.QPUDevice` class, a PennyLane device that allows
evaluation and differentiation of Rigetti's Forest Quantum Processing Units (QPUs)
using PennyLane.

Classes
-------

.. autosummary::
   QPUDevice

Code details
~~~~~~~~~~~~
"""
import re

from pyquil import get_qc

from .qvm import QVMDevice
from ._version import __version__

import numpy as np

from pyquil import get_qc
from pyquil.api._quantum_computer import _get_qvm_with_topology
from pyquil.gates import MEASURE, RESET, X
from pyquil.quil import Pragma, Program
from pyquil.paulis import sI, sX, sY, sZ
from pyquil.operator_estimation import ExperimentSetting, TensorProductState, Experiment, measure_observables, group_experiments
from pyquil.quilbase import Gate


class QPUDevice(QVMDevice):
    r"""Forest QPU device for PennyLane.

    Args:
        device (str): the name of the device to initialise.
        shots (int): number of circuit evaluations/random samples used
            to estimate expectation values of observables.
        active_reset (bool): whether to actively reset qubits instead of waiting for
            for qubits to decay to the ground state naturally.
            Setting this to ``True`` results in a significantly faster expectation value
            evaluation when the number of shots is larger than ~1000.
        load_qc (bool): set to False to avoid getting the quantum computing
            device on initialization. This is convenient if not currently connected to the QPU.
        readout_mitigation (bool): sets whether to perform error mitigation
            against bit flips during readout

    Keyword args:
        forest_url (str): the Forest URL server. Can also be set by
            the environment variable ``FOREST_SERVER_URL``, or in the ``~/.qcs_config``
            configuration file. Default value is ``"https://forest-server.qcs.rigetti.com"``.
        qvm_url (str): the QVM server URL. Can also be set by the environment
            variable ``QVM_URL``, or in the ``~/.forest_config`` configuration file.
            Default value is ``"http://127.0.0.1:5000"``.
        compiler_url (str): the compiler server URL. Can also be set by the environment
            variable ``COMPILER_URL``, or in the ``~/.forest_config`` configuration file.
            Default value is ``"http://127.0.0.1:6000"``.
        compiler_timeout (int): the time in seconds allowed to run on compiler before
            resulting in a timeout. Default value is 100 seconds.
    """
    name = "Forest QPU Device"
    short_name = "forest.qpu"
    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard", "Hermitian"}

    def __init__(self, device, *, shots=1024, active_reset=True, load_qc=True, readout_mitigation=False, **kwargs):

        self._eigs = {}

        if "wires" in kwargs:
            raise ValueError("QPU device does not support a wires parameter.")

        if shots <= 0:
            raise ValueError("Number of shots must be a positive integer.")

        aspen_match = re.match(r"Aspen-\d+-([\d]+)Q", device)
        num_wires = int(aspen_match.groups()[0])

        super(QVMDevice, self).__init__(num_wires, shots, **kwargs)

        if load_qc:
            self.qc = get_qc(device, as_qvm=False, connection=self.connection)
            self.qc.compiler.quilc_client.timeout = kwargs.pop("compiler_timeout", 100)
        else:
            self.qc = get_qc(device, as_qvm=True, connection=self.connection, noisy=True)
            self.qc.compiler.client.timeout = kwargs.pop("compiler_timeout", 100)

        self.active_reset = active_reset
        self.wiring = {i: q for i, q in enumerate(self.qc.qubits())}
        
        self.readout_mitigation = readout_mitigation
        if readout_mitigation:
            self.p10, self.p01 = self.calibrate()

    def calibrate(self):
        shots = 10000
        
        p = Program()
        p += Pragma("INITIAL_REWIRING", ['"NAIVE"'])

        if self.active_reset:
            p += RESET()

        ro = p.declare("ro", "BIT", len(self.qc.qubits()))
        
        for i, q in enumerate(self.qc.qubits()):
            p += MEASURE(q, ro[i])
        
        p.wrap_in_numshots_loop(shots)
        p_comp = self.qc.compile(p)
        
        results0 = self.qc.run(p_comp)
        
        p = Program()
        p += Pragma("INITIAL_REWIRING", ['"NAIVE"'])

        if self.active_reset:
            p += RESET()

        for q in self.qc.qubits():
            p += X(q)   
            
        ro = p.declare("ro", "BIT", len(self.qc.qubits()))
        
        for i, q in enumerate(self.qc.qubits()):
            p += MEASURE(q, ro[i])
        
        p.wrap_in_numshots_loop(shots)
        p_comp = self.qc.compile(p)
        
        results1 = self.qc.run(p_comp)

        p10 = np.sum(results0, axis=0) / shots
        p01 = np.sum(1 - results1, axis=0) / shots
        
        return p10, p01
        
    def multiqubit_sample(self, observable):
        wires = observable.wires
        name = observable.name

#         if isinstance(name, str) and name in {"PauliX", "PauliY", "PauliZ", "Hadamard"}:
#             # Process samples for observables with eigenvalues {1, -1}
#             return 1 - 2 * self._samples[:, wires[0]]

        # Replace the basis state in the computational basis with the correct eigenvalue.
        # Extract only the columns of the basis samples required based on ``wires``.
        wires = np.hstack(wires)
        samples = self._samples[:, np.array(wires)]
        return samples
        
    def expval(self, observable):

        if self.readout_mitigation and observable.name[0] is not 'Identity':
            samples = self.multiqubit_sample(observable)
            
            if len(observable.wires) > 1:
                samples = [tuple(s) for s in samples]
                obs_wiring = {i: w[0] for i, w in enumerate(observable.wires)}
            else:
                samples = [int(s) for s in samples]
                obs_wiring = {0: observable.wires[0]}
            
            from collections import Counter
            
            counts = Counter(samples)
            
            prod_sum = []
            
            for sample, count in counts.items():
                p = count / self.shots
                
                pplus = self.p01 + self.p10
                pminus = self.p01 - self.p10
                
                if len(observable.wires) > 1:
                    prod = np.prod([((-1) ** s - pminus[obs_wiring[i]]) / (1 - pplus[obs_wiring[i]]) for i, s in enumerate(sample)])
                else:
                    prod = ((-1) ** sample - pminus[obs_wiring[0]]) / (1 - pplus[obs_wiring[0]])
                
                prod_sum.append(prod * p)
                
            return np.sum(prod_sum)#, super().expval(observable)
                
        return super().expval(observable)
