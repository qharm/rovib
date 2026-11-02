# Copyright 2026 QHARM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import unittest

import numpy as np

from rovib import encode_harmonic_hamiltonian


class TestHarmonicPauliEncoding(unittest.TestCase):
    def test_encoding_matches_truncated_energies(self):
        encoded = encode_harmonic_hamiltonian([1.0, 2.0], vmax=2)
        matrix = []
        for v1 in range(3):
            for v2 in range(3):
                state = v1 | (v2 << 2)
                value = 0.0
                for pauli, coefficient in encoded.items():
                    eigenvalue = np.prod(
                        [
                            1.0
                            if bit == "I"
                            else (-1.0 if (state >> q) & 1 else 1.0)
                            for q, bit in enumerate(pauli)
                        ]
                    )
                    value += coefficient * eigenvalue
                matrix.append(value)
        expected = np.array(
            [v1 + 2 * v2 + 1.5 for v1 in range(3) for v2 in range(3)]
        )
        np.testing.assert_allclose(sorted(matrix), sorted(expected))

    def test_unused_binary_states_are_zero(self):
        encoded = encode_harmonic_hamiltonian([1.0], vmax=2)
        values = []
        for state in range(4):
            value = 0.0
            for pauli, coefficient in encoded.items():
                eigenvalue = np.prod(
                    [
                        1.0
                        if bit == "I"
                        else (
                            -1.0
                            if (state >> (len(pauli) - 1 - q)) & 1
                            else 1.0
                        )
                        for q, bit in enumerate(pauli)
                    ]
                )
                value += coefficient * eigenvalue
            values.append(value)
        np.testing.assert_allclose(values, [0.5, 1.5, 2.5, 0.0])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            encode_harmonic_hamiltonian([1.0], -1)
        with self.assertRaises(ValueError):
            encode_harmonic_hamiltonian([1.0], 1.5)
