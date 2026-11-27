"""Tests for shared time-discretization helpers."""

import numpy as np

from femcubes.core.time_discretizations import Newmark_Update


class _FakeVector:
    def __init__(self, values):
        self.array = np.array(values, dtype=float)
        self.scatter_calls = 0

    def scatter_forward(self):
        self.scatter_calls += 1


class _FakeField:
    def __init__(self, values):
        self.x = _FakeVector(values)


def test_newmark_update_updates_fields_and_returns_tangent_coefficients():
    u = _FakeField([1.4, -0.2])
    v = _FakeField([0.0, 0.0])
    a = _FakeField([0.0, 0.0])
    u_old = _FakeField([1.0, -0.5])
    v_old = _FakeField([0.3, -0.4])
    a_old = _FakeField([0.2, 0.1])

    dt = 0.5
    beta = 0.25
    gamma = 0.5

    dv_du, da_du = Newmark_Update(u, v, a, u_old, v_old, a_old, dt, beta, gamma)

    expected_v = (
        gamma / beta / dt * (u.x.array - u_old.x.array)
        + (1 - gamma / beta) * v_old.x.array
        + dt * (1 - gamma / (2 * beta)) * a_old.x.array
    )
    expected_a = (
        1 / (beta * dt**2) * (u.x.array - u_old.x.array)
        - 1 / (beta * dt) * v_old.x.array
        - (1 / (2 * beta) - 1) * a_old.x.array
    )

    np.testing.assert_allclose(v.x.array, expected_v)
    np.testing.assert_allclose(a.x.array, expected_a)
    assert dv_du == gamma / (beta * dt)
    assert da_du == 1 / (beta * dt**2)
    assert v.x.scatter_calls == 1
    assert a.x.scatter_calls == 1


def test_newmark_update_optionally_scatters_updated_fields():
    u = _FakeField([0.5])
    v = _FakeField([0.0])
    a = _FakeField([0.0])
    u_old = _FakeField([0.0])
    v_old = _FakeField([0.0])
    a_old = _FakeField([0.0])

    Newmark_Update(u, v, a, u_old, v_old, a_old, 1.0, 0.25, 0.5, scatter_forward=False)

    assert v.x.scatter_calls == 0
    assert a.x.scatter_calls == 0
