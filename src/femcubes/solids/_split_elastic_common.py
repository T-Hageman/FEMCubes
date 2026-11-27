"""Shared elastic energy drivers and configuration for solid models."""

import math

import numpy as np
import ufl

from femcubes.solids.cpp_kernels.viscoelastic._viscoelastic_kernels_cpp import elastic_split_drivers


_SPLITS = ("FullEnergy", "VolDev", "VolOnly", "Spectral", "NoTension", "Stress-Based")


def _fracture_driver(raw_driver, split, material, *, clip_at2=False, elastic_strain=None, stress=None):
    """PRIVATE: Convert any shared-kernel split result to the ElasticEnergy field.

    Split mathematics (energy, stresses, tangents) lives in elastic_split() in
    cpp_kernels/shared/phasefield_splits.h. Its raw driver is psi+ for energy
    splits and squared tensile principal stress for Stress-Based. This is the
    single conversion path for both original and consistent solid models.
    With material=None, Stress-Based reports total elastic energy instead of
    a phase-field strength driver, using the supplied elastic strain and stress.
    """
    if split not in _SPLITS: raise ValueError(f"Unknown CrackDrivingForce type: {split}")
    if split != "Stress-Based": return raw_driver
    if material is None: return 0.5 * (elastic_strain * stress).sum(axis=1)
    return _strength_driver(raw_driver, material["Gc"], material["ell"], material["f_t"], material.get("PF_formulation", "AT2"), clip_at2=clip_at2)

def _set_fracture_driver(mesh, strain, lame, shear, true_2d, split, material, *, clip_at2=False, domain=None):
    """PRIVATE: Evaluate any phase-field split from a six-component Mandel strain.

    Strains and spatially varying material properties are sampled on the live
    quadrature spaces, including after remeshing. Stress and history are untouched.
    With ``domain``, only the driver at the points of that (sub)domain is
    stored, so models sharing ``ElasticEnergy`` on other domains keep theirs.
    """
    if split not in _SPLITS: raise ValueError(f"Unknown CrackDrivingForce type: {split}")
    energy = mesh.Get_Field("ElasticEnergy")

    # Everything is sampled at the ElasticEnergy points, on its own support
    # (which may be a subdomain), so all arrays line up point by point.
    def sample(value):
        if isinstance(value, ufl.core.expr.Expr):
            return mesh.Evaluate_At_Field_Points("ElasticEnergy", value).reshape(-1)
        return np.full(energy.x.array.size, value, dtype=float)

    values = mesh.Evaluate_At_Field_Points("ElasticEnergy", ufl.as_vector([strain[i] for i in range(6)]))
    raw_driver = elastic_split_drivers(values, sample(lame), sample(shear), split, true_2d)
    if split == "Stress-Based":
        material = {**material, **{key: sample(material[key]) for key in ("Gc", "ell", "f_t")}}
    driver = _fracture_driver(raw_driver, split, material, clip_at2=clip_at2)
    dofs = mesh.Get_Field_Dofs("ElasticEnergy", domain)
    if dofs is None:
        energy.x.array[:] = driver
    else:
        energy.x.array[dofs] = driver[dofs]
    energy.x.scatter_forward()

def _validate_split_settings(settings, material, model_name):
    """PRIVATE: Validate damage and split names and return the selected split."""
    if settings.get("DamageModel") not in (None, "PhaseField"):
        raise ValueError(f"{model_name} supports DamageModel=None or 'PhaseField'.")
    split = material.get("CrackDrivingForce", "Stress-Based")
    if split not in _SPLITS:
        raise ValueError(f"Unknown CrackDrivingForce type: {split}")
    return split

def _validate_split_damage_material(material, split, *, Gc=None, f_t=None):
    """PRIVATE: Check phase-field material data and return its formulation.

    ``Gc`` and ``f_t`` replace the material's values when given; UFL expressions (spatially
    varying properties) are not checked here but by the model that builds them.
    """
    kappa = material.get("kappa", 1e-3)
    if not math.isfinite(kappa) or not 0.0 < kappa <= 1.0: raise ValueError("Split solid models require 0 < kappa <= 1.")
    formulation = material.get("PF_formulation", "AT2")
    if split == "Stress-Based":
        values = {"Gc": material["Gc"] if Gc is None else Gc, "ell": material["ell"],
                  "f_t": material["f_t"] if f_t is None else f_t}
        if any(not isinstance(value, ufl.core.expr.Expr) and (not math.isfinite(value) or value <= 0)
               for value in values.values()):
            raise ValueError("Stress-Based requires finite positive Gc, ell and f_t.")
        if formulation not in ("AT1", "AT2", "HO-AT1", "HO-AT2"):
            raise ValueError("Stress-Based supports PF_formulation AT1, AT2, HO-AT1 or HO-AT2.")
    return formulation


def _strength_driver(squared_positive_stress, gc, ell, tensile_strength, formulation, *, clip_at2=False):
    """PRIVATE: Convert squared tensile principal stresses to the strength driver."""
    strength = squared_positive_stress / tensile_strength**2
    if formulation in ("AT2", "HO-AT2"):
        excess = strength - 1.0
        if clip_at2:
            excess = np.maximum(excess, 0.0)
        return gc / (4.0*ell) * excess
    return 3.0*gc / (16.0*ell) * strength
