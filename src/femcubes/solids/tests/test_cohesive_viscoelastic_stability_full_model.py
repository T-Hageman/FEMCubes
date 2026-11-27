"""Long-form stability sweep for the cohesive visco-elastic return mapping."""

import numpy as np
import pytest

from femcubes.solids.cpp_kernels.cohesive_viscoelastic._cohesive_viscoelastic_kernels_cpp import (
    CohesiveViscoElasticKernels,
)


YOUNGS = 9.5e9
POISSON = 0.35
SHEAR = YOUNGS / (2.0 * (1.0 + POISSON))
BULK = YOUNGS / (3.0 * (1.0 - 2.0 * POISSON))
LAME = 2.0 * SHEAR * POISSON / (1.0 - 2.0 * POISSON)
GLEN_N = 3.0
GLEN_A = 1.0e-24
VISCOSITY = (1.0 / (2.0 * GLEN_A)) ** (1.0 / GLEN_N)
F_T = 0.4e6
F_S = 1.0e6
DP_PHI = np.degrees(np.arctan(3.0 / (1.0 + POISSON)))
DP_Q = 9.0e6
KAPPA = 1.0e-4

FAILURE_TYPES = ("r1", "r2", "r2_2L", "r2_2L_DP", "r1_DP", "r2_DP")
PLANE_STRESS_MODES = (False, True)
TIME_INCREMENTS = (1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 3600.0)
DAMAGE_LEVELS = (1.0, 0.5, 0.1, 1.0e-2, 1.0e-3, KAPPA)
HISTORY_NAMES = (
    "virgin",
    "viscous_aligned",
    "viscous_near_total",
    "viscous_reversed",
    "cohesive_aligned",
    "mixed_reversed",
)


def _strain_grid():
    directions = {
        "tension_x": (1.0, 0.0, 0.0),
        "compression_x": (-1.0, 0.0, 0.0),
        "tension_y": (0.0, 1.0, 0.0),
        "compression_y": (0.0, -1.0, 0.0),
        "biaxial_tension": (1.0, 1.0, 0.0),
        "biaxial_compression": (-1.0, -1.0, 0.0),
        "deviatoric_xy": (1.0, -1.0, 0.0),
        "deviatoric_yx": (-1.0, 1.0, 0.0),
        "positive_shear": (0.0, 0.0, 1.0),
        "negative_shear": (0.0, 0.0, -1.0),
        "mixed_tension": (1.0, -0.25, 0.5),
        "mixed_compression": (-1.0, 0.25, -0.5),
        "confined_shear": (-1.0, -0.5, 0.75),
        "near_hydrostatic_tension": (1.0, 0.999999, 1.0e-6),
        "near_hydrostatic_compression": (-1.0, -0.999999, 1.0e-6),
    }
    magnitudes = np.asarray(
        [1.0e-10, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 2.5e-5,
         5.0e-5, 1.0e-4, 2.5e-4, 1.0e-3, 1.0e-2, 1.0e-1]
    )

    strains = [np.zeros(6)]
    labels = ["zero"]
    for direction_name, direction_values in directions.items():
        direction = np.asarray(direction_values, dtype=float)
        direction /= np.linalg.norm(direction)
        for magnitude in magnitudes:
            strain = np.zeros(6)
            strain[[0, 1, 5]] = magnitude * direction
            strains.append(strain)
            labels.append(f"{direction_name}@{magnitude:.1e}")
    return np.asarray(strains), labels


def _history(history_name, strains):
    strain_v_old = np.zeros_like(strains)
    eta_old = np.zeros_like(strains)

    if history_name == "viscous_aligned":
        strain_v_old = 0.5 * strains
    elif history_name == "viscous_near_total":
        strain_v_old = 0.99 * strains
    elif history_name == "viscous_reversed":
        strain_v_old = -0.5 * strains
    elif history_name == "cohesive_aligned":
        eta_old = 0.5 * strains
    elif history_name == "mixed_reversed":
        strain_v_old = -0.25 * strains
        eta_old = 0.25 * strains

    if history_name in {"viscous_aligned", "viscous_near_total", "viscous_reversed", "mixed_reversed"}:
        strain_v_old[:, 2] = -(strain_v_old[:, 0] + strain_v_old[:, 1])

    lambda_v = np.linalg.norm(strain_v_old, axis=1)
    lambda_c = np.zeros((len(strains), 2))
    lambda_c[:, 0] = np.linalg.norm(eta_old, axis=1)
    lambda_c[:, 1] = 0.5 * lambda_c[:, 0]
    return strain_v_old, eta_old, lambda_v, lambda_c


def _build_cases():
    base_strains, strain_labels = _strain_grid()
    strain_blocks = []
    strain_v_blocks = []
    eta_blocks = []
    lambda_v_blocks = []
    lambda_c_blocks = []
    damage_blocks = []
    metadata = []

    for damage in DAMAGE_LEVELS:
        for history_name in HISTORY_NAMES:
            strain_v_old, eta_old, lambda_v, lambda_c = _history(history_name, base_strains)
            strain_blocks.append(base_strains)
            strain_v_blocks.append(strain_v_old)
            eta_blocks.append(eta_old)
            lambda_v_blocks.append(lambda_v)
            lambda_c_blocks.append(lambda_c)
            damage_blocks.append(np.full(len(base_strains), damage))
            metadata.extend(
                (damage, history_name, strain_label)
                for strain_label in strain_labels
            )

    return {
        "strain": np.concatenate(strain_blocks),
        "strain_v_old": np.concatenate(strain_v_blocks),
        "eta_old": np.concatenate(eta_blocks),
        "lambda_v": np.concatenate(lambda_v_blocks),
        "lambda_c": np.concatenate(lambda_c_blocks),
        "damage": np.concatenate(damage_blocks),
        "metadata": metadata,
    }


# @pytest.mark.full_model
# def test_cohesive_viscoelastic_return_mapping_exhaustive_stability_sweep():
#     """Exercise broad constitutive states and report every unstable combination."""
#     cases = _build_cases()
#     input_strains = cases["strain"]
#     n_ip = len(input_strains)
#     failures_by_variant = {}
#     failure_counts = {}
#     field_failure_counts = {}
#     failure_count = 0

#     for failure_type in FAILURE_TYPES:
#         for plane_stress in PLANE_STRESS_MODES:
#             kernel = CohesiveViscoElasticKernels(
#                 BULK, LAME, SHEAR,
#                 VISCOSITY, GLEN_N,
#                 F_T, F_S,
#                 1.0, DP_PHI, DP_Q,
#                 KAPPA, failure_type, plane_stress,
#             )

#             for dt in TIME_INCREMENTS:
#                 strains = input_strains.copy()
#                 stress = np.zeros((n_ip, 6))
#                 eta = cases["eta_old"].copy()
#                 strain_v = np.zeros((n_ip, 6))
#                 tangent = np.zeros((n_ip, 36))
#                 elastic_energy = np.zeros(n_ip)
#                 psi = np.zeros(n_ip)
#                 lambda_v = cases["lambda_v"].copy()
#                 lambda_c = cases["lambda_c"].copy()

#                 kernel(
#                     strains.ravel(), dt, cases["damage"],
#                     stress.ravel(), eta.ravel(), cases["eta_old"].ravel(),
#                     strain_v.ravel(), cases["strain_v_old"].ravel(), tangent.ravel(),
#                     elastic_energy, psi, lambda_v, lambda_c.ravel(), cases["lambda_c"].ravel(),
#                 )

#                 finite_by_field = {
#                     "strain/e_zz": np.isfinite(strains).all(axis=1),
#                     "stress": np.isfinite(stress).all(axis=1),
#                     "eta": np.isfinite(eta).all(axis=1),
#                     "strain_v": np.isfinite(strain_v).all(axis=1),
#                     "tangent": np.isfinite(tangent).all(axis=1),
#                     "elastic_energy": np.isfinite(elastic_energy),
#                     "psi": np.isfinite(psi),
#                     "lambda_v": np.isfinite(lambda_v),
#                     "lambda_c": np.isfinite(lambda_c).all(axis=1),
#                 }
#                 finite = np.logical_and.reduce(list(finite_by_field.values()))
#                 valid_multipliers = (lambda_v >= -1.0e-14) & (lambda_c[:, 0] >= -1.0e-14)
#                 if failure_type in {"r1", "r2_2L", "r2_2L_DP", "r1_DP"}:
#                     valid_multipliers &= lambda_c[:, 1] >= -1.0e-14

#                 bad_indices = np.flatnonzero(~finite | ~valid_multipliers)
#                 failure_count += len(bad_indices)
#                 context = (failure_type, plane_stress, dt)
#                 variant = (failure_type, plane_stress)
#                 failure_counts[context] = len(bad_indices)

#                 field_counts = field_failure_counts.setdefault(variant, {})
#                 for field_name, values in finite_by_field.items():
#                     field_counts[field_name] = field_counts.get(field_name, 0) + np.count_nonzero(~values)
#                 field_counts["negative multiplier"] = (
#                     field_counts.get("negative multiplier", 0)
#                     + np.count_nonzero(~valid_multipliers)
#                 )

#                 if len(bad_indices) and variant not in failures_by_variant:
#                     index = bad_indices[0]
#                     damage, history_name, strain_label = cases["metadata"][index]
#                     bad_fields = [name for name, values in finite_by_field.items() if not values[index]]
#                     if not valid_multipliers[index]:
#                         bad_fields.append("negative multiplier")
#                     failures_by_variant[variant] = (
#                         f"criterion={failure_type}, plane_stress={plane_stress}, dt={dt:.1e}, "
#                         f"damage={damage:.1e}, history={history_name}, strain_case={strain_label}, "
#                         f"strain={input_strains[index].tolist()}, strain_v_old={cases['strain_v_old'][index].tolist()}, "
#                         f"failed={bad_fields}"
#                     )

#     if failure_count:
#         summaries = []
#         for variant in ((failure_type, plane_stress)
#                         for failure_type in FAILURE_TYPES
#                         for plane_stress in PLANE_STRESS_MODES):
#             counts = [
#                 f"dt={dt:.1e}: {failure_counts.get((*variant, dt), 0)}/{n_ip}"
#                 for dt in TIME_INCREMENTS
#             ]
#             fields = [
#                 f"{name}={count}"
#                 for name, count in field_failure_counts.get(variant, {}).items()
#                 if count
#             ]
#             summaries.append(
#                 f"  criterion={variant[0]}, plane_stress={variant[1]}: "
#                 f"{', '.join(counts)}; fields: {', '.join(fields) or 'none'}"
#             )

#         representatives = "\n".join(
#             f"  {failures_by_variant[variant]}"
#             for variant in ((failure_type, plane_stress)
#                             for failure_type in FAILURE_TYPES
#                             for plane_stress in PLANE_STRESS_MODES)
#             if variant in failures_by_variant
#         )
#         summary_details = "\n".join(summaries)
#         pytest.fail(
#             f"Cohesive visco-elastic stability sweep found {failure_count} unstable cases "
#             f"across {len(FAILURE_TYPES) * len(PLANE_STRESS_MODES) * len(TIME_INCREMENTS) * n_ip} solves.\n"
#             f"Failures by solver and time increment:\n{summary_details}\n"
#             f"First failure for each affected solver:\n{representatives}"
#         )
