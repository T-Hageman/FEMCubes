"""Strong-scaling study of the cohesive-elastic FullPlate case with half the length scale.

The case of ``scaling_cohesive_elastic.py`` (the ``full_plate`` case of
``examples/fracture/cohesive_phasefield.py`` with the reduced return mapping)
with the phase-field length scale halved, from 10 mm to 5 mm. The refinement
target (``ell / 2``) and the crack growth allowed per step (``ell``) follow it,
so the refined crack band holds about twice as many cells and the final system
is larger. As the mesh is refined adaptively, the run starts on the same small
mesh and only becomes a large problem once the cracks have grown.

Run the study from the repository root with::

    python3 benchmarks/scaling_cohesive_elastic_half_ell.py --plotting off

The options, outputs and figures are those of ``scaling_cohesive_elastic.py``;
the results go to ``benchmarks/results/scaling_cohesive_elastic_half_ell``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from scaling_cohesive_elastic import CohesiveElasticScaling

ELL_FACTOR = 0.5                # Length scale relative to the example's (10 mm)
DEFAULT_CORE_COUNTS = (56, 48, 32, 24, 16, 8, 4, 2)


class CohesiveElasticHalfEllScaling(CohesiveElasticScaling):
    """Strong scaling of the ``full_plate`` case with half the phase-field length scale."""

    name = "scaling_cohesive_elastic_half_ell"
    case_label = "FullPlateHalfEll"
    title = "Cohesive-elastic FullPlate, half length scale"
    description = (
        "Run the cohesive-elastic FullPlate case with half the phase-field length "
        "scale as an MPI strong-scaling study and save its timings and runtime plot."
    )
    default_core_counts = DEFAULT_CORE_COUNTS
    default_cores_help = "56, 48, 32, 24, 16, 8, 4 and 2"

    def build_params(self, plotting_enabled: bool, plotting_folder: Path) -> Any:
        """Build the case of the parent study, then scale the length scale and everything set from it."""

        params = super().build_params(plotting_enabled, plotting_folder)
        ell = ELL_FACTOR * params.ell
        params.ell = ell
        params.Materials["Iron"]["ell"] = ell
        params.MeshRefinement = dataclasses.replace(params.MeshRefinement, target_h=ell / 2)
        params.TimeStepControl["Target_Change"] = ell  # The example's crack growth per step for full_plate
        return params


def main() -> None:
    """Run either the outer scaling driver or one internal MPI worker."""

    CohesiveElasticHalfEllScaling().main()


if __name__ == "__main__":
    main()
