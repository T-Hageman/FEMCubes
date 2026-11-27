<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo/femcubes_logo_dark.svg">
    <img src="docs/_static/logo/femcubes_logo.svg" alt="FEMCubes" width="420">
  </picture>
</p>

# FEMCubes

A modular multi-physics finite-element framework built on
[FEniCSx](https://fenicsproject.org/).

FEMCubes assembles a simulation from self-contained physics models. A
parameter class names the mesh, the materials and the models; each model adds
its fields and its weak form, and the framework takes care of assembly,
staggered or monolithic solution, adaptive time stepping, adaptive mesh
refinement, parallel output and plotting. Models and mesh generators register
themselves under a type string, so new physics is added as one module, or as
a separate plugin package.

The package provides models for

- **solid mechanics and fracture**: linear elasticity, viscoelasticity, Stokes
  flow, phase-field fracture, and strength-based cohesive phase-field fracture
  with C++ integration-point kernels;
- **glaciology**: ice rheology, shallow-ice and shallow-shelf flow, basal and
  floating-ice friction, ocean pressure;
- **electrochemistry**: electrolyte transport, metal-electrolyte interfaces,
  passivating surfaces, hydrogen uptake;
- **poromechanics**: pore pressure and fluid flow in fractures;

together with boundary conditions and parametric mesh generators. The
[documentation](https://t-hageman.github.io/FEMCubes/) lists every model with
its parameters.

## Installation

FEMCubes runs on Linux, macOS, and Windows through WSL2. The dependencies are
installed with Conda, as FEniCSx is not distributed on PyPI:

```bash
conda env create -f environment.yml
conda activate FenicsX
python -m pip install --no-build-isolation -e .
```

`environment.yml` holds the versions FEMCubes is tested with. On Linux x86-64,
`environment.lock.yml` reproduces the complete tested environment, including
all transitive packages. The install step compiles the C++ kernels with the
toolchain of the environment.

## Quick start

```python
from femcubes import Mesh, Physics, TimeSolver, ParamsTemplate
import femcubes.solids
import femcubes.boundary

class Params(ParamsTemplate):
    dim = 2
    MeshType = "Rectangle"
    dx = 0.1
    Lx = 1.0
    Ly = 1.0
    # ... materials, models, solver settings

params = Params()
mesh = Mesh(params)
physics = Physics(mesh, params)
physics.Initialize_Fields()
solver = TimeSolver(params, physics)
solver.run()
```

Complete, runnable set-ups are in [`examples/`](examples/), grouped into
fracture, glaciology and electrochemistry; [`benchmarks/`](benchmarks/) holds
performance studies. All of them run in parallel with MPI:

```bash
mpirun -np 4 python examples/glaciology/ice_cliffs.py --case ve_cohesive
```

## Documentation

The documentation is at <https://t-hageman.github.io/FEMCubes/>. To build it
locally:

```bash
bash scripts/build_docs.sh
bash scripts/start_doc_server.sh    # serve it at http://localhost:8137/ (--build rebuilds first)
bash scripts/stop_doc_server.sh
```

## Testing

```bash
# Fast subset
python -m pytest src -m "not slow and not full_model and not mpi and not examples" -n auto

# Full suite
python scripts/run_tests.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New physics can be developed as a
plugin package and registered through the `femcubes.plugins` entry-point
group.

## Citing

If you use FEMCubes in your work, please cite it with the metadata in
[CITATION.cff](CITATION.cff) (the "Cite this repository" button on GitHub).

## License

FEMCubes is released under the [MIT License](LICENSE).
