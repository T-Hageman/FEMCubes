# Contributing to FEMCubes

Contributions are welcome: bug reports, fixes, documentation, and new models.
For anything larger than a fix, please open an issue first to agree on the
approach.

## Set-up

```bash
conda env create -f environment.yml
conda activate FenicsX
python -m pip install --no-build-isolation -e .
```

The editable install compiles the C++ integration-point kernels. Python
changes are picked up directly; after a change to a C++ kernel, run
`bash scripts/reinstall.sh`.

## Tests

```bash
# Fast subset (also run by the CI on every push)
python -m pytest src -m "not slow and not full_model and not mpi and not examples" -n auto

# Full suite, including full-model, MPI and example smoke tests
python scripts/run_tests.py
```

Run the full suite before opening a pull request that touches a model, the
mesh, or the solvers. The developer documentation describes the test layout
and the markers.

## Standards

[AGENTS.md](AGENTS.md) holds the editing standards of this repository. They
apply to every contribution, whether written by hand or with an AI coding
tool. In short:

- Code, tests, examples and documentation are updated together.
- Every bug fix comes with a regression test.
- Tests check the public contract (boundary names, model-type strings), not
  implementation details.
- Modules are named in snake_case. Class names and the `MODEL_TYPE` and
  `MESH_TYPE` strings are the user-facing interface and are not renamed.

## Adding physics

A model is a subclass of `femcubes.core.base_model.BaseModel` with a unique
`MODEL_TYPE`; a mesh generator is a subclass of
`femcubes.core.mesh.generator_base.MeshGeneratorBase` with a unique
`MESH_TYPE`. Both register themselves when their module is imported.

New physics starts as a plugin: a separate package that uses the public
interface of FEMCubes and registers itself through the `femcubes.plugins`
entry-point group (see the developer documentation). Once the physics is
published, it moves into the core package, with its tests, an API page and,
where useful, an example.

## Commits and pull requests

- Describe the change itself: what was changed and, where it is not obvious,
  why.
- The person who commits is responsible for every line of the change,
  whichever tools were used to write it. Commits and pull requests carry no AI
  attribution.
- Keep unrelated changes in separate commits.

## Releases

1. The full suite passes with the pinned environment.
2. `CHANGELOG.md` has a section for the release.
3. `CITATION.cff` carries the new version and release date.
4. Tag the commit `vX.Y.Z`; the package version is derived from the tag.
