# FEMCubes Editing Standards for AI Agents

This file is intended as a persistent editing reference for AI coding
agents working in this repository. It is not part of the published user
documentation.

The goals are:
- keep tests meaningful and maintainable
- avoid regressions caused by over-specific assertions
- keep code, tests, and documentation updated together


## General Principles

- Prefer small, explicit interfaces over hidden side effects.
- Preserve existing behavior unless the task explicitly requests a change.
- Treat public API changes as cross-cutting changes: code, tests,
  examples, and docs should all be updated together.
- Optimize for maintainability over cleverness. Short, direct code is
  preferred.
- Avoid introducing broad compatibility layers unless they are actually
  needed.
- Long lines are acceptable when they are readable. Do not reformat code,
  tests, or documentation solely to satisfy a line-length preference.


## Public vs Private API

- Public methods and functions should not use a leading underscore.
- Internal helpers should use a leading underscore.
- In larger modules and mixins, place public methods first and private
  helpers afterwards.
- Use explicit section headers when a file grows large, for example:
  `Public API`, `Internal Helpers`, `Internal Helpers: Construction`.
- Private helper docstrings should begin with `PRIVATE:` when the file
  already follows that convention.
- Tests should primarily target public behavior. Add white-box tests only
  when a branch is hard to reach through the public API and the branch is
  still important to protect.


## Code Structure

- Prefer structured return values over mutating caller-owned state when a
  function is conceptually producing artifacts.
- When a helper already exists for a concept, route new code through it
  instead of duplicating logic.
- Use named marker/group APIs instead of directly manipulating backing
  dictionaries when a class already exposes add/get/list methods.
- Prefer stable names as the API surface; avoid making raw numeric IDs
  part of the supported contract unless they are intentionally stable.
- Reject invalid input explicitly. Do not silently reinterpret invalid
  parameters as another valid mode.


## Naming

- Modules and test files are named in snake_case
  (`cohesive_viscoelastic.py`, `test_mesh_groups.py`); private modules keep
  their leading underscore.
- Class names and the `MODEL_TYPE` and `MESH_TYPE` strings are the
  user-facing interface. Do not rename them to match a module name.
- The source lives in `src/femcubes/`; tests sit in a `tests` folder next to
  the code they cover.


## Imports

- Keep imports at module scope when the dependency is an inherent part of
  the module.
- Avoid unused imports, especially leftover function-local imports.
- Local imports are acceptable only when they serve a real purpose, such
  as avoiding an optional dependency at import time or breaking a cycle.


## Comments and Docstrings

- Add comments sparingly. Prefer code that is clear without narration.
- Use comments to explain non-obvious intent, branch rationale, or
  geometry/classification logic.
- Public classes and functions should have docstrings that describe the
  contract, not just the implementation.
- Keep docstrings aligned with actual behavior. If a function only
  supports 2-D and 3-D, the docstring should say so explicitly.
- When examples are useful, prefer minimal, runnable parameter examples.


## Testing Standards

- Every bug fix should come with a regression test unless the fix is
  purely documentation-only or unreachable defensive code.
- Prefer contract-level assertions over implementation-detail
  assertions.
- Do not overconstrain tests with details that are not part of the API.
  Example: prefer boundary names over raw marker integers.
- When a name-based API exists, tests should resolve tags/facets/cells
  through the public name-based helpers.
- Separate focused unit tests from heavier integration tests when
  practical.
- Use `# pragma: no cover` only for truly defensive, effectively
  unreachable, or environment-dependent branches. Do not use it to hide
  reachable logic that should be tested.
- For `numba.njit` functions, coverage may miss the compiled body. When
  branch coverage matters, add tests against `.py_func` in addition to
  normal runtime tests.
- For MPI/distributed logic, add dedicated MPI-aware tests when behavior
  depends on rank ownership, communication, or distributed layouts.


## Coverage Standards

- Aim to test reachable behavior rather than suppress coverage noise.
- Before excluding code from coverage, decide whether the branch
  represents:
  - real user-visible behavior that should be tested
  - a private defensive guard that is not a meaningful contract
- If a branch is protected only because of a test harness limitation,
  prefer improving the test first.


## Mesh-Specific Standards

- Mesh generators live in `src/femcubes/mesh_generators/`, one module per
  `MeshType`; only the shared infrastructure (`MeshGeneratorBase`,
  `GeneratedMeshResult`, `model_to_mesh`) stays in
  `src/femcubes/core/mesh/generator_base.py`. Helpers shared by several
  generators go in private (`_`-prefixed) modules of that package.
- Mesh generators should return a structured result object rather than
  mutating the mesh wrapper directly.
- Every mesh generator should provide `Sample_Parameters()` with a
  minimal valid configuration.
- Mesh tests should use `Sample_Parameters()` rather than inventing local
  parameter objects when possible.
- Boundary/domain names are the supported API. Numeric tag values are
  implementation details unless explicitly documented otherwise.
- Mesh wrapper code should populate markers through the `_MeshGroups`
  public methods, not by directly replacing dictionaries.
- New mesh generators should be integrated in all of the following:
  - generator registry discovery (automatic for public modules of
    `src/femcubes/mesh_generators/`)
  - mesh generation tests (`src/femcubes/mesh_generators/tests/`)
  - mesh generator documentation (`docs/api/mesh_generators.rst`)
  - examples if they are the intended consumer


## Output and Plotting Standards

- Plotting and file output should share neutral field-preparation helpers
  rather than depending on each other’s internal APIs.
- Field/component parsing should be explicit. Do not infer component
  access from arbitrary underscores in field names.
- Destructive output-directory cleanup is allowed in this repository, but
  it must be guarded. Use the output-dir marker mechanism rather than
  deleting arbitrary directories.


## Documentation Standards

- Update documentation whenever a public API, mesh generator, example,
  or supported parameter set changes.
- Mesh generator documentation should include:
  - supported dimensions
  - required/optional parameters
  - named markers
  - a sample `ParamsTemplate`
- Keep API docs and example scripts synchronized with the actual code.
- Repo-internal guidance like this file should stay outside the
  published Sphinx docs unless there is a reason to expose it to users.


## Examples and Reference Scripts

- Example scripts should prefer real geometry/model inputs over ad hoc
  workarounds when the geometry is part of the intended physical setup.
- If an example relies on a custom mesh feature, prefer implementing it
  in a generator rather than encoding it as a post-hoc field patch when
  the shape should exist in the mesh itself.
- When changing an example’s setup, preserve its main pedagogical
  purpose and keep filenames/output paths stable unless there is a clear
  reason not to.


## When Editing Existing Code

- First check whether there is already a nearby file following the
  desired pattern. Copy the established repo style rather than inventing
  a new one.
- If a refactor changes a module’s internal organization, update the
  tests so they still emphasize the public/private split correctly.
- If a change exposes a design inconsistency, prefer resolving it in the
  local subsystem rather than layering more exceptions on top.


## Commits

- Do not add AI attribution to commits or pull requests: no
  `Co-Authored-By` lines naming an AI model, no "generated with" footers,
  and no mention of the tool in the commit message.
- The person who commits is responsible for every change that goes into
  `main`, whichever tools were used to write it. The commit history
  should name that person only.
- Write commit messages that describe the change itself: what was
  changed and, where it is not obvious, why.


## Practical Checklist

Before finishing a non-trivial edit, check:
- Is the public API still explicit and consistent?
- Are invalid inputs rejected clearly?
- Are tests checking the right contract, not the wrong detail?
- Did any docs/examples need updating?
- Is there any duplicated logic that should have been routed through an
  existing helper?
