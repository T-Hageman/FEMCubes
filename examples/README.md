# Examples

Runnable scripts that show how to set up and run FEMCubes models, grouped by
field. Each script can be run from any folder, in serial or with MPI:

```bash
python examples/fracture/cohesive_phasefield.py --case dynamic
mpirun -n 8 python examples/glaciology/ice_cliffs.py --case ve_cohesive
```

All scripts accept the same options:

| Option | Meaning |
|---|---|
| `--case NAME` | case to run, for scripts with several cases (`--help` lists them) |
| `--quick` | short, coarse run that checks the script works; it does not reproduce the results |
| `--no-plots` | do not create or save figures |
| `--output-dir DIR` | folder for figures and results (default: `results/<script>[/<case>]` next to the script) |

Some scripts have further options, such as the loading rate or the
phase-field length scale; `--help` lists them. Each script also offers
`build_params(...)` and `run(...)` functions, so its set-up can be reused
from Python.

## Fracture

| Script | Cases | Shows |
|---|---|---|
| `fracture/cohesive_phasefield.py` | `full_plate`, `single_edge_notched`, `dynamic` | cohesive phase-field fracture with a strength surface; adaptive refinement; dynamic branching |
| `fracture/viscoelastic_fracture.py` | -- | creep and cohesive fracture of an ice plate with a hole (paper benchmark); options `--rate`, `--gc`, `--ell` |
| `fracture/multidomain_tensile_bar.py` | -- | models acting on subdomains; phase field only in the notched centre |

## Glaciology

| Script | Cases | Shows |
|---|---|---|
| `glaciology/ice_cliffs.py` | `le_cohesive`, `ve_cohesive`, `le_split`, `ve_split`, `stokes`, `glen` | crevasses at a grounded cliff as the water in front of it is lowered (the ice-cliff case of the cohesive viscoelastic paper, smaller), six rheologies; options `--ell`, `--water-end`, `--base`, `--friction` |
| `glaciology/shallow_ice.py` | `slope`, `flat` | shallow-ice/shallow-shelf flow with a level-set ice margin |
| `glaciology/tensile_ice.py` | `stokes`, `glen`, `le`, `ve`, `le_cohesive`, `ve_cohesive` | tension of a notched ice block with six rheologies; options `--ell`, `--n-notches` |

## Electrochemistry

| Script | Cases | Shows |
|---|---|---|
| `electrochemistry/corrosion_pit.py` | -- | corrosion of a pit under charge conservation; the metal potential as a global unknown |
| `electrochemistry/electrolyte_metal.py` | -- | an electrolyte next to a loaded steel plate, coupled by hydrogen evolution and absorption; option `--no-interface` |

## Smoke tests

`test_examples.py` runs every case with `--quick` on two MPI ranks and checks
that it finishes and saves its figures:

```bash
pytest examples/test_examples.py -n 8
```

These tests are marked `examples` and `slow`; `pytest -m "not examples"`
skips them.

Benchmarks that measure performance rather than show a model are in
[`../benchmarks`](../benchmarks).
