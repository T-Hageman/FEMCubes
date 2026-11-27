# Changelog

All notable changes to FEMCubes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the version
numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is 0, minor releases may change the public interface.

## [Unreleased]

## [0.1.0] - unreleased

First public release. FEMCubes continues the code base that was developed
under the name IceCubes.

### Added

- Core framework (`femcubes.core`): mesh wrapper with named boundaries and
  subdomains, quadrature fields, adaptive mesh refinement with MPI
  repartitioning, output and plotting; physics assembly from model-type
  strings; staggered and monolithic nonlinear solvers with adaptive time
  stepping; direct and iterative linear solvers through PETSc.
- Solid mechanics (`femcubes.solids`): linear elastic, viscoelastic and Stokes
  models; phase-field fracture; strength-based cohesive phase-field fracture
  for elastic and viscoelastic solids, with the integration-point update in
  C++ kernels and in a reduced (invariant) form.
- Boundary conditions (`femcubes.boundary`): prescribed values and rates,
  external forces, displacement control and torsion.
- Glaciology (`femcubes.glaciology`): ice material laws, shallow-ice and
  shallow-shelf flow, basal and floating-ice friction, ocean pressure.
- Electrochemistry (`femcubes.electrochemistry`): electrolyte transport,
  metal-electrolyte interfaces, passivating surfaces, hydrogen diffusion.
- Poromechanics (`femcubes.poromechanics`): pore pressure and fracture flow.
- Mesh generators (`femcubes.mesh_generators`) for the bundled examples.
- Entry-point group `femcubes.plugins` for external model packages.
- Examples for fracture, glaciology and electrochemistry, each covered by a
  smoke test.
