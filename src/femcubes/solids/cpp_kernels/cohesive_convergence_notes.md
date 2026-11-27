# Cohesive Viscoelastic Convergence Notes

These notes record the numerical behavior observed while adding plane-stress
support and stabilizing the cohesive viscoelastic return mapping. They are an
implementation reference for future changes to:

- cohesive_viscoelastic/cohesive_viscoelastic_kernels_internal.cpp
- cohesive_viscoelastic/cohesive_viscoelastic_kernels.h
- shared/fracture_relations.h

The main lesson is that local convergence depends on the residual equations,
linear solver, tangent construction, scaling, line search, and continuation
path being mutually consistent. Improving only one was generally not enough.

## Local System

The plane-strain state is:

~~~text
x = [sigma(6), Lambda_v, lambda_c]
~~~

The plane-stress state is:

~~~text
x = [sigma(6), Lambda_v, lambda_c, e_zz]
~~~

Plane stress is imposed by adding R_zz = sigma_zz = 0 to the same local
nonlinear system as the stress, viscous multiplier, and cohesive multipliers.
The converged e_zz is written back to the total-strain field so plotting and
post-processing use the actual three-dimensional strain state.

Direct elastic plane-stress condensation is only sufficient while the material
response is purely elastic. Once viscous and cohesive strains depend on stress,
e_zz must participate in the nonlinear return mapping unless a separate nested
scalar solve is implemented with an equally consistent tangent.

## Approaches That Worked

### Solve the physical mechanisms together

The accepted state solves stress equilibrium, viscous flow, cohesive
complementarity, and sigma_zz = 0 together. Freezing viscous deformation from
the old stress was considered but not used. It would introduce an operator
split and make the returned tangent inconsistent with the actual update.

Keeping a cold start for each constitutive call is useful. It prevents an
inaccurate state from an earlier global trial from becoming the initial state
of a later trial. State is carried only between internal continuation stages
within one constitutive call.

### Use a full-system Schur partition for plane stress

The stable plane-stress partition is:

~~~text
y = [sigma(6), e_zz]
v = [Lambda_v]
c = [lambda_c]
~~~

The solve order is:

1. Eliminate the coupled [sigma, e_zz] saddle-point block.
2. Eliminate the scalar viscous multiplier block.
3. Solve the cohesive block last.
4. Recover Lambda_v, then [sigma, e_zz].

This mirrors the successful plane-strain logic after incorporating the
plane-stress constraint into the stress block. It also uses the physical zero
diagonal of the sigma_zz constraint without ending the solve with a division
by an e_zz scalar Schur pivot.

The Newton solve and consistent-tangent inverse must use the same partition.
For the tangent, solve all right-hand sides together and transform the top six
inverse rows back to physical units.

### Equilibrate before forming Schur products

Equilibrating individual blocks was not sufficient. A block solve could return
finite values while products such as:

~~~text
K_cy * inverse(K_yy) * K_yc
~~~

overflowed or suffered destructive cancellation in physical units. This
produced an entirely NaN final cohesive Schur block.

The effective procedure is:

1. Row-equilibrate the complete local Jacobian.
2. Column-equilibrate the row-scaled complete Jacobian.
3. Form every plane-stress Schur product in the equilibrated system.
4. Transform the Newton direction or inverse rows back to physical units.
5. Apply iterative refinement and validate the residual in original units.

Block solves still use pivoted QR and local equilibration. Rank-deficient
minimum-norm solutions are accepted only when their original block residual is
compatible.

### Keep numerical damping out of the physical tangent

The e_zz equation retains artificial diagonal damping while the Newton iterate
is far from the solution. This improved nonlinear convergence.

The physical equation is still R_zz = sigma_zz, whose direct derivative with
respect to the e_zz state is zero. Therefore:

- remove the artificial K_zz term during the final Newton polish;
- always set it to zero when constructing the algorithmic tangent;
- include sigma_zz in the convergence test.

Leaving damping in the tangent gives a derivative of the numerical
stabilization rather than the constitutive update.

### Use complementarity for cohesive activation

A Fischer-Burmeister residual is used for each cohesive multiplier. This gives
a semismooth equation for:

~~~text
lambda_c >= lambda_min
cohesive_drive >= 0
(lambda_c - lambda_min) * cohesive_drive = 0
~~~

Important details are:

- evaluate inactive-mode drive using a small multiplier probe so the
  regularized direction does not collapse at the multiplier floor;
- use a damaged-strength-based conversion between multiplier distance and
  traction units;
- freeze the active equations while testing one line-search direction;
- update the active set only after accepting the step;
- use hysteresis around the activation threshold.

The failure-surface branch must be selected from the pre-cohesive strain state.
Selecting it from the final elastic strain allowed a growing cohesive opening
to switch its own potential into the compression branch and created spurious
roots.

### Scale residuals by their physical blocks

Fixed tolerances in Pa were unreliable after damage reduced material strength
by several orders of magnitude. Convergence and line-search merit now use the
same nondimensional residual:

~~~text
stress residual       / elastic stress scale
viscous residual      / viscous multiplier scale
cohesive residual     / current damaged strength scale
sigma_zz residual     / elastic stress scale
~~~

The damaged-strength denominator has only a very small floor based on the
undamaged strength. A large elastic-modulus floor would make a significant
cohesive error appear converged in damaged material.

Recovery tolerances remain block-specific and bounded. They must not allow a
small stress residual to hide a materially large cohesive residual.

### Preserve bounds through the step length

Projecting multipliers after a Newton update changes the computed direction
and can invalidate the line-search model. The more stable approach is:

- hold a multiplier fixed if it is already at its bound and the direction
  points outside;
- calculate a fraction-to-boundary maximum step;
- backtrack from that admissible step;
- use a short non-monotone merit history to pass mild active-set transitions.

### Use more than one continuation path

No single continuation path was best for all failure surfaces.

The recovery hierarchy is:

1. Attempt the exact cold solve.
2. Increase trial elastic strain to the target while holding target dt and
   damage fixed.
3. If needed, ramp strain, dt, and damage together.
4. If needed, use an ordered path:
   - increase trial elastic strain at reduced dt and undamaged strength;
   - increase dt geometrically at fixed target strain;
   - reduce damage to its target at fixed strain and dt.
5. Finish with exact target parameters and construct the tangent there.

The simultaneous path was important for compression-dominated r2_DP cases.
The ordered path was important for damaged, near-hydrostatic r1 and r1_DP
plane-stress cases. Retaining both was more reliable than selecting one
globally.

Continuation progress is adaptive: failed stages halve the progress step and
successful stages increase it. A failed stage does not modify the accepted
continuation state.

### Regularize directions relative to strain magnitude

The threshold for using a normalized deviatoric strain direction must be
relative to the total strain norm:

~~~text
||dev(e)||^2 > relative_epsilon^2 * max(||e||^2, regularization)
~~~

Using a fixed absolute threshold caused inconsistent behavior between small
and large strain states, particularly near hydrostatic loading.

## Approaches That Did Not Work

### Directly solve the full unscaled plane-stress matrix

A generic direct solve was sensitive to the many orders of magnitude between
elastic moduli, multiplier equations, damaged strength, and the plane-stress
constraint. It returned noisy directions and tangents in high-damage states.

### Eliminate e_zz as the final scalar

The initial plane-stress Schur solver eliminated stress, viscous multiplier,
and cohesive multipliers before dividing by a final scalar e_zz Schur
complement. A small or cancellation-dominated final pivot introduced a failure
mode absent from plane strain.

### Eliminate [Lambda_v, e_zz] together after stress

This avoided the final scalar division but remained unstable near hydrostatic
tension. The reduced two-by-two block could become rank deficient, and its
subsequent cohesive Schur block became non-finite. In the exhaustive sweep this
increased rather than reduced the number of failures.

Grouping [sigma, e_zz] instead was more stable because the plane-stress
constraint remained attached to the constitutive stress equations.

### Equilibrate only the stress or final block

Local block scaling improved individual factorizations but did not prevent
overflow in Schur products formed afterward in physical units. Full-system
equilibration must occur before any plane-stress reduction.

### Rely on regularization after NaNs appear

Adding a diagonal to the final cohesive block cannot recover a block that is
already NaN. Regularization is useful for a finite nearly singular generalized
Jacobian, but it is not a substitute for preventing overflow and cancellation
upstream.

### Use fixed absolute convergence tolerances

A one-Pa tolerance can be negligible for undamaged elasticity and significant
relative to residual damaged strength. It also gives the line search a merit
function inconsistent with the convergence decision.

### Ignore sigma_zz in convergence

The in-plane blocks can be converged while the plane-stress constraint is not.
Both the raw sigma_zz residual and its scaled value are useful diagnostics.

### Change active equations during line-search trials

Changing the active set for each trial alpha means the line search is not
evaluating the residual associated with the Newton direction it was given.
This caused discontinuous merit values and avoidable rejection of useful
steps.

### Use only time-increment continuation

Changing only dt does not directly control mechanical activation of cohesive
modes. Trial-elastic-strain continuation is the primary load continuation.
Time and damage homotopies are secondary recovery tools.

### Use only ordered continuation

Although physically intuitive, separating strain, time, and damage ramps
regressed a small set of r2_DP compression cases. The simultaneous path must
remain available before the ordered fallback.

### Print every failed internal attempt

Direct and intermediate continuation attempts can fail before a later path
succeeds. Printing those attempts as a NaN fallback was misleading and noisy
under MPI. Diagnostics are now emitted only if every recovery path fails and
the integration point actually returns NaNs.

## Failure Handling and MPI

A local failure must not throw out of one rank while other ranks continue into
collective operations. The integration-point boundary therefore converts an
unrecoverable local failure into NaN output. The global solver can then reject
the trial and reduce the time step consistently.

The diagnostic for the first actual fallback includes:

- failure reason;
- local iteration count;
- plane-stress mode, damage, and dt;
- total strain and old viscous strain;
- raw residual norms by block;
- scaled residual norms by block;
- sigma_zz for plane stress.

Intermediate continuation failures are discarded without changing the last
accepted local state or printing a false final-failure message.

## Tangent Invariants

When modifying the local system, preserve these properties:

1. K is the derivative of the residual equations used at the converged state.
2. The tangent uses the same active branch and complementarity derivatives as
   the final residual.
3. Artificial e_zz damping is absent from the physical tangent.
4. Plane-stress inverse rows are transformed correctly after full-system row
   and column equilibration.
5. Iterative refinement is checked in original units.
6. A continuation result is returned only after solving the exact target
   strain, dt, and damage.

Finite-difference tangent checks should be run at high damage, where scaling
and branch consistency are most likely to be exposed.

## Testing Strategy

Focused kernel tests are in:

~~~text
femcubes/solids/tests/test_cohesive_viscoelastic_kernels.py
~~~

They cover:

- elastic plane-stress e_zz and tangent;
- plane-strain behavior;
- finite-difference plane-stress tangent consistency;
- active high-damage updates;
- below-strength inactive behavior;
- representative active-set and continuation stalls;
- large-dt r2_DP recovery in both plane modes.

The long stability sweep is:

~~~text
femcubes/solids/tests/test_cohesive_viscoelastic_stability_full_model.py
~~~

It is marked full_model and covers:

- r1, r2, r1_DP, and r2_DP;
- plane strain and plane stress;
- ten time-increment sizes;
- six damage levels;
- six history states;
- zero, uniaxial, biaxial, shear, mixed, confined, and near-hydrostatic
  strain paths over a broad magnitude range.

The final verified sweep contained 521,280 constitutive updates with no
non-finite output or invalid multiplier. The focused suite contained 14
passing tests.

Always rebuild the native extension before running these tests. Otherwise
pytest may import an older shared library and give results unrelated to the
current C++ source.

## Debugging Checklist

For a new convergence failure:

1. Reproduce one integration point with the exact strain, old history,
   damage, dt, criterion, and plane mode.
2. Determine whether the return-map state or only the tangent is non-finite.
3. Inspect raw and scaled residuals separately by block.
4. Check sigma_zz independently in plane stress.
5. Check whether F and K are finite before the linear solve.
6. Check reduced Schur blocks before attempting diagonal regularization.
7. Confirm the active set is frozen during line-search trials.
8. Confirm multipliers remain inside their bounds without projection.
9. Test both continuation paths.
10. Add the exact failure as a focused regression before rerunning the full
    sweep.

Do not weaken tolerances solely to make the sweep pass. First establish
whether the residual is physically small, whether the Jacobian is consistent,
and whether numerical scaling is hiding the actual source of error.
