"""Passive level-set advection model for glaciology ice-margin tracking."""

import math

from ufl import CellDiameter, conditional, dot, grad, le, sin, sqrt
from dolfinx import fem
from mpi4py import MPI
import numpy as np

from femcubes.core.base_model import BaseModel
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.utils.mpi_utils import mprint


def smooth_level_set_heaviside(level_set, epsilon: float):
    """Return the smooth Heaviside phase indicator for the LS sign convention."""
    xi = level_set / epsilon
    return conditional(le(level_set, -epsilon), 0.0, conditional(le(epsilon, level_set), 1.0, 0.5 * (1.0 + xi + sin(math.pi * xi) / math.pi)))


class LevelSet(BaseModel):
    r"""Passive level-set advection of the ice margin with SUPG stabilization.

    The ice margin is represented implicitly as the zero contour of a level
    set :math:`\psi` (``LS``; :math:`\psi>0` in ice, :math:`\psi=0` at the
    margin, :math:`\psi<0` outside), so that the margin can advance and
    retreat across a fixed mesh :cite:p:`Osher1988`. The model transports :math:`\psi` with a
    velocity field of another model, in a narrow band around the margin, with
    backward Euler in time. As this pure advection problem contains no
    diffusion, a standard Galerkin discretization would give oscillatory
    solutions, and the streamline-upwind Petrov-Galerkin (SUPG) method is
    used instead :cite:p:`Brooks1982`.
    Advection gradually distorts :math:`\psi`, so it can be rebuilt as a
    signed distance to the margin at regular intervals. Used with
    :doc:`shallow_ice` and :doc:`shallow_ice_bc`; works in 2-D and 3-D.

    .. rubric:: Fields

    * ``LS``: level set :math:`\psi`, CG of order ``Field_Orders["LS"]``,
      solved in step ``Solution_Steps["LS"]`` (its own step).
    * ``LSOld``: committed :math:`\psi_n`.
    * ``LS_phase``: output phase indicator :math:`H_w(\psi)`, CG of the same
      order.
    * The velocity named by ``Velocity_Field`` (for example ``v`` of
      :doc:`shallow_ice`): read only.
    * ``h``: read only during reinitialization, if it exists.

    .. rubric:: Weak form

    With test function :math:`\delta\psi`, :math:`\Omega` the domain group
    ``domain``, :math:`w` = ``Level_Set["Transition_Width"]`` and the smooth
    Heaviside

    .. math::

        H_w(s) = \begin{cases} 0 & s \le -w\\
        \tfrac12\bigl(1 + \tfrac{s}{w} + \tfrac{1}{\pi}\sin\tfrac{\pi s}{w}\bigr) & |s| < w\\
        1 & s \ge w \end{cases}

    the contribution to the residual of ``LS`` is

    .. math::

        R_\psi = \int_\Omega \bigl(\delta\psi + \tau\,\mathbf{b}\cdot\nabla\delta\psi\bigr)\,r\,\mathrm{d}x,
        \qquad
        r = \frac{\psi-\psi_n}{\Delta t} + \mathbf{b}\cdot\nabla\psi,

    with the band-limited velocity
    :math:`\mathbf{b} = H_w\bigl(6w - \sqrt{\psi_n^2+\epsilon^2}\bigr)\,\mathbf{v}`,
    the SUPG parameter
    :math:`\tau = \bigl((2/\Delta t)^2 + (2\lVert\mathbf{b}\rVert_\epsilon/h_K)^2\bigr)^{-1/2}`,
    :math:`\lVert\mathbf{b}\rVert_\epsilon = \sqrt{\mathbf{b}\cdot\mathbf{b}+\epsilon^2}`,
    :math:`h_K` the cell diameter and :math:`\epsilon` =
    ``SUPG_Velocity_Regularization``. Outside the band :math:`\mathbf{b}=0`,
    so :math:`\psi` keeps its committed value there. The Jacobian is exact
    for a fixed velocity.

    .. rubric:: Outside the weak form

    * Before every assembly ``LS_phase`` is set to :math:`H_w(\psi)`.
    * ``commit``: every ``Reinitialize_Every``-th commit calls
      :meth:`reinitialize` (otherwise only ``LS_phase`` is updated), then
      copies ``LS`` to ``LSOld``. Reinitialization first sets
      :math:`\psi = -\max(|\psi|, w)` where ``h`` :math:`\le`
      ``Reinitialize_Min_Thickness`` (thin-ice retreat), then replaces
      :math:`\psi` by the signed distance to its zero contour, clipped to
      :math:`\pm` ``Reinitialize_Clip_Distance``.
    * ``reset_step`` restores ``LS`` from ``LSOld``.
    * ``initialize_fields`` does nothing: set ``LS`` and ``LSOld`` after
      ``Physics.Initialize_Fields``.
    * No Dirichlet conditions (see :doc:`shallow_ice_bc`).

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            Solution_Steps = {"v": 0, "h": 1, "LS": 2}  # REQUIRED: "LS" in its own step
            Field_Orders = {"v": 2, "h": 1, "LS": 1}    # REQUIRED: "LS", the CG order of LS
            Level_Set = {                    # REQUIRED
                "Transition_Width": 2000.0,  # REQUIRED: width w of the smooth Heaviside (m); band half-width 6 w
                "Reinitialize_Every": 10,    # OPTIONAL (default 0, never): commits between reinitializations
                "Reinitialize_Clip_Distance": None,  # OPTIONAL (default None, no clipping): |LS| limit (m)
                "Reinitialize_Min_Thickness": 1.0e-2,  # OPTIONAL (default 1e-2; None or <= 0 disables): thin-ice limit (m)
            }
            Models = {
                "IceMargin": {
                    "Type": "Glaciology:LevelSet",       # REQUIRED
                    "Velocity_Field": "v",               # REQUIRED: name of the advecting vector field
                    "SUPG_Velocity_Regularization": 1.0e-12,  # OPTIONAL (default 1e-12): epsilon (m/s)
                    "domain": "domain",                  # OPTIONAL (default "domain", the full mesh): domain group
                },
            }
    """

    MODEL_TYPE = "Glaciology:LevelSet"

    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        super().__init__(name, params, mesh)

        model_cfg = params.Models[name]
        if "Velocity_Field" not in model_cfg:
            raise ValueError("Glaciology:LevelSet requires a 'Velocity_Field' model setting.")
        if "LS" not in params.Solution_Steps:
            raise ValueError("Glaciology:LevelSet requires Solution_Steps['LS'].")
        if "LS" not in params.Field_Orders:
            raise ValueError("Glaciology:LevelSet requires Field_Orders['LS'].")

        self.velocity_field_name = model_cfg["Velocity_Field"]
        ls_config = params.Level_Set
        self.step_LS = params.Solution_Steps["LS"]
        self.velocity_regularization = model_cfg.get("SUPG_Velocity_Regularization", 1.0e-12)
        self.level_set_transition_width = ls_config.get("Transition_Width")
        self.advection_band_width = 6.0 * self.level_set_transition_width
        self.reinitialize_every = ls_config.get("Reinitialize_Every", 0)
        self.reinitialize_clip_distance = ls_config.get("Reinitialize_Clip_Distance", None)
        self.reinitialize_min_thickness = ls_config.get("Reinitialize_Min_Thickness", 1.0e-2)
        self._commit_count = 0
        self._reinitialize_count = 0
        self._last_thin_ice_removed_dofs = 0

        mesh.Add_Field("LS", "Level Set [-]", "CG", params.Field_Orders["LS"], domain=self.domain)
        mesh.Add_Field("LSOld", "Old Level Set [-]", "CG", params.Field_Orders["LS"], domain=self.domain)
        mesh.Add_Field("LS_phase", "Level Set Phase Indicator [-]", "CG", params.Field_Orders["LS"], domain=self.domain)
        mesh.Add_TestFunction("LS", self.step_LS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """Assemble the backward-Euler SUPG level-set advection equation."""
        if step != self.step_LS:
            return None

        LS = self.mesh.Get_Field("LS", step=self.step_LS)
        LSOld = self.mesh.Get_Field("LSOld")
        velocity = self.mesh.Get_Field(self.velocity_field_name)

        LS_t = self.mesh.Get_TestFunction("LS", self.step_LS)
        LS_tr = self.mesh.Get_Trial_Function("LS", self.step_LS)

        dx = self.mesh.Get_Domain_Integrator(self.domain)
        cell_h = CellDiameter(self.mesh.mesh)
        band_weight = self._get_advection_band_weight(LSOld)
        advection_velocity = band_weight * velocity
        velocity_norm = sqrt(dot(advection_velocity, advection_velocity) + self.velocity_regularization**2)
        tau = 1.0 / sqrt((2.0 / dt_constant) ** 2 + (2.0 * velocity_norm / cell_h) ** 2)

        residual = (LS - LSOld) / dt_constant + dot(advection_velocity, grad(LS))
        residual_tr = LS_tr / dt_constant + dot(advection_velocity, grad(LS_tr))
        streamline_test = dot(advection_velocity, grad(LS_t))

        F = LS_t * residual * dx
        K = LS_t * residual_tr * dx
        F += tau * streamline_test * residual * dx
        K += tau * streamline_test * residual_tr * dx

        return K, F

    def Update_Weak_Constants(self, time_constant, dt_constant, step: int) -> bool:
        self._update_phase_indicator()
        return False

    def initialize_fields(self) -> None:
        """Leave ``LS`` initialization to the user/example setup."""
        return None

    def commit(self) -> None:
        """Commit the level-set state for the next time step."""
        self._commit_count += 1
        if self._should_reinitialize():
            self.reinitialize()
        else:
            self._update_phase_indicator()
        self.mesh.Copy_Field("LS", "LSOld", domain=self.domain)

    def reset_step(self) -> None:
        """Restore ``LS`` from the last committed state."""
        self.mesh.Copy_Field("LSOld", "LS", domain=self.domain)
        self._update_phase_indicator()

    def reinitialize(self) -> bool:
        """Rebuild ``LS`` as a signed-distance field to its current zero contour.

        The current zero contour is extracted from sign changes in the CG
        level-set field on each local cell, gathered across MPI ranks, and
        used as a point cloud for nearest-distance reconstruction.  The sign
        of the original field is preserved.  Returns ``True`` when a zero
        contour was found and the field was updated.
        """
        LS = self.mesh.Get_Field_For_Output("LS").field
        old_values = np.asarray(LS.x.array, dtype=float).copy()
        dof_coords = np.asarray(LS.function_space.tabulate_dof_coordinates(), dtype=float)
        if old_values.size == 0:
            self._update_phase_indicator()
            return False

        corrected_values, removed_dofs = self._apply_thin_ice_retreat_filter(LS, old_values)
        self._last_thin_ice_removed_dofs = removed_dofs

        zero_points = self._collect_global_zero_contour_points(LS, corrected_values, dof_coords)
        if zero_points.size == 0:
            if removed_dofs > 0:
                LS.x.array[:] = corrected_values
                LS.x.scatter_forward()
            self._update_phase_indicator()
            return removed_dofs > 0

        distances = self._nearest_zero_contour_distances(dof_coords, zero_points)
        signs = np.where(corrected_values < 0.0, -1.0, 1.0)
        signs[np.isclose(corrected_values, 0.0, atol=self._zero_contour_tolerance(corrected_values))] = 0.0
        new_values = signs * distances
        if self.reinitialize_clip_distance is not None:
            new_values = np.clip(new_values, -self.reinitialize_clip_distance, self.reinitialize_clip_distance)

        local_max_change = float(np.max(np.abs(new_values - old_values))) if new_values.size else 0.0
        max_change = self.mesh.mesh.comm.allreduce(local_max_change, op=MPI.MAX)
        LS.x.array[:] = new_values
        LS.x.scatter_forward()
        self._update_phase_indicator()
        self._reinitialize_count += 1
        point_count = self.mesh.mesh.comm.allreduce(int(zero_points.shape[0]), op=MPI.MAX)
        mprint(f"LevelSet reinitialization: , count={self._reinitialize_count}, zero_points={point_count}, thin_ice_removed_dofs={removed_dofs}, max|dLS|={max_change:.3e} m")
        return True

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _should_reinitialize(self) -> bool:
        """PRIVATE: Return whether the current commit should reinitialize LS."""
        return self.reinitialize_every > 0 and self._commit_count % self.reinitialize_every == 0

    def _get_advection_band_weight(self, old_level_set):
        """PRIVATE: Return a smooth narrow-band weight for LS advection."""
        if self.advection_band_width is None:
            return 1.0

        abs_old_level_set = sqrt(old_level_set * old_level_set + self.velocity_regularization**2)
        return smooth_level_set_heaviside(self.advection_band_width - abs_old_level_set, self.level_set_transition_width)

    def _update_phase_indicator(self) -> None:
        """PRIVATE: Update ``LS_phase`` from the current ``LS`` field."""
        LS = self.mesh.Get_Field("LS")
        phase = smooth_level_set_heaviside(LS, self.level_set_transition_width)
        self.mesh.Set_From_Expression("LS_phase", phase, domain=self.domain)

    def _apply_thin_ice_retreat_filter(self, LS, level_set_values: np.ndarray) -> tuple[np.ndarray, int]:
        """PRIVATE: Force near-zero-thickness regions outside before reinitialization."""
        if self.reinitialize_min_thickness is None or self.reinitialize_min_thickness <= 0.0:
            return level_set_values.copy(), 0
        if "h" not in self.mesh.fields:
            return level_set_values.copy(), 0

        thickness_values = self._interpolate_thickness_to_level_set_space(LS)
        thin_ice = thickness_values <= self.reinitialize_min_thickness
        corrected_values = level_set_values.copy()
        removed = thin_ice & (corrected_values > 0.0)
        if np.any(thin_ice):
            corrected_values[thin_ice] = -np.maximum(np.abs(corrected_values[thin_ice]), self.level_set_transition_width)

        removed_count = int(np.count_nonzero(removed))
        removed_count = self.mesh.mesh.comm.allreduce(removed_count, op=MPI.SUM)
        return corrected_values, removed_count

    def _interpolate_thickness_to_level_set_space(self, LS) -> np.ndarray:
        """PRIVATE: Return the ice thickness sampled at level-set DOFs."""
        thickness_view = self.mesh.Get_Field_For_Output("h")
        thickness = thickness_view.collapse_component(0)
        if thickness.function_space.mesh is not LS.function_space.mesh:
            raise ValueError(f"LevelSet '{self.name}' reinitialization needs 'h' on the same domain as 'LS' ({self.domain!r}).")
        thickness_on_level_set = fem.Function(LS.function_space)
        expr = fem.Expression(thickness, LS.function_space.element.interpolation_points)
        thickness_on_level_set.interpolate(expr)
        thickness_on_level_set.x.scatter_forward()
        return np.asarray(thickness_on_level_set.x.array, dtype=float)

    def _collect_global_zero_contour_points(self, LS, values: np.ndarray, dof_coords: np.ndarray) -> np.ndarray:
        """PRIVATE: Gather zero-contour sample points from local cell sign changes."""
        local_points = self._collect_local_zero_contour_points(LS, values, dof_coords)
        chunks = self.mesh.mesh.comm.allgather(local_points)
        nonempty = [chunk.reshape((-1, dof_coords.shape[1])) for chunk in chunks if chunk.size > 0]
        if not nonempty:
            return np.empty((0, dof_coords.shape[1]), dtype=float)
        return np.vstack(nonempty)

    def _collect_local_zero_contour_points(self, LS, values: np.ndarray, dof_coords: np.ndarray) -> np.ndarray:
        """PRIVATE: Return LS=0 points found from owned local cells."""
        # LS may live on a submesh: loop over the cells of its own mesh.
        ls_mesh = LS.function_space.mesh
        tdim = ls_mesh.topology.dim
        cell_count = ls_mesh.topology.index_map(tdim).size_local
        dofmap = LS.function_space.dofmap
        tolerance = self._zero_contour_tolerance(values)
        points = []

        for cell in range(cell_count):
            cell_dofs = np.asarray(dofmap.cell_dofs(cell), dtype=np.int32)
            cell_values = values[cell_dofs]
            cell_coords = dof_coords[cell_dofs]
            zero_mask = np.isclose(cell_values, 0.0, atol=tolerance)
            if np.any(zero_mask):
                points.extend(cell_coords[zero_mask])

            for i in range(len(cell_dofs)):
                value_i = cell_values[i]
                for j in range(i + 1, len(cell_dofs)):
                    value_j = cell_values[j]
                    if value_i * value_j >= 0.0:
                        continue
                    alpha = abs(value_i) / (abs(value_i) + abs(value_j))
                    points.append((1.0 - alpha) * cell_coords[i] + alpha * cell_coords[j])

        if not points:
            return np.empty((0, dof_coords.shape[1]), dtype=float)
        return np.asarray(points, dtype=float).reshape((-1, dof_coords.shape[1]))

    def _nearest_zero_contour_distances(self, dof_coords: np.ndarray, zero_points: np.ndarray) -> np.ndarray:
        """PRIVATE: Return nearest distances from DOFs to the zero-contour point cloud."""
        distances = np.empty(dof_coords.shape[0], dtype=float)
        chunk_size = 4096
        for start in range(0, dof_coords.shape[0], chunk_size):
            stop = min(start + chunk_size, dof_coords.shape[0])
            delta = dof_coords[start:stop, None, :] - zero_points[None, :, :]
            distances[start:stop] = np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))
        return distances

    def _zero_contour_tolerance(self, values: np.ndarray) -> float:
        """PRIVATE: Return a scale-aware tolerance for detecting nodal zeros."""
        scale = float(np.max(np.abs(values))) if values.size else 1.0
        return max(1.0e-12 * max(scale, 1.0), 1.0e-14)
