"""Plots of the strength surfaces of the cohesive solid models."""

from pathlib import Path

import numpy as np

from femcubes.core.utils.mpi_utils import rank
from femcubes.solids.cpp_kernels.cohesive_viscoelastic._cohesive_viscoelastic_kernels_cpp import (
    compute_strength_surface_point,
)


def plot_FailureSurface(material_params, srange, save_path=None, export_hdf5=False, true_2d=False):
    """Save cohesive strength contours in plane-stress, plane-strain, intrinsic 2-D, invariant and 3-D stress space.

    The strength surface bounds the stress states that intact material can
    carry: cohesive eigenstrains, and with them cracks, develop where the
    stress reaches it. Plotting it in several stress spaces shows how the
    chosen failure type and strengths act under multi-axial loading, for
    example before a simulation is started.

    The contour is the zero level of the strength indicator of the failure
    potential of :doc:`cohesive_viscoelastic` (and of the other cohesive
    solids), evaluated by the shared fracture relations of the C++
    kernel: the polar ratio of the stress over the admissible cohesive
    directions, including the unilateral treatment of compression. For the
    strain-dependent ``r2_2L_DP`` potential this is a finite-probe onset
    diagnostic; its pure deviatoric direction gives the exact initial
    Drucker-Prager envelope. For ``r2_2L_PSD`` it is the exact opening onset
    of the update; that criterion has no intrinsic 2-D form, so its intrinsic
    2-D panel stays empty. Only rank 0 plots.

    The plane-stress, plane-strain, invariant and 3-D panels use the 3-D
    elastic constants, the intrinsic 2-D panel the intrinsic 2-D ones. With
    ``true_2d=False``, ``Lame``, ``Shear`` and ``Bulk`` of ``material_params``
    are the 3-D constants, and the intrinsic 2-D panel uses ``Lame`` and
    ``Shear`` as a ``true_2d`` model would. With ``true_2d=True`` they are
    the constants of the intrinsic 2-D material (``Bulk`` is not read), and
    the 3-D constants are those of the same Young's modulus and Poisson
    ratio, :math:`\\lambda = 2\\mu\\lambda_{2D}/(2\\mu-\\lambda_{2D})`. The
    out-of-plane stress of the plane-strain panel uses ``Poisson`` if
    given, else the Poisson ratio of the 3-D constants.

    Parameters
    ----------
    material_params:
        Material parameter dictionary containing the cohesive material definition.
    srange:
        One-dimensional stress sampling range in Pa.
    save_path:
        Output image path. Parent directories are created automatically.
    export_hdf5:
        If ``True``, also write a sibling HDF5 file with the same stem as
        ``save_path``. The file contains the sampled failure criteria grids,
        the invariant-space grid, and the 3-D surface vertices and faces.
        Stress coordinates are stored in Pa. ``save_path`` is required when
        this option is enabled.
    true_2d:
        Whether the elastic constants of ``material_params`` are those of the
        intrinsic 2-D material (a model with ``true_2d=True``).
    """

    if rank != 0:
        return
    if save_path is None:
        if export_hdf5:
            raise ValueError("save_path is required when export_hdf5=True.")
        return

    from skimage.measure import marching_cubes
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    srange = np.asarray(srange, dtype=float)
    if srange.ndim != 1 or srange.size < 2:
        raise ValueError("srange must be a one-dimensional array with at least two values.")
    save_path = Path(save_path)
    if not save_path.name:
        raise ValueError("save_path must point to an output file.")

    failure_type = material_params.get("FailureType", "r1")
    shear = float(material_params["Shear"])
    if true_2d:
        lame_2d = float(material_params["Lame"])
        if not 0.0 <= lame_2d < 2.0 * shear:
            raise ValueError("The intrinsic 2-D Lame parameter must satisfy 0 <= Lame < 2 Shear.")
        lame = 2.0 * shear * lame_2d / (2.0 * shear - lame_2d)
        bulk = lame + 2.0 * shear / 3.0
    else:
        bulk = float(material_params["Bulk"])
        lame = float(material_params.get("Lame", bulk - 2.0 * shear / 3.0))
        lame_2d = lame
    bulk_2d = lame_2d + shear
    poisson = float(material_params.get("Poisson", lame / (2.0 * (lame + shear)) if shear > 0.0 else 0.0))

    f_t = float(material_params["f_t"])
    f_s = float(material_params["f_s"])
    dp_phi = float(material_params.get("DP_phi", 0.0))
    dp_q = float(material_params.get("DP_q", f_s))

    stress_mat = np.ones((3, 3), dtype=float) * lame + 2.0 * shear * np.eye(3, dtype=float)
    stress_to_strain = np.linalg.inv(stress_mat)
    stress_mat_2d = np.array(
        [[lame_2d + 2.0 * shear, lame_2d], [lame_2d, lame_2d + 2.0 * shear]],
        dtype=float,
    )
    stress_to_strain_2d = np.linalg.inv(stress_mat_2d)

    stress_scale = 1.0e6

    def _evaluate_surface(stress, intrinsic_2d=False):
        stress_vec = np.zeros((6, 1), dtype=float)
        stress_vec[0:3, 0] = np.asarray(stress, dtype=float)

        strain_vec = np.zeros((6, 1), dtype=float)
        if intrinsic_2d:
            strain_vec[0:2, 0:1] = stress_to_strain_2d @ stress_vec[0:2, 0:1]
        else:
            strain_vec[0:3, 0:1] = stress_to_strain @ stress_vec[0:3, 0:1]

        failure_bulk = bulk_2d if intrinsic_2d else bulk
        return compute_strength_surface_point(
            f_t, f_s, failure_bulk, dp_phi, dp_q, failure_type,
            stress_vec.flatten(), strain_vec.flatten(), true_2d=intrinsic_2d,
        )

    s1_grid, s2_grid = np.meshgrid(srange, srange, indexing="xy")
    plane_stress = np.empty_like(s1_grid)
    plane_strain = np.empty_like(s1_grid)
    intrinsic = np.empty_like(s1_grid)

    for i in range(s1_grid.shape[0]):
        for j in range(s1_grid.shape[1]):
            s1 = s1_grid[i, j]
            s2 = s2_grid[i, j]
            plane_stress[i, j] = _evaluate_surface((s1, s2, 0.0))
            s3 = poisson * (s1 + s2)
            plane_strain[i, j] = _evaluate_surface((s1, s2, s3))
            # r2_2L_PSD is available for three stress components only
            intrinsic[i, j] = (np.nan if failure_type == "r2_2L_PSD"
                               else _evaluate_surface((s1, s2, 0.0), intrinsic_2d=True))

    # Build an independent p-q grid (not mapped from sigma-space samples).
    # Use axisymmetric principal stresses to embed each (p, q) point in 3-D:
    #   sigma1 = -p + 2q/3, sigma2 = sigma3 = -p - q/3.
    p_max = float(np.max(np.abs(srange)))
    q_max = np.sqrt(3.0) * p_max
    n_pq = max(80, int(srange.size))
    p_axis = np.linspace(-p_max, p_max, n_pq)
    q_axis = np.linspace(0.0, q_max, n_pq)
    p_grid, q_grid = np.meshgrid(p_axis, q_axis, indexing="xy")
    pq_surface = np.empty_like(p_grid)
    for i in range(p_grid.shape[0]):
        for j in range(p_grid.shape[1]):
            p_val = p_grid[i, j]
            q_val = q_grid[i, j]
            s1 = -p_val + (2.0 / 3.0) * q_val
            s2 = -p_val - (1.0 / 3.0) * q_val
            s3 = s2
            pq_surface[i, j] = _evaluate_surface((s1, s2, s3))

    # Sample sigma-space directly and extract the zero strength-indicator surface.
    n_sig = max(14, int(srange.size // 2))
    sigma_axis = np.linspace(float(np.min(srange)), float(np.max(srange)), n_sig)
    sigma_surface = np.empty((n_sig, n_sig, n_sig), dtype=float)
    for i, s1 in enumerate(sigma_axis):
        for j, s2 in enumerate(sigma_axis):
            for k, s3 in enumerate(sigma_axis):
                sigma_surface[i, j, k] = _evaluate_surface((s1, s2, s3))

    surface_vertices = np.empty((0, 3), dtype=float)
    surface_faces = np.empty((0, 3), dtype=np.int32)

    fig = Figure(figsize=(15.0, 10.0), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax_ps = fig.add_subplot(2, 3, 1)
    ax_pe = fig.add_subplot(2, 3, 2)
    ax_2d = fig.add_subplot(2, 3, 3)
    ax_pq = fig.add_subplot(2, 3, 4)
    ax_3d = fig.add_subplot(2, 3, (5, 6), projection="3d")

    ax_ps.contour(s1_grid/stress_scale, s2_grid/stress_scale, plane_stress, levels=[0.0], colors="C0", linewidths=1.5)
    ax_ps.set_title(r"Plane stress, $\sigma_3=0$")
    ax_ps.set_xlabel(r"$\sigma_1$ [MPa]")
    ax_ps.set_ylabel(r"$\sigma_2$ [MPa]")

    ax_pe.contour(s1_grid / stress_scale, s2_grid / stress_scale, plane_strain, levels=[0.0], colors="C1", linewidths=1.5)
    ax_pe.set_title(r"Plane strain, $\varepsilon_3=0$")
    ax_pe.set_xlabel(r"$\sigma_1$ [MPa]")
    ax_pe.set_ylabel(r"$\sigma_2$ [MPa]")

    ax_2d.set_title(r"True 2-D, $\sigma_3=0$")
    if np.isfinite(intrinsic).any():
        ax_2d.contour(s1_grid / stress_scale, s2_grid / stress_scale, intrinsic, levels=[0.0], colors="C4", linewidths=1.5)
        ax_2d.set_xlabel(r"$\sigma_1$ [MPa]")
        ax_2d.set_ylabel(r"$\sigma_2$ [MPa]")
    else:
        # r2_2L_PSD has no intrinsic 2-D form (it needs the three principal directions)
        ax_2d.text(0.5, 0.5, f"Not available:\n{failure_type} has no\nintrinsic 2-D form", ha="center",
                   va="center", transform=ax_2d.transAxes)
        ax_2d.set_xticks([])
        ax_2d.set_yticks([])

    ax_pq.contour(
        p_grid / stress_scale,
        q_grid / stress_scale,
        pq_surface,
        levels=[0.0],
        colors="C2",
        linewidths=1.5,
    )
    ax_pq.set_title(r"Invariant space, $p$-$q$")
    ax_pq.set_xlabel(r"$p$ [MPa]")
    ax_pq.set_ylabel(r"$q \; (|\sigma_{dev}|)$ [MPa]")

    s_min = float(sigma_axis[0])
    s_max = float(sigma_axis[-1])
    if float(np.min(sigma_surface)) <= 0.0 <= float(np.max(sigma_surface)):
        ds = float((s_max - s_min) / max(n_sig - 1, 1))
        verts, faces, _, _ = marching_cubes(
            sigma_surface,
            level=0.0,
            spacing=(ds / stress_scale, ds / stress_scale, ds / stress_scale),
        )
        # marching_cubes vertices start at origin; shift to actual sigma range.
        shift = s_min / stress_scale
        verts = verts + np.array([shift, shift, shift], dtype=float)
        surface_vertices = verts * stress_scale
        surface_faces = np.asarray(faces, dtype=np.int32)
        ax_3d.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            faces,
            verts[:, 2],
            color="C3",
            alpha=0.85,
            linewidth=0.0,
            antialiased=True,
            shade=True,
        )

    ax_3d.set_title(r"$\sigma_1$-$\sigma_2$-$\sigma_3$ strength surface")
    ax_3d.set_xlabel(r"$\sigma_1$ [MPa]")
    ax_3d.set_ylabel(r"$\sigma_2$ [MPa]")
    ax_3d.set_zlabel(r"$\sigma_3$ [MPa]")

    for ax in (ax_ps, ax_pe, ax_2d, ax_pq):
        ax.grid(True, ls=":")
        ax.set_aspect("equal")
    ax_3d.grid(True)

    title = f"{failure_type} failure surface"
    if failure_type in {"r1_DP", "r2_DP", "r2_2L_DP"}:
        title += f" (phi={dp_phi:.1f} deg, q_max={dp_q / stress_scale:.1f} MPa)"
    fig.suptitle(title)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.clear()

    if export_hdf5:
        import h5py

        hdf5_path = save_path.with_suffix(".hdf5")
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(hdf5_path, "w") as h5f:
            h5f.attrs["format"] = "FEMCubes cohesive failure surface"
            h5f.attrs["format_version"] = 1
            h5f.attrs["failure_type"] = failure_type
            h5f.attrs["stress_units"] = "Pa"
            h5f.attrs["criterion_zero_level"] = 0.0

            for name, sigma_3, criterion in (
                ("plane_stress", np.zeros_like(s1_grid), plane_stress),
                ("plane_strain", poisson * (s1_grid + s2_grid), plane_strain),
                ("true_2d", np.zeros_like(s1_grid), intrinsic),
            ):
                group = h5f.create_group(name)
                group.create_dataset("sigma_1", data=s1_grid)
                group.create_dataset("sigma_2", data=s2_grid)
                group.create_dataset("sigma_3", data=sigma_3)
                group.create_dataset("failure_criterion", data=criterion)

            invariant = h5f.create_group("invariant")
            invariant.create_dataset("p", data=p_grid)
            invariant.create_dataset("q", data=q_grid)
            invariant.create_dataset("failure_criterion", data=pq_surface)

            sigma_space = h5f.create_group("sigma_space")
            sigma_space.create_dataset("axis", data=sigma_axis)
            sigma_space.create_dataset("failure_criterion", data=sigma_surface)
            sigma_space.create_dataset("surface_vertices", data=surface_vertices)
            sigma_space.create_dataset("surface_faces", data=surface_faces)

    return save_path
