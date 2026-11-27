"""Reusable adaptive mesh-refinement contracts and algorithms.

Topology, marker transfer, field transfer, visualization, and orchestration
are added incrementally behind this package boundary.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "FieldThresholdIndicatorConfig",
    "RefinementConfig",
    "BufferedCellSelection",
    "CellIndicatorResult",
    "MeshMarkerState",
    "PartitionMigrationData",
    "RepartitionResult",
    "RefinementInstallResult",
    "FieldTransferDiagnostic",
    "FieldTransferSnapshot",
    "QuadratureReconstruction",
    "RefinedMeshArtifacts",
    "RefinementReport",
    "TransferredFieldState",
    "expand_refinement_buffer",
    "evaluate_field_indicator",
    "get_quadrature_reconstruction",
    "install_refined_mesh_state",
    "migrate_repartitioned_fields",
    "refine_from_indicator",
    "normalize_refinement_config",
    "refine_mesh_topology",
    "repartition_mesh",
    "redistribute_mesh_markers",
    "resolve_owned_child_input_positions",
    "reconstruct_scalar_quadrature_samples",
    "snapshot_registered_fields",
    "transfer_mesh_markers",
    "transfer_quadrature_fields",
    "transfer_standard_fields",
]

_PUBLIC_EXPORTS = {
    "FieldThresholdIndicatorConfig": (
        "femcubes.core.mesh.refinement.config",
        "FieldThresholdIndicatorConfig",
    ),
    "RefinementConfig": ("femcubes.core.mesh.refinement.config", "RefinementConfig"),
    "normalize_refinement_config": (
        "femcubes.core.mesh.refinement.config",
        "normalize_refinement_config",
    ),
    "CellIndicatorResult": (
        "femcubes.core.mesh.refinement.indicators",
        "CellIndicatorResult",
    ),
    "BufferedCellSelection": (
        "femcubes.core.mesh.refinement.indicators",
        "BufferedCellSelection",
    ),
    "expand_refinement_buffer": (
        "femcubes.core.mesh.refinement.indicators",
        "expand_refinement_buffer",
    ),
    "evaluate_field_indicator": (
        "femcubes.core.mesh.refinement.indicators",
        "evaluate_field_indicator",
    ),
    "QuadratureReconstruction": (
        "femcubes.core.mesh.refinement.quadrature",
        "QuadratureReconstruction",
    ),
    "get_quadrature_reconstruction": (
        "femcubes.core.mesh.refinement.quadrature",
        "get_quadrature_reconstruction",
    ),
    "reconstruct_scalar_quadrature_samples": (
        "femcubes.core.mesh.refinement.quadrature",
        "reconstruct_scalar_quadrature_samples",
    ),
    "RefinedMeshArtifacts": (
        "femcubes.core.mesh.refinement.topology",
        "RefinedMeshArtifacts",
    ),
    "RefinementReport": (
        "femcubes.core.mesh.refinement.topology",
        "RefinementReport",
    ),
    "refine_mesh_topology": (
        "femcubes.core.mesh.refinement.topology",
        "refine_mesh_topology",
    ),
    "repartition_mesh": (
        "femcubes.core.mesh.refinement.partition",
        "repartition_mesh",
    ),
    "PartitionMigrationData": (
        "femcubes.core.mesh.refinement.partition",
        "PartitionMigrationData",
    ),
    "RepartitionResult": (
        "femcubes.core.mesh.refinement.partition",
        "RepartitionResult",
    ),
    "MeshMarkerState": (
        "femcubes.core.mesh.refinement.markers",
        "MeshMarkerState",
    ),
    "RefinementInstallResult": (
        "femcubes.core.mesh.refinement.manager",
        "RefinementInstallResult",
    ),
    "install_refined_mesh_state": (
        "femcubes.core.mesh.refinement.manager",
        "install_refined_mesh_state",
    ),
    "refine_from_indicator": (
        "femcubes.core.mesh.refinement.manager",
        "refine_from_indicator",
    ),
    "resolve_owned_child_input_positions": (
        "femcubes.core.mesh.refinement.markers",
        "resolve_owned_child_input_positions",
    ),
    "transfer_mesh_markers": (
        "femcubes.core.mesh.refinement.markers",
        "transfer_mesh_markers",
    ),
    "redistribute_mesh_markers": (
        "femcubes.core.mesh.refinement.markers",
        "redistribute_mesh_markers",
    ),
    "migrate_repartitioned_fields": (
        "femcubes.core.mesh.refinement.migration",
        "migrate_repartitioned_fields",
    ),
    "FieldTransferDiagnostic": (
        "femcubes.core.mesh.refinement.transfer",
        "FieldTransferDiagnostic",
    ),
    "FieldTransferSnapshot": (
        "femcubes.core.mesh.refinement.transfer",
        "FieldTransferSnapshot",
    ),
    "TransferredFieldState": (
        "femcubes.core.mesh.refinement.transfer",
        "TransferredFieldState",
    ),
    "snapshot_registered_fields": (
        "femcubes.core.mesh.refinement.transfer",
        "snapshot_registered_fields",
    ),
    "transfer_standard_fields": (
        "femcubes.core.mesh.refinement.transfer",
        "transfer_standard_fields",
    ),
    "transfer_quadrature_fields": (
        "femcubes.core.mesh.refinement.transfer",
        "transfer_quadrature_fields",
    ),
}


def __getattr__(name: str):
    """Lazily resolve refinement exports so configuration stays lightweight."""

    try:
        module_name, attr_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    """Return module attributes plus lazy public exports."""

    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from femcubes.core.mesh.refinement.config import (
        FieldThresholdIndicatorConfig,
        RefinementConfig,
        normalize_refinement_config,
    )
    from femcubes.core.mesh.refinement.indicators import (
        BufferedCellSelection,
        CellIndicatorResult,
        expand_refinement_buffer,
        evaluate_field_indicator,
    )
    from femcubes.core.mesh.refinement.partition import (
        PartitionMigrationData,
        RepartitionResult,
        repartition_mesh,
    )
    from femcubes.core.mesh.refinement.quadrature import (
        QuadratureReconstruction,
        get_quadrature_reconstruction,
        reconstruct_scalar_quadrature_samples,
    )
    from femcubes.core.mesh.refinement.markers import (
        MeshMarkerState,
        redistribute_mesh_markers,
        resolve_owned_child_input_positions,
        transfer_mesh_markers,
    )
    from femcubes.core.mesh.refinement.manager import (
        RefinementInstallResult,
        install_refined_mesh_state,
        refine_from_indicator,
    )
    from femcubes.core.mesh.refinement.topology import (
        RefinedMeshArtifacts,
        RefinementReport,
        refine_mesh_topology,
    )
    from femcubes.core.mesh.refinement.transfer import (
        FieldTransferDiagnostic,
        FieldTransferSnapshot,
        TransferredFieldState,
        snapshot_registered_fields,
        transfer_quadrature_fields,
        transfer_standard_fields,
    )
