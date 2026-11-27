#pragma once
#include "viscoelastic_kernels.h"
#include "../shared/phasefield_splits.h"

namespace viscoelastic {

enum class CreepCoupling { EffectiveStress, DegradedStress, FullDamagedStress, PartialExplicit };

struct SplitIPResult {
    IPResult state;
    shared_kernels::ElasticSplitResult split;
};

SplitIPResult update_split_ip(const MaterialParams& mp,
                             shared_kernels::ElasticSplit split,
                             CreepCoupling creep_coupling, const Vec6& strain,
                             const Vec6& ep_old, double dt, double damage,
                             const Vec6& old_total_strain);

} // namespace viscoelastic
