"""Shared OS-backed ownership of produced scratch and vault bytes."""

from dinkster_library.produced_scratch import (
    ProducedScratchLease as ProducedScratchLease,
)
from dinkster_library.produced_scratch import (
    ProducedScratchState as ProducedScratchState,
)
from dinkster_library.produced_scratch import (
    claim_inactive_produced_job as claim_inactive_produced_job,
)
from dinkster_library.produced_scratch import (
    claim_inactive_produced_scratch as claim_inactive_produced_scratch,
)
from dinkster_library.produced_scratch import (
    claim_inactive_produced_vault as claim_inactive_produced_vault,
)
from dinkster_library.produced_scratch import (
    hold_produced_asset_read as hold_produced_asset_read,
)
from dinkster_library.produced_scratch import (
    hold_produced_job as hold_produced_job,
)
from dinkster_library.produced_scratch import (
    hold_produced_vault as hold_produced_vault,
)
from dinkster_library.produced_scratch import (
    is_link_or_junction as is_link_or_junction,
)
from dinkster_library.produced_scratch import (
    produced_job_scratch_root as produced_job_scratch_root,
)
from dinkster_library.produced_scratch import (
    retain_produced_asset_path as retain_produced_asset_path,
)
from dinkster_library.produced_scratch import (
    use_produced_asset_path_owner as use_produced_asset_path_owner,
)
