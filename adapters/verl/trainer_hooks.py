"""No-op trainer hook skeletons for future verl integration."""

from __future__ import annotations

from typing import Any


class VerlTrainerHooks:
    """Interface placeholder for trainer lifecycle hooks."""

    def on_batch_start(self, batch: Any) -> None:
        """Hook called before a training batch is processed."""

        return None

    def on_batch_end(self, batch: Any, outputs: Any) -> None:
        """Hook called after a training batch is processed."""

        return None

    def on_rollout_end(self, rollout_result: Any) -> None:
        """Hook called after adapter-side rollout collection."""

        return None
