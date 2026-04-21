"""
hmt2.py — CLI entry point and backward-compatibility shim.

The implementation lives in the hmt/ package:

    src/hmt/
        __init__.py      public API
        constants.py     ALFRED action prefixes, patterns, Node loader
        types.py         DecompositionState, DecompositionAction
        tree_utils.py    tree serialization and artifact export
        inner_mcts.py    _MCTSChatAdapter, ActionMCTSWrapper
        outer_mcts.py    OuterMCTSPlanner
        integration.py   AlfredReactreeWithHMT
        runner.py        _configure_logging, _main_impl

The @hydra.main decorator lives here (src/ level) so Hydra resolves
config_path="../conf" relative to src/, not src/hmt/.
"""

import logging

try:
    import hydra  # type: ignore[import-not-found]
except ImportError:
    hydra = None


from hmt import (
    AlfredReactreeWithHMT,
    OuterMCTSPlanner,
    ActionMCTSWrapper,
    DecompositionState,
    DecompositionAction,
    export_outer_tree_artifacts,
)
from hmt.runner import _main_impl

__all__ = [
    "AlfredReactreeWithHMT",
    "OuterMCTSPlanner",
    "ActionMCTSWrapper",
    "DecompositionState",
    "DecompositionAction",
    "export_outer_tree_artifacts",
    "main",
]

logger = logging.getLogger(__name__)

if hydra is not None:
    @hydra.main(version_base=None, config_path="../conf", config_name="alfred_reactree")
    def main(cfg) -> None:
        try:
            _main_impl(cfg)
        except Exception:
            logger.exception("HMT run failed")
            raise
else:
    def main(_cfg=None) -> None:
        logging.basicConfig(level=logging.INFO)
        logger.error(
            "Hydra is not installed. Install with `pip install hydra-core` to run hmt2."
        )

if __name__ == "__main__":
    main()
