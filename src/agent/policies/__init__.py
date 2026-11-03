"""Decision policies of the inner agent, one module per version, selected by `policy:` in the config."""
from __future__ import annotations

from importlib import import_module


def get_policy(name: str):
    """The `next_action(records, cfg, compare)` function of policy `name` (e.g. 'v01', 'v02')."""
    return import_module(f"src.agent.policies.{name}").next_action
