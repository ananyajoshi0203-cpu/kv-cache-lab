"""Tests for the method registry + scenario map."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvlab import registry, scenarios   # noqa: E402


def test_all_five_families_present():
    fams = {m.family for m in registry.REGISTRY}
    for f in registry.FAMILIES:
        assert f in fams


def test_every_method_has_valid_lever():
    valid = set(registry.LEVER_EXPLAINED)
    for m in registry.REGISTRY:
        assert m.lever in valid, f"{m.key} has bad lever {m.lever}"


def test_keys_are_unique():
    keys = [m.key for m in registry.REGISTRY]
    assert len(keys) == len(set(keys))


def test_at_least_three_runnable():
    # h2o, snapkv, kivi are implemented survey methods (the 'full' baseline is separate)
    assert len(registry.by_status("implemented")) >= 3


def test_scenarios_reference_real_methods():
    known = {m.key for m in registry.REGISTRY}
    for s in scenarios.SCENARIOS:
        for k in s.prefer + s.avoid:
            assert k in known, f"scenario '{s.key}' references unknown method '{k}'"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all registry tests passed")
