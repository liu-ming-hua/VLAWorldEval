from dataclasses import replace

import pytest

from adapters.actions.base import IdentityActionBridge, build_action_bridge
from adapters.observations.base import IdentityViewBridge, build_view_bridge
from evaluation.manifests import ViewBridgeSpec
from vlaworldeval.demo import demo_manifest


def test_local_plugins_need_no_core_factory_edits():
    spec = replace(demo_manifest().action_bridge, name="adapters.actions.base:IdentityActionBridge")
    assert isinstance(build_action_bridge(spec), IdentityActionBridge)
    view = ViewBridgeSpec(name="adapters.observations.base:IdentityViewBridge")
    assert isinstance(build_view_bridge(view), IdentityViewBridge)


@pytest.mark.parametrize(
    "name",
    [
        "os:system",
        "adapters.actions.base:ActionChunk",
        "adapters.actions...base:IdentityActionBridge",
        "adapters.actions.base:ActionBridge",
        "adapters.actions.base:CallableActionBridge",
    ],
)
def test_bad_plugin_rejected(name):
    spec = replace(demo_manifest().action_bridge, name=name)
    with pytest.raises(ValueError):
        build_action_bridge(spec)
