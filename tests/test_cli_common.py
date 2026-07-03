import json

from crowdvit.cli.common import _parse_scalar, apply_overrides, resolve_num_classes
from crowdvit.config import Config


def test_parse_scalar_handles_exponent_only_floats():
    assert _parse_scalar("1e-4") == 1e-4
    assert isinstance(_parse_scalar("1e-4"), float)


def test_parse_scalar_handles_ints_bools_lists():
    assert _parse_scalar("10") == 10
    assert _parse_scalar("true") is True
    assert _parse_scalar("[8,4,2]") == [8, 4, 2]


def test_parse_scalar_passes_through_plain_strings():
    assert _parse_scalar("kinetics400") == "kinetics400"


def test_apply_overrides_sets_nested_fields():
    cfg = Config()
    apply_overrides(cfg, ["--optim.lr", "5e-5", "--model.num_classes", "10"])
    assert cfg.optim.lr == 5e-5
    assert cfg.model.num_classes == 10


def test_resolve_num_classes_is_noop_without_class_map(tmp_path):
    cfg = Config()
    cfg.data.class_map = str(tmp_path / "does_not_exist.json")
    cfg.model.num_classes = 400
    resolve_num_classes(cfg)
    assert cfg.model.num_classes == 400


def test_resolve_num_classes_syncs_to_actual_class_map(tmp_path):
    class_map_path = tmp_path / "classes.json"
    with open(class_map_path, "w") as f:
        json.dump({"fall": 0, "overcrowding": 1, "normal": 2}, f)

    cfg = Config()
    cfg.data.class_map = str(class_map_path)
    cfg.model.num_classes = 400  # stale default that should be overridden

    resolve_num_classes(cfg)
    assert cfg.model.num_classes == 3
