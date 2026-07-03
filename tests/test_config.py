from pathlib import Path

from crowdvit.config import Config, set_by_path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_base_config_loads():
    cfg = Config.from_yaml(CONFIG_DIR / "base.yaml")
    assert cfg.model.hidden_dim == 768
    assert cfg.model.view_window_sizes == [8, 4, 2]
    assert cfg.model.num_views == 3
    assert cfg.optim.batch_size == 16
    assert cfg.optim.epochs == 50


def test_ablation_config_inherits_base():
    cfg = Config.from_yaml(CONFIG_DIR / "ablation_affinity_uniform.yaml")
    assert cfg.model.affinity_type == "uniform"
    # everything else should still come from base.yaml
    assert cfg.model.hidden_dim == 768
    assert cfg.model.fusion_type == "kga"


def test_set_by_path():
    cfg = Config.from_yaml(CONFIG_DIR / "base.yaml")
    set_by_path(cfg, "optim.lr", 5e-5)
    set_by_path(cfg, "model.view_window_sizes", [8, 2])
    assert cfg.optim.lr == 5e-5
    assert cfg.model.view_window_sizes == [8, 2]


def test_all_configs_load_without_error():
    for path in CONFIG_DIR.glob("*.yaml"):
        Config.from_yaml(path)
    for path in (CONFIG_DIR / "datasets").glob("*.yaml"):
        Config.from_yaml(path)
