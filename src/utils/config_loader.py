from __future__ import annotations
import os
from pathlib import Path
from types import SimpleNamespace
import yaml

class Config(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, value):
        setattr(self, key, value)
    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.__dict__.items()}

def _dict_to_config(d):
    cfg = Config()
    for k, v in d.items():
        setattr(cfg, k, _dict_to_config(v) if isinstance(v, dict) else v)
    return cfg

def _apply_path_resolution(cfg):
    if hasattr(cfg, 'paths'):
        for k, v in cfg.paths.__dict__.items():
            if isinstance(v, str):
                setattr(cfg.paths, k, Path(v).expanduser())

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / 'configs' / 'config.yaml'

def load_config(config_path: str | Path | None = None) -> Config:
    if config_path is None:
        config_path = os.environ.get('RADIOLOGY_CONFIG_PATH', _DEFAULT_CONFIG_PATH)
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'Config not found: {config_path}')
    with config_path.open('r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError('config.yaml must contain a mapping')
    cfg = _dict_to_config(raw)
    _apply_path_resolution(cfg)
    return cfg

if __name__ == '__main__':
    cfg = load_config()
    print('✅ Config loaded successfully')
    print('Patch-MIL encoder:', cfg.patch_mil.encoder_type)
    print('max_patches:', cfg.patch_mil.max_patches)
