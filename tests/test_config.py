def test_config_file_exists():
    from pathlib import Path
    assert (Path(__file__).parents[1] / 'configs' / 'config.yaml').exists()
