from graydataviz.discovery import list_dates, list_monkeys, list_sessions

from .conftest import DATE, MONKEY, SESSION


def test_list_monkeys(data_config):
    assert list_monkeys(data_config) == [MONKEY]


def test_list_dates(data_config):
    assert list_dates(MONKEY, data_config) == [DATE]


def test_list_dates_ignores_incomplete_session_dir(data_config):
    incomplete = data_config.raw_root / MONKEY / "999999" / "session01"
    incomplete.mkdir(parents=True)
    (incomplete / "recording_info.mat").touch()  # trial_info.mat missing

    assert list_dates(MONKEY, data_config) == [DATE]


def test_list_sessions(data_config):
    assert list_sessions(MONKEY, DATE, data_config) == [SESSION]


def test_list_monkeys_missing_root(tmp_path):
    from graydataviz.config import DataConfig

    config = DataConfig(raw_root=tmp_path / "does_not_exist")
    assert list_monkeys(config) == []
