import pytest

from oplab import remote


def test_read_env_parses_values_comments_and_quotes(tmp_path):
    env = tmp_path / "remote.env"
    env.write_text('# the mac mini\nMINI_HOST = murat@mac-mini.local\n\nMINI_REPO="~/github/op-tcg-ar"\nNOISE\n')
    assert remote.read_env(env) == {"MINI_HOST": "murat@mac-mini.local", "MINI_REPO": "~/github/op-tcg-ar"}
    assert remote.read_env(tmp_path / "missing.env") == {}


def test_mini_requires_host_and_repo(tmp_path):
    env = tmp_path / "remote.env"
    with pytest.raises(remote.RemoteConfigError, match=r"set MINI_HOST and MINI_REPO in .*remote.env"):
        remote.mini(env)
    env.write_text("MINI_HOST=mac-mini.local\n")
    with pytest.raises(remote.RemoteConfigError):
        remote.mini(env)
    env.write_text("MINI_HOST=mac-mini.local\nMINI_REPO=~/github/op-tcg-ar\n")
    assert remote.mini(env) == ("mac-mini.local", "~/github/op-tcg-ar")
