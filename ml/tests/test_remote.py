import shlex

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


def test_train_command_runs_make_train_in_tmux_over_ssh():
    cmd = remote.train_command("murat@mac-mini.local", "~/github/op-tcg-ar", "v1")
    assert cmd[:2] == ["ssh", "murat@mac-mini.local"]
    shell = shlex.split(cmd[2])
    assert shell[:2] == ["zsh", "-lc"]
    inner = shell[2]
    assert inner.startswith("tmux new-session -d -s train-v1 ")
    job = shlex.split(inner)[-1]
    assert job == "cd ~/github/op-tcg-ar && mkdir -p ml/runs && make train NAME=v1 2>&1 | tee ml/runs/train-v1.log"


def test_train_command_passes_extra_args():
    cmd = remote.train_command("mini", "~/repo", "v2", extra="--epochs 3")
    assert "make train NAME=v2 ARGS='--epochs 3'" in shlex.split(shlex.split(cmd[2])[2])[-1]


@pytest.mark.parametrize("name", ["", "v 1", "v1;rm -rf ~", "$(whoami)", "../v1", "v1'"])
def test_train_command_rejects_unsafe_names(name):
    with pytest.raises(ValueError, match="NAME"):
        remote.train_command("mini", "~/repo", name)
