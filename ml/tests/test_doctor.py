from oplab import doctor


class FakeProbe:
    def __init__(self, **overrides):
        self.values = {"macos_version": "26.0", "xcode_version": "Xcode 27.0", "which": {"uv", "tmux", "rsync"},
                       "modules": {"torch", "coremltools"}, "inbox_exists": True, "inbox_shared": True,
                       "ssh_listening": True, "authorized_keys": True, "index_rows": 4212, **overrides}

    def macos_version(self): return self.values["macos_version"]
    def xcode_version(self): return self.values["xcode_version"]
    def which(self, cmd): return cmd in self.values["which"]
    def has_module(self, name): return name in self.values["modules"]
    def inbox_exists(self): return self.values["inbox_exists"]
    def inbox_shared(self): return self.values["inbox_shared"]
    def ssh_listening(self): return self.values["ssh_listening"]
    def authorized_keys(self): return self.values["authorized_keys"]
    def index_rows(self): return self.values["index_rows"]


def test_all_good():
    results = doctor.checks(FakeProbe())
    assert [c.name for c in results] == ["macOS", "Xcode", "uv", "tmux", "training extras", "inbox folder",
                                         "inbox shared (SMB)", "Remote Login (SSH)", "MacBook key", "full-catalog index"]
    text, code = doctor.render(results)
    assert code == 0 and "✗" not in text
    assert "macOS 26.0" in text and "Xcode 27.0" in text


def test_missing_items_fail_with_hints():
    results = doctor.checks(FakeProbe(xcode_version=None, which={"rsync"}, modules=set(), inbox_exists=False,
                                      inbox_shared=False, ssh_listening=False, authorized_keys=False, index_rows=14))
    text, code = doctor.render(results)
    assert code == 1
    assert "✗ Xcode" in text and "✗ uv" in text and "brew install tmux" in text and "make ml-setup-train" in text
    assert "mkdir ~/oplab-inbox" in text and "File Sharing" in text and "Remote Login" in text
    assert "ssh-copy-id" in text and "generate_embeddings.py" in text


def test_unknown_sharing_state_is_not_a_failure():
    results = doctor.checks(FakeProbe(inbox_shared=None))
    text, code = doctor.render(results)
    assert code == 0 and "? inbox shared (SMB)" in text


def test_unreadable_authorized_keys_is_unknown(monkeypatch, tmp_path):
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "authorized_keys").write_bytes(b"\xff\xfe\x00bad")
    monkeypatch.setattr(doctor.Path, "home", lambda: tmp_path)
    assert doctor.SystemProbe().authorized_keys() is None
    (tmp_path / ".ssh" / "authorized_keys").write_text("ssh-ed25519 AAA me\n")
    assert doctor.SystemProbe().authorized_keys() is True


def test_unknown_keys_render_question_mark():
    text, code = doctor.render(doctor.checks(FakeProbe(authorized_keys=None)))
    assert code == 0 and "? MacBook key: couldn't read ~/.ssh/authorized_keys" in text


def test_malformed_index_meta_counts_zero(monkeypatch, tmp_path):
    meta = tmp_path / "meta.json"
    monkeypatch.setattr(doctor.paths, "INDEX_META", meta)
    for content in ['{"rows": [1, 2', '{"nope": 1}', "[1]", '{"rows": 5}']:
        meta.write_text(content)
        assert doctor.SystemProbe().index_rows() == 0
    meta.write_text('{"rows": [1, 2, 3]}')
    assert doctor.SystemProbe().index_rows() == 3


def _sharing(monkeypatch, stdout=None):
    def run(*args, **kwargs):
        if stdout is None:
            raise doctor.subprocess.CalledProcessError(1, "sharing")
        return doctor.subprocess.CompletedProcess(args, 0, stdout=stdout)
    monkeypatch.setattr(doctor.subprocess, "run", run)


def test_inbox_shared_normalizes_paths(monkeypatch, tmp_path):
    inbox = tmp_path / "oplab-inbox"
    inbox.mkdir()
    monkeypatch.setattr(doctor.paths, "INBOX", inbox)
    _sharing(monkeypatch, f"name:\t\tinbox\npath:\t\t{inbox}/\n")
    assert doctor.SystemProbe().inbox_shared() is True
    _sharing(monkeypatch, "path:\t\t/somewhere/else\n")
    assert doctor.SystemProbe().inbox_shared() is False


def test_inbox_shared_unknown_when_sharing_fails(monkeypatch):
    _sharing(monkeypatch, None)
    assert doctor.SystemProbe().inbox_shared() is None
