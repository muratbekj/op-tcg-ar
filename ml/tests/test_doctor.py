from oplab import doctor


class FakeProbe:
    def __init__(self, **overrides):
        self.values = {"macos_version": "26.0", "swift_version": "Apple Swift version 6.1.2 (swiftlang-6.1.2.1.2 clang-1700.0.13.5)", "which": {"uv", "tmux", "rsync"},
                       "modules": {"torch", "coremltools"}, "inbox_exists": True, "inbox_shared": True,
                       "ssh_listening": True, "smb_listening": True, "login": {"uv", "tmux"}, "authorized_keys": True, "index_rows": 4212, **overrides}

    def macos_version(self): return self.values["macos_version"]
    def swift_version(self): return self.values["swift_version"]
    def which(self, cmd): return cmd in self.values["which"]
    def has_module(self, name): return name in self.values["modules"]
    def inbox_exists(self): return self.values["inbox_exists"]
    def inbox_shared(self): return self.values["inbox_shared"]
    def smb_listening(self): return self.values["smb_listening"]
    def login_has(self, cmd): return None if self.values["login"] is None else cmd in self.values["login"]
    def ssh_listening(self): return self.values["ssh_listening"]
    def authorized_keys(self): return self.values["authorized_keys"]
    def index_rows(self): return self.values["index_rows"]


def test_all_good():
    results = doctor.checks(FakeProbe())
    assert [c.name for c in results] == ["macOS", "Swift toolchain", "uv", "tmux", "training extras", "inbox folder",
                                         "inbox shared (SMB)", "Remote Login (SSH)", "MacBook key", "full-catalog index"]
    text, code = doctor.render(results)
    assert code == 0 and "✗" not in text
    assert "macOS 26.0" in text and "Apple Swift version 6.1.2" in text


def test_missing_items_fail_with_hints():
    results = doctor.checks(FakeProbe(swift_version=None, which={"rsync"}, login=set(), modules=set(), inbox_exists=False,
                                      inbox_shared=False, ssh_listening=False, authorized_keys=False, index_rows=14))
    text, code = doctor.render(results)
    assert code == 1
    assert "✗ Swift toolchain" in text and "xcode-select --install" in text and "✗ uv" in text and "brew install tmux" in text and "make ml-setup-train" in text
    assert "mkdir ~/oplab-inbox" in text and "File Sharing" in text and "Remote Login" in text
    assert "ssh-copy-id" in text and "generate_embeddings.py" in text


def test_swift_older_than_6_fails():
    results = doctor.checks(FakeProbe(swift_version="Apple Swift version 5.10 (swiftlang-5.10.0.13 clang-1500.3.9.4)"))
    text, code = doctor.render(results)
    assert code == 1 and "✗ Swift toolchain" in text and "Swift 6" in text


def test_swift_major():
    assert doctor.swift_major("Apple Swift version 6.3.3 (swiftlang-6.3.3.1.3 clang-2100.1.1.101)") == 6
    assert doctor.swift_major("swift-driver version: 1.148.6 Apple Swift version 6.3.3 (swiftlang-6.3.3)") == 6
    assert doctor.swift_major("something else") is None


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


def test_share_listed_but_smb_off_fails():
    text, code = doctor.render(doctor.checks(FakeProbe(smb_listening=False)))
    assert code == 1 and "✗ inbox shared (SMB)" in text and "turn on File Sharing" in text


def test_uv_installed_but_not_on_login_path():
    text, code = doctor.render(doctor.checks(FakeProbe(login=set(), which={"uv", "tmux"})))
    assert code == 1 and "✗ uv: installed but not on the login PATH" in text and "~/.zprofile" in text


def test_login_has_uses_login_shell(monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return doctor.subprocess.CompletedProcess(cmd, 0 if cmd[-1].endswith("uv") else 1)
    monkeypatch.setattr(doctor.subprocess, "run", run)
    probe = doctor.SystemProbe()
    assert probe.login_has("uv") is True and probe.login_has("tmux") is False
    assert calls[0][:2] == ["zsh", "-lc"]

    def boom(*a, **k):
        raise OSError
    monkeypatch.setattr(doctor.subprocess, "run", boom)
    assert probe.login_has("uv") is None
