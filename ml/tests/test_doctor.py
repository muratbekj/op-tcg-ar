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
