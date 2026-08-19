"""The two guards standing between a small model and the filesystem.

Neither is defensive programming for its own sake. Both were written after
something went wrong.

Argument coercion exists because Llama sends `"5"` where the schema says
integer, and nothing checked types, so the string reached the function and
crashed it on a comparison. Three skills failed that way in one run. Qwen
happens to send the right types, which is not a property to depend on.

The sandbox exists because the model is small enough to occasionally produce
`../../..` in a filename, and because granting an assistant a folder is not
granting it the credentials that happen to sit in that folder — this one reads
its answers out loud in a room.
"""

from __future__ import annotations

import pytest

from talos.skills import Registry
from talos.skills.base import Permission, skill


@skill(Permission.OPEN)
def sample_skill(name: str, limit: int = 5, ratio: float = 1.0, loud: bool = False) -> str:
    """A skill that exists only to be called by these tests.

    Args:
        name: Anything at all
        limit: How many
        ratio: A fraction
        loud: Whether to shout
    """
    return f"{name}|{limit}|{ratio}|{loud}"


@pytest.fixture
def registry() -> Registry:
    made = Registry()
    made._skills["sample_skill"] = sample_skill._skill
    return made


class TestArgumentCoercion:
    def test_a_number_sent_as_a_string_is_converted(self, registry):
        # The exact failure: "5" reaching a function that compares it to an int.
        assert registry.execute("sample_skill", {"name": "x", "limit": "5"}) == "x|5|1.0|False"

    def test_a_float_sent_as_a_string_is_converted(self, registry):
        assert "0.5" in registry.execute("sample_skill", {"name": "x", "ratio": "0.5"})

    @pytest.mark.parametrize("sent,expected", [("true", "True"), ("no", "False"), ("1", "True")])
    def test_booleans_arrive_in_several_spellings(self, registry, sent, expected):
        assert expected in registry.execute("sample_skill", {"name": "x", "loud": sent})

    def test_something_genuinely_unconvertible_is_refused_readably(self, registry):
        # Phrased for the model to correct on its next attempt, not as a
        # traceback, which teaches it nothing.
        result = registry.execute("sample_skill", {"name": "x", "limit": "a few"})
        assert "integer" in result and "a few" in result

    def test_a_missing_required_argument_says_which(self, registry):
        assert "name" in registry.execute("sample_skill", {"limit": 3})

    def test_an_unexpected_argument_is_refused(self, registry):
        assert "colour" in registry.execute("sample_skill", {"name": "x", "colour": "red"})

    def test_an_unknown_skill_lists_the_real_ones(self, registry):
        result = registry.execute("delete_everything", {})
        assert "sample_skill" in result


class TestSandbox:
    @pytest.fixture(autouse=True)
    def workspace(self, tmp_path, monkeypatch):
        """Point the sandbox at a real temporary directory.

        Not mocked: the whole value of these tests is that resolution, symlink
        following and containment behave the same as they will in use.
        """
        from talos.skills import _workspace

        (tmp_path / "project").mkdir()
        (tmp_path / "project" / "README.md").write_text("hello", encoding="utf-8")
        (tmp_path / "project" / "gizli.json").write_text("{}", encoding="utf-8")
        (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")

        monkeypatch.setattr(_workspace, "WORKSPACES", (tmp_path / "project",))
        self.root = tmp_path
        return tmp_path

    def test_a_file_inside_resolves(self):
        from talos.skills._workspace import resolve

        assert resolve("README.md").name == "README.md"

    @pytest.mark.parametrize(
        "path",
        ["../outside.txt", "../../outside.txt", "..\\outside.txt"],
    )
    def test_climbing_out_is_refused(self, path):
        from talos.skills._workspace import OutsideWorkspace, resolve

        with pytest.raises(OutsideWorkspace):
            resolve(path)

    def test_an_absolute_path_from_elsewhere_is_refused(self):
        from talos.skills._workspace import OutsideWorkspace, resolve

        with pytest.raises(OutsideWorkspace):
            resolve(str(self.root / "outside.txt"))

    def test_a_credential_file_is_refused_even_though_it_is_inside(self):
        from talos.skills._workspace import SecretFile, resolve

        with pytest.raises(SecretFile):
            resolve("gizli.json")

    @pytest.mark.parametrize(
        "name",
        [".env", "id_rsa", "server.key", "secrets.yaml", "my_token.txt"],
    )
    def test_the_shapes_credentials_come_in(self, name):
        from talos.skills._workspace import is_secret
        from pathlib import Path

        assert is_secret(Path(name))

    def test_ordinary_files_are_not_mistaken_for_secrets(self):
        from talos.skills._workspace import is_secret
        from pathlib import Path

        for name in ["README.md", "main.py", "settings.toml", "notes.md"]:
            assert not is_secret(Path(name))
