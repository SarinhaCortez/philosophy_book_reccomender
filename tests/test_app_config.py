import os

from myrecsys.app import load_dotenv


def test_load_dotenv_keeps_existing_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("PORT=9000\nLOCAL_ONLY=yes\n", encoding="utf-8")
    monkeypatch.setenv("PORT", "8000")
    monkeypatch.delenv("LOCAL_ONLY", raising=False)

    load_dotenv(env_file)

    assert os.environ["PORT"] == "8000"
    assert os.environ["LOCAL_ONLY"] == "yes"
