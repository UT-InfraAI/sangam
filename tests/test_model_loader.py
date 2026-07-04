import pytest

from sangam.model import model_loader


def test_read_default_kv_max_pages_dream(monkeypatch) -> None:
    monkeypatch.setattr(
        model_loader,
        "_read_hf_config",
        lambda _: {"architectures": ["DreamModel"]},
    )

    assert (
        model_loader.read_default_kv_max_pages("Dream-org/Dream-v0-Instruct-7B")
        == 49152
    )


def test_read_default_kv_max_pages_llada(monkeypatch) -> None:
    monkeypatch.setattr(
        model_loader,
        "_read_hf_config",
        lambda _: {"architectures": ["LLaDAModelLM"]},
    )

    assert model_loader.read_default_kv_max_pages("GSAI-ML/LLaDA-8B-Instruct") == 5632


def test_read_default_kv_max_pages_unknown_arch_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        model_loader,
        "_read_hf_config",
        lambda _: {"architectures": ["SomeOtherModel"]},
    )

    with pytest.raises(ValueError, match="No default kv_max_pages for architecture"):
        model_loader.read_default_kv_max_pages("acme/some-model")


def test_read_default_kv_max_pages_missing_architectures_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        model_loader,
        "_read_hf_config",
        lambda _: {},
    )

    with pytest.raises(ValueError, match="has no `architectures`"):
        model_loader.read_default_kv_max_pages("acme/some-model")
