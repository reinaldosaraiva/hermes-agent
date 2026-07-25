"""neuralwatt prefix stripping — the provider is config-defined (v12
``providers`` dict, OpenAI-compatible endpoint), so ``-m neuralwatt/<model>``
must reach the API as bare ``<model>`` exactly like zai/kimi-coding."""

from hermes_cli.model_normalize import normalize_model_for_provider


def test_matching_prefix_is_stripped():
    assert (
        normalize_model_for_provider("neuralwatt/qwen3.6-35b-fast", "neuralwatt")
        == "qwen3.6-35b-fast"
    )


def test_bare_model_passes_through():
    assert (
        normalize_model_for_provider("qwen3.6-35b-fast", "neuralwatt")
        == "qwen3.6-35b-fast"
    )


def test_foreign_prefix_is_not_mangled():
    """A non-matching vendor prefix is preserved (no accidental rewrite)."""
    assert (
        normalize_model_for_provider("other/qwen3.6-35b-fast", "neuralwatt")
        == "other/qwen3.6-35b-fast"
    )
