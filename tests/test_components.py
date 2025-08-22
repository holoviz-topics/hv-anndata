"""Tests for the components."""

from __future__ import annotations

from hv_anndata.components import GeneGroupSelector


def test_autocomplete_multichoice_init() -> None:
    GeneGroupSelector()


def test_autocomplete_multichoice_init_value() -> None:
    GeneGroupSelector(value={"a": ["1", "2"]})


def test_autocomplete_multichoice_init_options() -> None:
    GeneGroupSelector(options=["1", "2"])


def test_autocomplete_multichoice_new_groups() -> None:
    w = GeneGroupSelector()

    w.w_key_input.value = "a"
    assert w.value == {"a": []}
    assert w.w_key_input.options == ["", "a"]

    w.w_key_input.value = "b"
    assert w.value == {"a": [], "b": []}
    assert w.w_key_input.options == ["", "a", "b"]


def test_autocomplete_multichoice_new_values() -> None:
    w = GeneGroupSelector()

    w.w_key_input.value = "a"

    w.w_value_input.value = "1"

    assert w.options == ["1"]
    assert w.w_multi_choice.value == ["1"]
    assert w.w_value_input.value == ""
    assert w.value == {"a": ["1"]}

    w.w_value_input.value = "2"

    assert w.options == ["1", "2"]
    assert w.w_multi_choice.value == ["1", "2"]
    assert w.w_value_input.value == ""
    assert w.value == {"a": ["1", "2"]}


def test_autocomplete_multichoice_update_selected() -> None:
    w = GeneGroupSelector(value={"a": ["1", "2"]})

    w.w_key_input.value = "a"
    assert w.w_multi_choice.value == ["1", "2"]
    assert w.value == {"a": ["1", "2"]}

    w.w_multi_choice.value = ["1"]

    assert w.value == {"a": ["1"]}


def test_autocomplete_multichoice_value_init_key_options() -> None:
    w = GeneGroupSelector(value={"a": ["1", "2"]})

    assert w.param._input_key.objects == ["a"]
    assert list(w.w_key_input.options) == ["a"]
