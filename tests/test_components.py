"""Tests for the components."""

from __future__ import annotations

from hv_anndata.components import AutoCompleteMultiChoice


def test_autocomplete_multichoice_init() -> None:
    AutoCompleteMultiChoice()


def test_autocomplete_multichoice_init_value() -> None:
    AutoCompleteMultiChoice(value={"a": ["1", "2"]})


def test_autocomplete_multichoice_init_options() -> None:
    AutoCompleteMultiChoice(options=["1", "2"])


def test_autocomplete_multichoice_new_groups() -> None:
    w = AutoCompleteMultiChoice()

    w._key_input.value = "a"
    assert w.value == {"a": []}
    assert w._key_input.options == ["a"]

    w._key_input.value = "b"
    assert w.value == {"a": [], "b": []}
    assert w._key_input.options == ["a", "b"]


def test_autocomplete_multichoice_new_values() -> None:
    w = AutoCompleteMultiChoice()

    w._key_input.value = "a"

    w._value_input.value = "1"

    assert w.options == ["1"]
    assert w._multi_choice.value == ["1"]
    assert w._value_input.value == ""
    assert w.value == {"a": ["1"]}

    w._value_input.value = "2"

    assert w.options == ["1", "2"]
    assert w._multi_choice.value == ["1", "2"]
    assert w._value_input.value == ""
    assert w.value == {"a": ["1", "2"]}


def test_autocomplete_multichoice_update_selected() -> None:
    w = AutoCompleteMultiChoice(value={"a": ["1", "2"]})

    w._key_input.value = "a"
    assert w._multi_choice.value == ["1", "2"]
    assert w.value == {"a": ["1", "2"]}

    w._multi_choice.value = ["1"]

    assert w.value == {"a": ["1"]}
