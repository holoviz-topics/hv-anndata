"""Tests for the components."""

from __future__ import annotations

import pytest

from hv_anndata.components import GeneGroupSelector


def test_autocomplete_multichoice_init() -> None:
    GeneGroupSelector()


@pytest.mark.parametrize("value", [{"a": ["1", "2"]}, ["1", "2"]])
def test_autocomplete_multichoice_init_value(value: list | dict) -> None:
    GeneGroupSelector(value=value)


def test_autocomplete_multichoice_init_options() -> None:
    GeneGroupSelector(options=["1", "2"])


def test_autocomplete_multichoice_dict_new_groups() -> None:
    w = GeneGroupSelector()

    w.w_key_input.value = "a"
    assert w.value == {"a": []}
    assert w.w_key_input.options == ["", "a"]

    w.w_key_input.value = "b"
    assert w.value == {"a": [], "b": []}
    assert w.w_key_input.options == ["", "a", "b"]


def test_autocomplete_multichoice_dict_new_values() -> None:
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


def test_autocomplete_multichoice_list_new_values() -> None:
    w = GeneGroupSelector(value=[])

    w.w_value_input.value = "1"

    assert w.options == ["1"]
    assert w.w_multi_choice.value == ["1"]
    assert w.w_value_input.value == ""
    assert w.value == ["1"]

    w.w_value_input.value = "2"

    assert w.options == ["1", "2"]
    assert w.w_multi_choice.value == ["1", "2"]
    assert w.w_value_input.value == ""
    assert w.value == ["1", "2"]


def test_autocomplete_multichoice_dict_update_selected() -> None:
    w = GeneGroupSelector(value={"a": ["1", "2"]})

    w.w_key_input.value = "a"
    assert w.w_multi_choice.value == ["1", "2"]
    assert w.value == {"a": ["1", "2"]}

    w.w_multi_choice.value = ["1"]

    assert w.value == {"a": ["1"]}


def test_autocomplete_multichoice_list_update_selected() -> None:
    w = GeneGroupSelector(value=["1", "2"])

    w.w_multi_choice.value = ["1"]

    assert w.value == ["1"]


def test_autocomplete_multichoice_dict_value_init_key_options() -> None:
    w = GeneGroupSelector(value={"a": ["1", "2"]})

    assert w.param._input_key.objects == ["a"]
    assert w.w_key_input.options == {"a": "a"}


def test_autocomplete_multichoice_list_value_init_mc() -> None:
    w = GeneGroupSelector(value=["1", "2"])

    assert w._current_selection == ["1", "2"]
    assert w.w_multi_choice.value == ["1", "2"]
    assert w.w_multi_choice.options == ["1", "2"]
