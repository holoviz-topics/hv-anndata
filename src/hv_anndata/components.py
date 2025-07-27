"""Collection of Panel components."""

from __future__ import annotations

import panel as pn
import panel_material_ui as pmui
import param
from panel.custom import PyComponent
from panel.widgets.base import WidgetBase


class AutoCompleteMultiChoice(WidgetBase, PyComponent):
    """A composite component combining a text input with a MultiChoice widget.

    The text input serves as a key for a dictionary where each key maps to a
    list of selected values.
    """

    value: dict[str, list[str]] = param.Dict(  # type: ignore[assignment]
        default={}, doc="Dictionary mapping keys to lists of selected values"
    )

    options: list[str] = param.List(  # type: ignore[assignment]
        default=[], doc="List of available options for the MultiChoice"
    )

    width = param.Integer(  # type: ignore[assignment]
        default=300, allow_None=True, doc="Width of this component."
    )

    _input_key: str = param.String(  # type: ignore[assignment]
        default="", doc="Current value of the text input (key)"
    )

    _input_value: str = param.String(  # type: ignore[assignment]
        default="", doc="Current value of the text input (value)"
    )

    _current_selection: list[str] = param.List(  # type: ignore[assignment]
        default=[], doc="Current selection for the active key"
    )

    def __init__(self, **params: object) -> None:
        """Initialize the component with the given parameters."""
        super().__init__(**params)
        self._current_key = ""

        self._key_input = pmui.AutocompleteInput.from_param(
            self.param._input_key,
            name="Group Key",
            placeholder="Enter/select key name",
            restrict=False,
            min_characters=0,
            description="",
            sizing_mode="stretch_width",
        )

        self._value_input = pmui.AutocompleteInput.from_param(
            self.param._input_value,
            options=self.param.options,
            placeholder="Enter/select value",
            name="Available values",
            restrict=False,
            min_characters=0,
            disabled=self.param._input_key.rx().rx.bool().rx.not_(),
            description="",
            sizing_mode="stretch_width",
        )

        self._multi_choice = pmui.MultiChoice.from_param(
            self.param._current_selection,
            options=self.param.options,
            name="Values for the selected group",
            searchable=True,
            disabled=self._value_input.param.disabled,
            description="",
            sizing_mode="stretch_width",
        )

        self._json_editor = pn.widgets.JSONEditor.from_param(
            self.param.value,
            name="JSON Editor",
            mode="tree",
            menu=False,
            sizing_mode="stretch_width",
        )

    @param.depends("_input_key", watch=True)
    def _handle_key_input(self) -> None:
        """Handle when a key is entered in the text input."""
        key = self._input_key.strip()
        if not key:
            return
        # Set the current key
        self._current_key = key

        # Initialize the key in the value dict if it doesn't exist
        if key not in self.value:
            new_value = dict(self.value)
            new_value[key] = []
            self.value = new_value

        # Update current selection to match the key's current values
        self._current_selection = list(self.value.get(key, []))

        if key not in self._key_input.options:
            self._key_input.options = [*self._key_input.options, key]

    @param.depends("_input_value", watch=True)
    def _handle_value_input(self) -> None:
        """Handle when a value is entered in the text input."""
        value = self._input_value.strip()
        if not value or not self._current_key:
            return

        if value not in self.options:
            self.options = [*self.options, value]

        # Add the value to the current selection for the active key
        if value not in self._current_selection:
            self._current_selection = [*self._current_selection, value]

        self._value_input.value = ""

    @param.depends("_current_selection", watch=True)
    def _handle_selection_change(self) -> None:
        """Handle when the MultiChoice selection changes."""
        if not self._current_key:
            return

        # Update the value dict with the new selection
        new_value = dict(self.value)
        new_value[self._current_key] = list(self._current_selection)
        self.value = new_value

    def __panel__(self) -> pn.layout.Column:
        return pn.Column(
            self._key_input,
            self._value_input,
            self._multi_choice,
            self._json_editor,
        )
