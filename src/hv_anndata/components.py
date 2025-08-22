"""Collection of Panel components."""

from __future__ import annotations

import panel as pn
import panel_material_ui as pmui
import param
from panel.custom import PyComponent
from panel.widgets.base import WidgetBase


class GeneGroupSelector(WidgetBase, PyComponent):
    """A custom Panel widget for managing groups of marker genes.

    This component allows users to create, update, and manage groups of
    marker genes through a composite interactive widget.

    - Add new groups (keys) and associate them with marker genes (values).
    - Select and modify marker genes for a specific group using a MultiChoice
      widget.
    - View and edit the entire group-to-marker mapping in a JSON editor widget.
    """

    value: dict[str, list[str]] = param.Dict(  # type: ignore[assignment]
        default={}, doc="Dictionary mapping groups to lists of marker genes."
    )

    options: list[str] = param.List(  # type: ignore[assignment]
        default=[], doc="List of available marker genes for the MultiChoice."
    )

    width = param.Integer(  # type: ignore[assignment]
        default=300, allow_None=True, doc="Width of this component."
    )

    _input_key: str = param.Selector(  # type: ignore[assignment]
        default="",
        objects=[],
        check_on_set=False,
        doc="Current value of the text input (group)",
    )

    _input_value: str = param.String(  # type: ignore[assignment]
        default="", doc="Current value of the text input (marker gene)"
    )

    _current_selection: list[str] = param.List(  # type: ignore[assignment]
        default=[], doc="Current selection of marker genes for the active group"
    )

    def __init__(self, **params: object) -> None:
        """Initialize the component with the given parameters."""
        super().__init__(**params)
        self._current_key = ""
        if self.value:
            self.param._input_key.objects = list(self.value)  # noqa: SLF001

        self.w_key_input = pmui.AutocompleteInput.from_param(
            self.param._input_key,  # noqa: SLF001
            name="Active group",
            placeholder="Enter/select group name",
            restrict=False,
            min_characters=0,
            description="",
            sizing_mode="stretch_width",
        )

        self.w_value_input = pmui.TextInput.from_param(
            self.param._input_value,  # noqa: SLF001
            name="Add new marker gene to group",
            disabled=self.param._input_key.rx().rx.bool().rx.not_(),  # noqa: SLF001
            description="",
            sizing_mode="stretch_width",
        )

        self.w_multi_choice = pmui.MultiChoice.from_param(
            self.param._current_selection,  # noqa: SLF001
            options=self.param.options,
            name="Marker genes for the selected group",
            searchable=True,
            disabled=self.w_value_input.param.disabled,
            description="",
            sizing_mode="stretch_width",
        )

        self.w_json_editor = pn.widgets.JSONEditor.from_param(
            self.param.value,
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

        if key not in self.w_key_input.options:
            self.w_key_input.options = [*self.w_key_input.options, key]

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

        self.w_value_input.value = ""

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
            self.w_key_input,
            self.w_value_input,
            self.w_multi_choice,
            pmui.Typography(
                "JSON Editor:", variant="caption", color="primary", margin=(0, 10)
            ),
            self.w_json_editor,
        )
