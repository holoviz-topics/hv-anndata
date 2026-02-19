"""Dimension references for anndata."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from anndata.acc import AdAcc, AdRef, GraphAcc, LayerAcc, MetaAcc, MultiAcc
from holoviews.core.dimension import Dimension

if TYPE_CHECKING:
    from types import NotImplementedType
    from typing import Literal, Self

    from anndata.acc import RefAcc


__all__ = ["A", "AdDim"]


def mk_label[I](p: AdRef[I], /) -> str | None:
    match p.acc:
        case MultiAcc():
            return f"{p.acc.k} {p.idx}"
        case GraphAcc():
            return next((f"{p.acc.k} {i}" for i in p.idx if isinstance(i, str)), None)
        case LayerAcc():
            return next(
                (
                    f"{p.acc.k} {i}" if p.acc.k else i
                    for i in p.idx
                    if isinstance(i, str)
                ),
                None,
            )
        case MetaAcc():
            return f"{p.acc.dim} index" if p.idx is None else p.idx
        case _:  # pragma: no cover
            msg = f"Unsupported vector accessor {p.acc!r}"
            raise AssertionError(msg)


class AdDim[I](AdRef[I], Dimension):
    """An AnnData reference that can be used as a dimension."""

    def __init__(self, acc: RefAcc[Self, I], idx: I, /, **params: object) -> None:
        AdRef.__init__(self, acc, idx)
        spec = params.pop("spec", repr(self))
        if "label" not in params and (label := mk_label(self)):
            params["label"] = label
        Dimension.__init__(self, spec, **params)

    @overload
    @classmethod
    def from_dimension(
        cls, dim: Dimension, *, strict: Literal[True] = True
    ) -> Self: ...
    @overload
    @classmethod
    def from_dimension(
        cls, dim: Dimension, *, strict: Literal[False]
    ) -> Self | None: ...

    @classmethod
    def from_dimension(cls, dim: Dimension, *, strict: bool = True) -> Self | None:
        """Create accessor from another dimension."""
        if TYPE_CHECKING:
            assert isinstance(dim.name, str)

        if isinstance(dim, cls):
            return dim
        if (rv := A.resolve(dim.name, strict=strict)) is None:
            return None
        if dim.name != dim.label:
            rv.label = dim.label
        return rv

    def clone(
        self,
        spec: str | tuple[str, str] | None = None,
        **overrides: object,
    ) -> Self:
        """Clones the Dimension with new parameters.

        Derive a new Dimension that inherits existing parameters
        except for the supplied, explicit overrides

        Parameters
        ----------
        spec
            Dimension tuple specification
        overrides
            Dimension parameter overrides

        Returns
        -------
        Cloned Dimension object

        """
        settings = dict(self.param.values(), **overrides)
        acc = settings.pop("acc", self.acc)
        idx = settings.pop("idx", self.idx)

        match spec, ("label" in overrides):
            case None | str(), _:
                spec = (spec or self.name, overrides.get("label", self.label))
            case (name, label), True:
                if overrides["label"] != label:
                    self.param.warning(
                        f"Using label as supplied by keyword ({overrides['label']!r}), "
                        f"ignoring tuple value {label!r}"
                    )
                spec = (name, overrides["label"])

        return type(self)(
            acc,
            idx,
            **{k: v for k, v in settings.items() if k not in {"name", "label"}},
        )

    def __hash__(self) -> int:
        return hash((type(self), repr(self)))

    def __eq__(self, value: object) -> bool | NotImplementedType:
        if isinstance(value, Dimension) and not isinstance(value, AdRef):
            if value.name == self.name:
                return True
            if (value := type(self).from_dimension(value, strict=False)) is None:
                return False
        return super().__eq__(value)

    def __ne__(self, value: object) -> bool:
        return not self == value


A = AdAcc(ref_class=AdDim)
