from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, cast, overload

import holoviews as hv
import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


class Raise(Enum):
    Sentry = auto()


@dataclass
class AnnDataProxy:
    adata: AnnData

    @overload
    def get(self, k: str, /, default: None = None) -> np.ndarray | None: ...
    @overload
    def get(self, k: str, /, default: np.ndarray | Raise) -> np.ndarray: ...
    def get(
        self, k: str, /, default: np.ndarray | Raise | None = None
    ) -> np.ndarray | None:
        k_orig = k
        if "." not in k:
            if default is not Raise.Sentry and k not in self.adata.var_names:
                return default
            return self.adata[:, k].X.flatten()
        attr_name, k = k.split(".", 1)
        attr = getattr(self.adata, attr_name)
        if "." not in k:
            if default is not Raise.Sentry and k not in attr:
                return default
            return attr[k]
        k, i = k.split(".", 1)
        arr = attr[k]
        if "." not in i:
            if default is not Raise.Sentry and not (0 <= int(i) < arr.shape[1]):
                return default
            return arr[:, int(i)]
        raise KeyError(k_orig)

    def __contains__(self, k: str) -> bool:
        return self.get(k) is not None

    def __getitem__(self, k: str) -> object:
        return self.get(k, Raise.Sentry)

    def __len__(self) -> int:
        return len(self.adata)


class AnnDataInterface(hv.core.Interface):
    types = (AnnData,)
    datatype = "anndata"

    @classmethod
    def init(
        cls,
        eltype,
        data: AnnData | AnnDataProxy,
        kdims: list[str] | None,
        vdims: list[str] | None,
    ) -> tuple[AnnDataProxy, dict]:
        proxy = AnnDataProxy(data) if isinstance(data, AnnData) else data
        return proxy, {"kdims": kdims, "vdims": vdims}, {}

    @classmethod
    def values(
        cls,
        data: hv.Dataset,
        dim: hv.Dimension | str,
        expanded=True,
        flat=True,
        compute=True,
        keep_index=False,
    ) -> np.ndarray:
        dim = data.get_dimension(dim)
        proxy = cast(AnnDataProxy, data.data)
        return proxy[dim.name]

    @classmethod
    def dimension_type(cls, data: hv.Dataset, dim: hv.Dimension | str) -> np.dtype:
        dim = data.get_dimension(dim)
        proxy = cast(AnnDataProxy, data.data)
        return proxy[dim.name].dtype


if AnnDataInterface.datatype not in hv.core.data.datatypes:
    hv.core.data.datatypes.append(AnnDataInterface.datatype)
hv.core.Interface.register(AnnDataInterface)
