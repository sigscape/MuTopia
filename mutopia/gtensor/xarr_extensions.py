import sparse
import xarray as xr
import typing
from xarray.core.formatting_html import *
from xarray.core.formatting_html import _get_indexes_dict, _obj_repr
from collections import defaultdict

def _get_section(var_name, raw=False):
    if var_name == "X":
        return "X"

    section_name = var_name.split("/")[0] if "/" in var_name else "data"
    return section_name if raw else section_name.title()


section_priorities = defaultdict(
    lambda: 4,
    {
        "X": 0,
        "Data": 1,
        "Features": 2,
        "Regions": 3,
        "State": 100,
    },
)


def dataset_repr(ds) -> str:

    obj_type = f"{ds.attrs.get('name', 'Data')}"
    header_components = [f"<div class='xr-obj-type'>G-Tensor: {escape(obj_type)}</div>"]

    sections = defaultdict(dict)
    for var_name, var in ds.data_vars.items():
        sections[_get_section(var_name)][var_name] = var

    sections = dict(sorted(sections.items(), key=lambda x: section_priorities[x[0]]))

    sections = [
        dim_section(ds),
        *[
            datavar_section(
                section,
                name=section_name,
                max_items_collapse=(
                    1
                    if section_name
                    in [
                        "Regions",
                        "State",
                    ]
                    else 15
                ),
            )
            for section_name, section in sections.items()
        ],
        coord_section(ds.coords),
        index_section(_get_indexes_dict(ds.xindexes)),
        attr_section(ds.attrs, max_items_collapse=1),
    ]

    return _obj_repr(ds, header_components, sections)


## MONKEY PATCH!
xr.Dataset._repr_html_ = dataset_repr
##


class BaseAccessor:
    def __init__(self, xrds):
        self._xrds = xrds

    @staticmethod
    def _validate(self, req_dim=None, req_vars=None):
        if req_dim is not None:
            if all([not dim in list(self._xrds.dims) for dim in req_dim]):
                raise AttributeError("Required dimensions are missing")
        if req_vars is not None:
            if all([not var in self._xrds.variables for var in req_vars.keys()]):
                raise AttributeError("Required variables are missing")


@xr.register_dataset_accessor("coo_to_sparse")
class OnDiskCoo(BaseAccessor):

    def __call__(
        self,
        data_var="data",
    ):

        self._validate(self, ["obs_indices"], {"indices": None, "data": None})

        indices = self._xrds["indices"].data
        data = self._xrds[data_var].data
        shape = self._xrds.attrs["shape"]
        dims = tuple(list(self._xrds.coords["obs_indices"].values))

        sparsified_data = sparse.COO(
            indices,
            data,
            shape=shape,
            fill_value=0.0,
        )

        return xr.DataArray(
            sparsified_data,
            dims=dims,
            attrs={
                k: v
                for k, v in self._xrds.attrs.items()
                if not k in ("shape", "format")
            },
        )


@xr.register_dataarray_accessor("sparse_to_coo")
class OnDiskSparse(BaseAccessor):

    def __call__(
        self,
    ):

        if not isinstance(self._xrds.data, sparse.SparseArray):
            raise TypeError("Data is not in a sparse format")

        if not isinstance(self._xrds.data, sparse.COO):
            sparse_matrix = self._xrds.data.tocoo()
        else:
            sparse_matrix = self._xrds.data

        return xr.Dataset(
            {
                "indices": xr.DataArray(
                    sparse_matrix.coords,
                    dims=["obs_indices", "n_observations"],
                ),
                "data": xr.DataArray(
                    sparse_matrix.data,
                    dims=("n_observations"),
                ),
            },
            coords={
                "obs_indices": list(self._xrds.dims),
            },
            attrs={
                "shape": sparse_matrix.shape,
                "format": "COO",
                **self._xrds.attrs,
            },
        )


@xr.register_dataarray_accessor("ascoo")
class AsCOO(BaseAccessor):
    def __call__(self):
        if isinstance(self._xrds.data, sparse.COO):
            return self._xrds
        elif isinstance(self._xrds.data, sparse.GCXS):
            self._xrds.data = self._xrds.data.tocoo()
        else:
            raise TypeError("Data is not in a sparse format")

        return self._xrds


@xr.register_dataarray_accessor("ascsr")
class AsCSR(BaseAccessor):
    def __call__(self, *compress_dims: typing.List[str]):
        if not len(compress_dims) > 0:
            raise ValueError("Provide at least one dimension must be compressed")

        dims = self._xrds.dims

        self._xrds.data = sparse.GCXS(
            self._xrds.data,
            compressed_axes=tuple([dims.index(dim) for dim in compress_dims]),
        )

        return self._xrds


@xr.register_dataarray_accessor("asdense")
class AsCSR(BaseAccessor):
    def __call__(self):
        try:
            self._xrds.data = self._xrds.data.todense()
        except AttributeError:
            pass
        return self._xrds


@xr.register_dataarray_accessor("is_sparse")
class IsSparse(BaseAccessor):
    def __call__(self):
        return isinstance(self._xrds.data, sparse.SparseArray)


@xr.register_dataset_accessor("list_samples")
class FetchSample(BaseAccessor):
    def __call__(self):
        return self._xrds.sample.values


@xr.register_dataset_accessor("mutate")
class FetchSample(BaseAccessor):
    def __call__(self, fn):
        return fn(self._xrds)


@xr.register_dataset_accessor("fetch_sample")
class FetchSample(BaseAccessor):
    def __call__(self, sample_name):
        sample_vars = [v for v, k in self._xrds.data_vars.items() if "sample" in k.dims]
        return self._xrds[sample_vars].sel(sample=sample_name)


@xr.register_dataset_accessor("iter_samples")
class FetchSample(BaseAccessor):
    def __call__(self):
        for sample_name in self._xrds.list_samples():
            yield self._xrds.fetch_sample(sample_name)


@xr.register_dataset_accessor("sections")
class Section(BaseAccessor):

    def __getitem__(self, section: str):
        section = section.title()
        subset_datavars = [
            k for k in self._xrds.data_vars.keys() if _get_section(k) == section
        ]

        if len(subset_datavars) == 0:
            raise ValueError(f"Section {section} not found in dataset")

        return self._xrds[subset_datavars].pipe(
            lambda x: x.rename(
                {
                    k: k.removeprefix(_get_section(k, raw=True) + "/")
                    for k in x.data_vars.keys()
                }
            )
        )

    @property
    def groups(self):
        sections = defaultdict(list)
        for k in self._xrds.data_vars.keys():
            sections[_get_section(k)].append(k)
        return dict(sections)

    @property
    def names(self):
        return list(self.groups.keys())

    def __iter__(self):
        for section in self.groups.keys():
            yield section, self[section]


@xr.register_dataarray_accessor("sum_except")
class SumExcept(BaseAccessor):
    def __call__(self, *dims):
        """
        Sums the dataset along all dimensions except the specified ones.
        """
        if not len(dims) > 0:
            raise ValueError("Provide at least one dimension to exclude from summation")

        return self._xrds.sum(
            dim=[d for d in self._xrds.dims if d not in dims],
            skipna=True,
        )
