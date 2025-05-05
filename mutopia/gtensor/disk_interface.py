import sparse
import xarray as xr
import pickle
import datatree
from numpy import nan, array, float32, int32
from functools import wraps
import time
import netCDF4 as nc
import tqdm
import os
from ..utils import FeatureType, logger
from .gtensor import *

WRITE_KW = dict(
    format="NETCDF4",
    engine="netcdf4",
)


def retry_until_write(func, n_tries=1000, sleep=1):

    @wraps(func)
    def wrapper(*args, **kwargs):

        for t in range(n_tries):
            try:
                return func(*args, **kwargs)
            except OSError:
                if t > 10:
                    logger.warning(
                        "Could not open netCDF file for writing - likely because of concurrent write traffic.\n"
                        "If the G-tensor is in-use, this will block writing."
                    )
                time.sleep(sleep)
                pass

        raise TimeoutError(
            "Could not open netCDF file for writing.\n"
            "This is likely because there is too much concurrent write traffic."
        )

    return wrapper


@retry_until_write
def read_attrs(dataset):
    with nc.Dataset(dataset, "r") as dset:
        attrs = {attr: getattr(dset, attr) for attr in dset.ncattrs()}

        attrs["name"] = dset.__dict__["name"]

    return attrs


def fetch_regions_path(dataset):

    regions_file = read_attrs(dataset)["regions_file"]

    if regions_file is None:
        raise ValueError(
            "No regions file found in dataset attributes. This probably means you are attempting an ingestion operation on a subset corpus view, which is not allowed."
        )

    regions_path = os.path.join(os.path.dirname(dataset), regions_file)

    if not os.path.exists(regions_path):
        raise FileNotFoundError(
            f"No such file exists: {regions_path}, maybe it was deleted? You can set a new regions_file with the command:\n\t"
            f"gtensor set-attr {dataset} -set regions_file <path/to/file>"
        )

    return regions_path


@retry_until_write
def read_dims(dataset):
    with nc.Dataset(dataset, "r") as dset:
        return {dim: len(dset.dimensions[dim]) for dim in dset.dimensions.keys()}


@retry_until_write
def read_coords(dataset):
    with nc.Dataset(dataset, "r") as dset:
        return {coord: dset.variables[coord][:] for coord in dset.dimensions.keys()}


@retry_until_write
def write_context_freqs(
    dataset,
    context_freqs,
):
    context_freqs.name = "context_frequencies"

    context_freqs.to_netcdf(
        dataset,
        group="regions",
        mode="a",
        **WRITE_KW,
    )


##
# feature write, remove, and list
##
@retry_until_write
def write_feature(
    dataset,
    vals,
    group: str = "default",
    *,
    name: str,
    normalization: FeatureType,
    **attrs,
):
    xr.DataArray(
        array(vals).astype(normalization.save_dtype),
        dims=("locus",),
        name=name,
        attrs={
            "normalization": normalization.value,
            "group": group,
            "active": 1,
            **attrs,
        },
    ).to_netcdf(
        dataset,
        group="features",
        mode="a",
        **WRITE_KW,
    )


@retry_until_write
def rm_feature(
    dataset,
    name: str,
):
    with nc.Dataset(dataset, "a") as dset:

        if not "features" in dset.groups:
            raise ValueError(
                "No `features` group found in dataset - are you sure this is a G-Tensor?"
            )

        features = dset.groups["features"]
        try:
            features[name].active = 0
        except IndexError:
            pass


@retry_until_write
def list_features(dataset):

    with nc.Dataset(dataset, "r") as dset:
        if not "features" in dset.groups:
            raise ValueError(
                "No `features` group found in dataset - are you sure this is a G-Tensor?"
            )

        return {
            feature_name: feature.__dict__
            for feature_name, feature in dset.groups["features"].variables.items()
            if not "active" in feature.__dict__ or feature.__dict__["active"]
        }


@retry_until_write
def edit_feature_attrs(
    dataset,
    name,
    group=None,
    normalization: FeatureType = None,
):

    with nc.Dataset(dataset, "a") as dset:
        if not "features" in dset.groups:
            raise ValueError(
                "No `features` group found in dataset - are you sure this is a G-Tensor?"
            )

        features = dset.groups["features"]
        try:
            feature = features[name]
        except IndexError:
            raise ValueError(f"Feature {name} not found in dataset.")

        if group is not None:
            feature.group = group

        if normalization is not None:
            feature.normalization = normalization.value


##
#
##


##
# Sample write, remove, and list
##
@retry_until_write
def write_sample(
    filename,
    sample,
    sample_name,
):
    
    arr = sample.X
    
    dset = (
        arr.sparse_to_coo()
        if isinstance(arr.data, sparse.SparseArray)
        else arr.to_dataset(name="data", promote_attrs=True)
    )

    if len(dset.data) == 0:
        maxval = 1.0
    else:
        maxval = dset.data.max().item() + 1.0

    prec = float32(maxval / 65535)

    dset.attrs["active"] = 1

    dset.to_netcdf(
        filename,
        group=f"/raw/X/{sample_name}",
        mode="a",
        encoding={
            "data": {
                "dtype": "uint16",
                "scale_factor": prec,
                "_FillValue": 0.0,
                "add_offset": -prec,
            },
            "indices": {
                "dtype": "uint32",
            },
        },
        **WRITE_KW,
    )


@retry_until_write
def list_samples(dataset):

    with nc.Dataset(dataset, "r") as dset:
        if not "raw" in dset.groups:
            raise ValueError(
                "No `raw` group found in dataset - are you sure this is a G-Tensor?"
            )

        return list(
            sname
            for sname, sample in dset.groups["raw"]["X"].groups.items()
            if not "active" in sample.__dict__ or sample.__dict__["active"]
        )


@retry_until_write
def rm_sample(
    dataset,
    sample_name: str,
):

    with nc.Dataset(dataset, "a") as dset:
        if not "raw" in dset.groups:
            raise ValueError(
                "No `raw` group found in dataset - are you sure this is a G-Tensor?"
            )

        raw = dset.groups["raw"]
        raw["X"][sample_name].active = 0


##
#
##


def write_dataset(dataset, filename, bar=False):

    section_names = dataset.sections.names

    # write base data (coords, index, attrs)
    (
        dataset.drop_vars(list(dataset.data_vars)).to_netcdf(
            filename, group="/", mode="w", **WRITE_KW
        )
    )

    for section_name, section in dataset.drop_vars(["X"]).sections:

        # check if any are sparse - if so error with a nice message
        sparse_vars = [varname for varname, var in section.data_vars.items() if var.is_sparse()]

        if len(sparse_vars) > 0:
            raise ValueError(
                f"Section {section_name} contains sparse variables: {sparse_vars}. "
                "Please convert to dense before writing."
            )

        section.to_netcdf(
            filename,
            group="/" + section_name,
            mode="a",
            encoding={k: {"dtype": v.dtype} for k, v in section.data_vars.items()},
            **WRITE_KW,
        )

    if len(dataset.list_samples()) > 0:
        
        for sample_name in (
            dataset.list_samples()
            if not bar
            else tqdm.tqdm(dataset.list_samples(), desc="Writing samples", ncols=100)
        ):
            write_sample(
                filename,
                dataset.fetch_sample(sample_name),
                sample_name,
            )

def _is_sparse_coo(group):
    return group.__dict__.get("format", "not coo") == "COO"


def _load_sparse(group, coo=False, fill_value=0.0, **kw):

    weights = group["data"][...].data
    coords = group["indices"][...].data.astype(int32, copy=False)
    dims = tuple(map(str, group["obs_indices"][...]))
    shape = group.shape

    arr = sparse.COO(
        coords,
        weights,
        shape=shape,
        fill_value=fill_value,
    )

    arr = xr.DataArray(
        arr,
        dims=dims,
        name="X",
    )

    if not coo:
        return arr.ascsr("locus")

    return arr


def _load_dense(filename, sample_name, **kwargs):
    raise NotImplementedError()
    with xr.open_dataarray(filename, group=sample_name, engine="netcdf4") as dset:
        return dset.load()


def _backend_load_sample(dset, sample_name, coo=False):
    
    X_group = dset.groups["raw"]["X"][sample_name]
    X = (_load_sparse if _is_sparse_coo(X_group) else _load_dense)(X_group, coo=coo)
    
    return (
        xr.Dataset({"X": X})
        .assign_coords({"sample": sample_name})
    )


def load_sample(filename, sample_name, coo=False):

    if filename.endswith(".pkl"):
        raise NotImplementedError(
            "Loading samples from pickled files is not supported."
        )

    with retry_until_write(nc.Dataset)(filename, "r") as dset:
        return _backend_load_sample(
            dset,
            sample_name,
            coo=coo,
        )


def yield_samples(filename, *sample_names, coo=False):

    if filename.endswith(".pkl"):
        raise NotImplementedError(
            "Loading samples from pickled files is not supported."
        )

    with retry_until_write(nc.Dataset)(filename, "r") as dset:
        for sample_name in sample_names:
            try:
                yield _backend_load_sample(
                    dset,
                    sample_name,
                    coo=coo,
                )
            except KeyError:
                raise ValueError(f"Sample {sample_name} not found in dataset.")


class NoSamplesError(ValueError):
    pass


def _list_sample_names(filename):
    with nc.Dataset(filename, "r") as dset:
        if "raw" in dset.groups and "X" in dset.groups["raw"].groups:
            return list(
                sorted(
                    sname
                    for sname, sample in dset.groups["raw"]["X"].groups.items()
                    if not "active" in sample.__dict__ or sample.__dict__["active"]
                )
            )
        else:
            raise NoSamplesError(
                "This dataset has no samples yet, it will not be compatible with many functions."
            )


def _pack_samples(samples):

    samples = xr.concat(samples, dim="sample")

    if isinstance(samples.X.data, sparse.SparseArray):
        samples.X.ascsr('sample','locus')
        X = samples.X.data
        X.coords = X.astype(int32, copy=False)
    
    return samples


def load_dataset(filename, with_samples=True, with_state=True):

    def open_ds(**kw):
        with retry_until_write(xr.open_dataset)(filename, engine="netcdf4", **kw) as ds:
            return ds.load()
        

    if filename.endswith(".pkl"):
        with open(filename, "rb") as f:
            corpus = pickle.load(f)

        return corpus

    with retry_until_write(nc.Dataset)(filename, "r") as dset:
        section_names = list(dset.groups.keys())

    # load the base data
    dataset = open_ds()

    # load the sections
    for section_name in section_names:

        if section_name == "raw" or (section_name == "state" and not with_state):
            continue

        section = open_ds(group=section_name)
        section = section[[k for k,v in section.data_vars.items() if v.attrs.get("active", 1) == 1]]

        # the "data" section is special - it contains everything not in another section
        if not section_name.title() == "Data":
            section = section.rename(
                {k: f"{section_name.title()}/{k}" for k in section.data_vars}
            )

        dataset = dataset.merge(section)

    ## load the samples
    try:

        if not with_samples:  # just jump out if we don't want samples
            raise NoSamplesError

        sample_names = _list_sample_names(filename)
        dataset = dataset.assign_coords({"sample": sample_names})
        samples = _pack_samples(list(yield_samples(filename, *sample_names, coo=True)))
        
        dataset = dataset.merge(samples)

    except NoSamplesError:
        pass

    dataset.attrs["filename"] = filename
    return dataset
