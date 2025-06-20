import numpy as np
from joblib import Parallel, delayed
from contextlib import contextmanager
from enum import Enum
import inspect
from functools import wraps
import logging
import subprocess
from gzip import open as gzopen
import time
from matplotlib.colors import LinearSegmentedColormap

# Create a custom diverging colormap
diverging_palette = LinearSegmentedColormap.from_list(
    "custom_diverging",
    ["#427aa1ff", "#FAFAFA", "#e07a5fff"],  # White or a neutral color at center
)

categorical_palette = ["#427aa1ff", "#e07a5fff", "#acacacff", "#83c5beff"]


@contextmanager
def safe_read(filename):
    yield gzopen(filename, "rt") if filename.endswith(".gz") else open(filename, "r")


logger = logging.getLogger(" Mutopia")
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)


def timer_wrapper(func, name=None):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.debug(f"{name or func.__name__} took {time.time() - start:.2f} seconds.")
        return result

    return wrapper


class FeatureType(Enum):
    GEX = "gex"
    LOG1P_CPM = "log1p_cpm"
    MESOSCALE = "mesoscale"
    STRAND = "strand"
    CATEGORICAL = "categorical"
    POWER = "power"
    MINMAX = "minmax"
    QUANTILE = "quantile"
    STANDARDIZE = "standardize"
    ROBUST = "robust"

    @classmethod
    def continuous_types(cls):
        return (
            FeatureType.GEX,
            FeatureType.LOG1P_CPM,
            FeatureType.POWER,
            FeatureType.MINMAX,
            FeatureType.QUANTILE,
            FeatureType.STANDARDIZE,
            FeatureType.ROBUST,
        )

    @property
    def is_continuous(self):
        return self in self.continuous_types()

    @property
    def allowed_dtypes(self):
        if self in (FeatureType.MESOSCALE, FeatureType.CATEGORICAL):
            return (str, np.str_, int, np.int_)
        elif self == FeatureType.STRAND:
            return (np.int8, np.int32, np.int_)
        elif self.is_continuous:
            return (float, np.float64, np.float32, np.double, int, np.int_)
        else:
            raise ValueError(f"FeatureType {self} not recognized.")

    @property
    def save_dtype(self):
        if self in (FeatureType.MESOSCALE, FeatureType.CATEGORICAL):
            return np.str_
        elif self == FeatureType.STRAND:
            return np.int8
        elif self.is_continuous:
            return np.float32
        else:
            raise ValueError(f"FeatureType {self} not recognized.")


def str_wrapped_list(x, n=4):
    x = list(map(str, x))
    if len(x) == 0:
        return "[]"
    return "\n\t" + ",\n\t".join([", ".join(x[i : i + n]) for i in range(0, len(x), n)])


@contextmanager
def ParContext(n_jobs, verbose=0, ordered=True):
    yield Parallel(
        n_jobs=n_jobs,
        backend="threading",
        return_as="generator" if ordered else "generator_unordered",
        verbose=verbose,
        pre_dispatch="n_jobs",
    )


def parallel_gen(function_generator, par_context=None, threads=1, ordered=True):
    with par_context or ParContext(threads, ordered=ordered) as par:
        for x in par(delayed(fn)() for fn in function_generator):
            yield x


def parallel_map(function_generator, par_context=None, threads=1, ordered=True):
    with par_context or ParContext(threads, ordered=ordered) as par:
        return list(par(delayed(fn)() for fn in function_generator))


def using_exposures_from(corpus):
    try:
        corpus.contributions
    except AttributeError:
        raise AttributeError(
            "The corpus does not have contributions. Run `model.annot_contributions(corpus)` first."
        )

    return lambda _, sample_name: corpus.contributions.sel(sample=sample_name).data


def using_priors_from(model_state):
    return lambda corpus, _: model_state.alpha[corpus.attrs["name"]]


def borrow_kwargs(*borrow_sigs):

    def decorator(func):

        # start the signature with the incipient function
        combined_params = dict(
            (
                (name, param)
                for name, param in inspect.signature(func).parameters.items()
                if not param.kind == inspect.Parameter.VAR_KEYWORD
            )
        )

        # iterate over the borrowed functions
        for f in borrow_sigs:
            sig = inspect.signature(f)
            # add the kwargs from the borrowed function
            for name, param in sig.parameters.items():
                if not param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    continue
                combined_params[name] = param

        # Sort the parameters so that the order is valid
        sorted_params = sorted(
            combined_params.values(),
            key=lambda p: (p.kind, p.default is not inspect.Parameter.empty),
        )
        combined_params = {param.name: param for param in sorted_params}
        # Create a new signature with combined parameters
        merged_signature = inspect.Signature(parameters=combined_params.values())

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Update the wrapper's signature
        wrapper.__signature__ = merged_signature
        return wrapper

    return decorator


def close_process(process):

    if not process.stdout is None:
        process.stdout.close()
    process.wait()

    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode,
            process.args,
        )


def stream_subprocess_output(process):

    while True:
        line = process.stdout.readline().strip()
        if not line:
            break
        yield line

    close_process(process)


# Parse regions into a list of (chrom, start, end) tuples
def parse_region(region):
    region = str(region).strip()

    if ":" in region:
        # Format: chr:start-end
        chrom, pos = region.split(":", 1)
        if "-" in pos:
            start, end = map(lambda x: int(x.replace(",", "")), pos.split("-", 1))
        else:
            # Handle case like "chr1:1000" (no end specified)
            start = int(pos)
            end = np.inf
    else:
        # Format: chr (entire chromosome)
        chrom = region
        start = 0
        end = np.inf

    return (chrom, start, end)
