from contextlib import contextmanager
from enum import Enum
from functools import wraps
import logging
from joblib import Parallel, delayed

logger = logging.getLogger(" Mutopia")
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

def fill_jinja_template(template_str, **kwargs):
    from jinja2 import Environment
    template = Environment().from_string(template_str)
    return template.render(**kwargs)

def plot_presets():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Helvetica",
            "font.weight": "ultralight",
            "axes.labelweight": "ultralight",
            "axes.titleweight": "ultralight",
            "figure.titleweight": "ultralight",
            "xtick.labelsize": "medium",
            "ytick.labelsize": "medium",
            "axes.linewidth": 0.5,
        }
    )


@contextmanager
def safe_read(filename):
    from gzip import open as gzopen

    with (
        gzopen(filename, "rt") if filename.endswith(".gz") else open(filename, "r")
    ) as f:
        yield f


def timer_wrapper(func, name=None):

    import time

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
        import numpy as np

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
        import numpy as np

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


def close_process(process):

    import subprocess

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
            end = float("inf")
    else:
        # Format: chr (entire chromosome)
        chrom = region
        start = 0
        end = float("inf")

    return (chrom, start, end)
