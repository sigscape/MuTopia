from functools import wraps
import subprocess
import os
from .utils import logger


def compose_config(
    *,
    template_file,
    **kwargs,
):
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        raise ImportError("Please install Jinja2 to render the configuration template.")

    env = Environment(loader=FileSystemLoader("./"))
    template = env.get_template(template_file)

    return template.render(**kwargs)


def _run_if_not_exists(fn):
    @wraps(fn)
    def wrapper(target, *args, **kwargs):
        if not os.path.exists(target):
            logger.info(f"Running function for target: {target}")
            fn(target, *args, **kwargs)
        else:
            logger.info(f"Target already exists: {target}")
        return target

    return wrapper


def _rm_if_fail(fn):
    @wraps(fn)
    def wrapper(target, *args, **kwargs):
        try:
            return fn(target, *args, **kwargs)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(target):
                os.remove(target)
            raise Exception from e

    return wrapper


def _return_target(fn):
    @wraps(fn)
    def wrapper(target, *args, **kwargs):
        fn(target, *args, **kwargs)
        return target

    return wrapper


def set_target(conversion_fn):
    def outer(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            target = conversion_fn(*args, **kwargs)
            return fn(target, *args, **kwargs)

        return inner

    return outer


def prefix_with(prefix):
    dirname = os.path.dirname(prefix)
    os.makedirs(dirname, exist_ok=True)

    def outer(fn):
        return set_target(lambda x, *args, **kwargs: prefix + os.path.basename(x))(fn)

    return outer


def make_pipeline_fn(fn):

    @wraps(fn)
    @_run_if_not_exists
    @_return_target
    @_rm_if_fail
    def wrapper(target, *args, **kwargs):
        return fn(target, *args, **kwargs)

    return wrapper


def pipeline(*fns):
    def inner(input, *args, **kwargs):
        for fn in fns:
            input = fn(input, *args, **kwargs)
        return input

    return inner


def command_step(cmd):
    def inner(target, input, **kwargs):
        try:
            subprocess.check_call(
                cmd.format(
                    target=target,
                    input=input,
                    **kwargs,
                ),
                shell=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            raise Exception from e

    return make_pipeline_fn(inner)


@make_pipeline_fn
def fetch_data(target, url):
    import requests
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        os.makedirs("downloads", exist_ok=True)
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return target
