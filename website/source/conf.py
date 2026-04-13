# Configuration file for the Sphinx documentation builder.
from __future__ import annotations
import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# Add repository root (one level above `website/`) to sys.path so autodoc imports local sources
DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
WEBSITE_DIR = os.path.abspath(os.path.join(DOCS_DIR, '..'))
REPO_ROOT = os.path.abspath(os.path.join(WEBSITE_DIR, '..'))
# Prepend the repo root which contains the `mutopia/` package directory
sys.path.insert(0, REPO_ROOT)

# -- Project information -----------------------------------------------------
project = 'MuTopia'
author = 'Allen W. Lynch'
copyright = f"{datetime.now():%Y}, {author}"
release = ''

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'sphinx_copybutton',
    'sphinx_design',
]

autosummary_generate = True
# Only use the class docstring for classes (not the __init__ docstring)
autoclass_content = 'class'
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': False,
}
# Prefer docstring typehints in descriptions and avoid noisy duplication
typehints_fully_qualified = False
always_document_param_types = False
autodoc_typehints = 'none'
typehints_use_signature = True
typehints_use_signature_return = True
typehints_format = 'short'
set_type_checking_flag = True
suppress_warnings = [
    'ref.ref',  # suppress undefined label warnings treated as errors by Makefile -W
]

templates_path = ['_templates']
exclude_patterns: list[str] = []

# -- nbsphinx configuration --------------------------------------------------
# Don't execute notebooks during build
nbsphinx_execute = 'never'
# Allow errors in notebooks
nbsphinx_allow_errors = True
# Timeout for notebook execution (if needed)
nbsphinx_timeout = 180
# Remove execution prompt numbers from rendered notebooks
nbsphinx_prompt_width = '0'

# -- sphinx-copybutton configuration -----------------------------------------
# Strip prompts from copied code
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
# Remove output prompts
copybutton_only_copy_prompt_lines = False

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    # Branding
    'light_logo': 'mutopia_logo.png',
    'dark_logo': 'mutopia_logo.png',
    # Also configurable via CSS variables; set here for both themes
    'light_css_variables': {
        'content-width': '85rem',
        'toc-width': '22rem',
    },
    'dark_css_variables': {
        'content-width': '85rem',
        'toc-width': '22rem',
    },
}


def setup(app):
    """Sphinx hook to customize autodoc processing.

    Suppresses the module-level docstring for mutopia.gtensor.gtensor so the
    page can provide its own introduction without duplication.
    """

    def _skip_module_docstring(app, what, name, obj, options, lines):
        # Only target the module docstring for the GTensor module
        if what == 'module' and name == 'mutopia.gtensor.gtensor':
            # Clear the collected lines to hide the module docstring
            lines.clear()

    def _skip_topography_init(app, what, name, obj, skip, options):
        # Skip the __init__ for TopographyModel so its parameters don't show
        try:
            if name.endswith('__init__') and getattr(obj, '__qualname__', '').startswith('TopographyModel.'):
                return True
        except Exception:
            pass
        return None

    def _hide_topography_signature(app, what, name, obj, options, signature, return_annotation):
        # Hide the class signature from the header for TopographyModel
        if what == 'class' and name == 'mutopia.model.base.TopographyModel':
            return '', None
        return None

    def _scrub_topography_parameters(app, what, name, obj, options, lines):
        # As a safety net, remove any "Parameters" section from TopographyModel doc content
        if what == 'class' and name == 'mutopia.model.base.TopographyModel':
            out = []
            i = 0
            while i < len(lines):
                if lines[i].strip() == 'Parameters' and i + 1 < len(lines) and set(lines[i+1].strip()) == {'-'}:
                    # skip until a blank line followed by a non-indented line (next section)
                    i += 2
                    while i < len(lines) and (lines[i].strip() == '' or lines[i].startswith(' ') or lines[i].startswith('    ')):
                        i += 1
                    continue
                out.append(lines[i])
                i += 1
            lines[:] = out

    app.connect('autodoc-process-docstring', _skip_module_docstring)
    app.connect('autodoc-skip-member', _skip_topography_init)
    app.connect('autodoc-process-signature', _hide_topography_signature)
    app.connect('autodoc-process-docstring', _scrub_topography_parameters)
