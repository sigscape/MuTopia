# MuTopia documentation

Sphinx site using the [Furo](https://pradyunsg.me/furo/) theme. Source lives in
`source/`, built HTML goes to `build/html/`.

## Building locally

Create the docs environment once:

```bash
conda create -n mutopia-docs -c conda-forge python=3.11
conda activate mutopia-docs
pip install -e "..[docs]"          # MuTopia + docs extras (autodoc needs the package)
pip install -r requirements.txt    # pinned Sphinx stack
```

Then build:

```bash
conda activate mutopia-docs
make html
open build/html/index.html
```

`make html` syncs any `*.ipynb` files from `../tutorials/` into
`source/tutorials/` before building. If no notebooks are present locally (e.g.
fresh clone) the sync is skipped and the committed copies in
`source/tutorials/` are used instead.

## Source layout

```
source/
├── conf.py                 Sphinx config (theme, extensions, autodoc hooks)
├── index.rst               Landing page
├── getting_started.rst
├── _static/
│   ├── custom.css
│   └── mutopia_logo.png
├── api/                    API reference (autodoc from docstrings)
│   ├── gtensor.rst
│   ├── model.rst
│   ├── plot.rst
│   ├── track_plot/
│   └── modes/
└── tutorials/
    ├── 1.building_a_gtensor.ipynb
    ├── 2.analyzing_gtensors.ipynb
    ├── 3.training_models.ipynb
    ├── 4.analyzing_models.ipynb
    └── 5.annotating_vcfs.rst
```

Notebooks are rendered as-is by nbsphinx (`nbsphinx_execute = 'never'`), so
outputs must be saved in the notebook before committing. Notebook sources are
tracked in `source/tutorials/` via a `.gitignore` exception; the raw dev copies
in `../tutorials/` remain gitignored.

## Deployment (CI)

The workflow at `.github/workflows/docs.yml` deploys to the `gh-pages` branch:

| Trigger | Deployed to |
|---------|-------------|
| Push to `main` | `/latest/` |
| Push `vX.Y.Z` tag | `/vX.Y.Z/` + `/stable/` + root redirect |

Old version directories are never deleted (`clean: false`). The root
`index.html` redirects to `/stable/`, so the canonical URL always points to the
latest release.

**One-time repo setup:** GitHub Settings → Pages → Source → "Deploy from a
branch" → `gh-pages` / `/ (root)`.

## Adding a new tutorial

1. Write the notebook in `../tutorials/` and save it with outputs.
2. Run `make html` locally — it will sync and render the notebook.
3. Commit `source/tutorials/<name>.ipynb` and add it to `source/tutorials/index.rst`.
