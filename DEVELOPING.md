# Developer notes

Internal notes on releasing, CI workflows, and where published artifacts live.
Not rendered on the GitHub repo home page — put user-facing docs in `README.rst`
or the Sphinx site.

## Remotes

| Remote | URL | Purpose |
|---|---|---|
| `origin` | `github.com/AllenWLynch/Mutopia` | Allen's personal fork |
| `sigscape` | `github.com/sigscape/MuTopia` | Canonical published home (docs + CI) |

Push both when releasing:

```bash
git push origin main
git push sigscape main
```

## Workflows

All workflow files live in `.github/workflows/`.

| File | Triggers | What it does |
|---|---|---|
| `docs.yml` | push to `main`, `v*.*.*` tag | Builds Sphinx docs → deploys to `gh-pages` branch. Main → `/latest/`. Tag → `/vX.Y.Z/` + `/stable/` + root redirect. |
| `publish.yml` | `v*.*.*` tag | Builds sdist + wheel → publishes to PyPI via trusted publishing (no secrets). |
| `docker.yml` | push to `main`, `v*.*.*` tag | Builds multi-arch image (linux/amd64, linux/arm64) → pushes to Docker Hub. Main → `:edge`. Tag → `:X.Y.Z`, `:X.Y`, `:X`, `:latest`. |

## Cutting a release

1. Bump the version in `setup.cfg` (or `pyproject.toml`, whichever holds it).
2. Commit: `git commit -am "bump version 1.0.5"`
3. Push: `git push sigscape main`
4. Tag: `git tag v1.0.5`
5. Push the tag: `git push sigscape v1.0.5`

That triggers, in parallel:

- **PyPI publish** — live at `pypi.org/project/mutopia/` within ~30 seconds
- **Docker Hub publish** — live at `hub.docker.com/r/allenlynch/mutopia` in ~5–10 minutes (multi-arch build is slow)
- **Docs versioning** — `sigscape.github.io/MuTopia/v1.0.5/` and `/stable/` updated

### Race condition caveat

`publish.yml` and `docker.yml` fire simultaneously on a tag push. The Docker
image build runs `pip install mutopia==X.Y.Z`, which needs PyPI to have already
indexed the new version. Usually fine (PyPI is fast), but if it fails:

1. Wait until PyPI shows the new version
2. Re-run the failed `docker.yml` run from the Actions tab

If this becomes a chronic problem, merge the two workflows so Docker `needs:` PyPI.

## Manual workflow triggers

All three workflows have `workflow_dispatch:` enabled. From the GitHub UI:

**Actions → [workflow name] → Run workflow → Run workflow**

Useful for:
- Testing the Docker build without cutting a real release (publishes as `:edge`)
- Rebuilding docs without pushing to `main`
- Re-running a failed PyPI publish after fixing a problem

## Required secrets and environments

On `sigscape/MuTopia` in **Settings**:

### Repository secrets (Settings → Secrets and variables → Actions)

- `DOCKERHUB_USERNAME` = `allenlynch`
- `DOCKERHUB_TOKEN` = personal access token from `hub.docker.com/settings/security`

### Environments (Settings → Environments)

- `pypi` — used by `publish.yml`. No secrets stored; the environment gate is
  what PyPI's trusted publishing validates against.

### One-time Pages setup

Settings → Pages → Source → "Deploy from a branch" → `gh-pages` / `/ (root)`.

### One-time PyPI setup

Configure the trusted publisher at
`pypi.org/manage/project/mutopia/settings/publishing/`:

| Field | Value |
|---|---|
| Owner | `sigscape` |
| Repository | `MuTopia` |
| Workflow filename | `publish.yml` |
| Environment name | `pypi` |

## Docs build (local)

See `website/README.md`. TL;DR:

```bash
conda activate mutopia-docs
cd website && make html
open build/html/index.html
```

Served live at:

- **Latest main:** https://sigscape.github.io/MuTopia/latest/
- **Latest release:** https://sigscape.github.io/MuTopia/stable/
- **Specific version:** https://sigscape.github.io/MuTopia/v1.0.5/

## Future: bioconda

Recipe lives in a fork of `bioconda/bioconda-recipes`, not in this repo. See
the conda section in the release notes — the bioconda bump bot watches PyPI
and opens update PRs automatically once the initial recipe is merged.
