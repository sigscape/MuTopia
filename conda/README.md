# Bioconda recipe

Staging area for the bioconda recipe. The actual recipe lives in
[bioconda-recipes](https://github.com/bioconda/bioconda-recipes); this copy
exists so we can iterate on it in-repo before submitting.

## Why bioconda and not conda-forge

MuTopia calls four bioinformatics CLIs via `subprocess`:

- `bedtools`
- `bcftools`
- `tabix` (from `htslib`)
- `bigWigAverageOverBed` (from `ucsc-bigwigaverageoverbed`)

All four are on bioconda. By declaring them in `requirements.run`,
`conda install -c bioconda mutopia` pulls everything in one command. None of
this is possible on conda-forge (they don't host bioinformatics tools) or
PyPI (no way to express non-Python system dependencies).

## First-time submission

1. **Fork** https://github.com/bioconda/bioconda-recipes on GitHub.

2. **Clone the fork and copy the recipe:**

   ```bash
   git clone https://github.com/<your-user>/bioconda-recipes
   cd bioconda-recipes
   mkdir recipes/mutopia
   cp /path/to/signaturemodels/conda/meta.yaml recipes/mutopia/
   ```

3. **Create a branch, commit, push:**

   ```bash
   git checkout -b add-mutopia
   git add recipes/mutopia/meta.yaml
   git commit -m "Add mutopia"
   git push origin add-mutopia
   ```

4. **Open a PR** against `bioconda/bioconda-recipes`. Bioconda CI will:
   - Lint the recipe
   - Build it on `linux-64`, `linux-aarch64`, and `osx-arm64`
   - Run the `test:` commands inside a clean environment
   - Post results as PR comments

5. **Respond to review.** Bioconda maintainers will often request tweaks —
   common ones are version pins, renamed dependencies, or missing test
   commands. Push fixes to the same branch.

6. **Merge.** Once approved, a maintainer merges and the package goes live at
   `anaconda.org/bioconda/mutopia` within a few hours.

## Updating for future releases

You don't need to do anything manually — the bioconda **bump bot** watches
PyPI and automatically opens a PR updating `version` and `sha256` in
`recipes/mutopia/meta.yaml` every time a new PyPI release appears.

1. Cut a normal MuTopia release (`git tag v1.0.5 && git push`)
2. `publish.yml` uploads to PyPI
3. Within ~24 h, the bioconda bump bot opens a PR with the version bump
4. You (as recipe maintainer) review and merge the PR

The staging copy in this directory should be kept in sync with whatever's
actually merged to `bioconda-recipes`, so update it manually on each bump
if you want it to stay accurate.

## Keeping this recipe in sync

When a new version ships on PyPI, update `meta.yaml` here:

1. Change `{% set version = "..." %}` to the new version.
2. Fetch the new sdist sha256:

   ```bash
   curl -sL https://pypi.org/pypi/mutopia/<version>/json \
     | python3 -c "import sys, json; d = json.load(sys.stdin); \
         print([u['digests']['sha256'] for u in d['urls'] \
                if u['packagetype'] == 'sdist'][0])"
   ```

3. Update the `sha256:` line.
4. If `install_requires` in `setup.cfg` changed, update `requirements.run`.
