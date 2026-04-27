# MuTopia / signaturemodels

Genomic mutational topography modeling toolkit. Builds G-Tensors (genomic tensors integrating features with mutation data), trains signature decomposition models, and provides analysis/visualization tools.

## Project structure

- `mutopia/` - Main Python package
  - `analysis/` - Core analysis module, imported as `mutopia.analysis as mu`
  - `gtensor/` - G-Tensor data structures and operations (`mu.gt.*`)
  - `model/` - Topography model implementations
  - `plot/` - Plotting utilities (`mu.pl.*`, `mutopia.plot.track_plot as tr`)
  - `cli/` - CLI entry points (`gtensor` and `topo-model` commands)
  - `tuning.py` - Hyperparameter tuning via Optuna (`mu.tuning.*`)
  - `utils.py` - Utilities including `fill_jinja_template`
- `tutorials/` - Jupyter notebook tutorials (1-4)

## CLI tools

### `gtensor` - G-Tensor management
- `gtensor compose <config.yaml> -w <workers>` - Build G-Tensor from YAML config
- `gtensor info <file.nc>` - Display G-Tensor info
- `gtensor feature {ls,add,rm,edit} <file.nc>` - Manage features
- `gtensor sample add -id <id> --sample-weight <w> --mutation-rate-file <path> --pass-only <file.nc> <vcf>` - Add samples
- `gtensor split <file.nc> <test_chrom>` - Split train/test (creates `.train.nc` / `.test.nc`)
- `gtensor slice {regions,samples}` - Subset G-Tensors
- `gtensor offsets` - Manage locus exposure offsets
- `gtensor convert` - Convert modality
- `gtensor set-attr` - Set attributes
- `gtensor utils make-expression-bedfile` - Convert expression TSV to bed format

### `topo-model` - Model training and tuning
- `topo-model train -ds <train.nc> <test.nc> -k <components> -o <output.pkl> -@ <threads> --lazy --seed <n> --locus-subsample <rate> --num-epochs <n> --eval-every <n>`
- `topo-model study create <path> -ds <train.nc> <test.nc> -min <n> -max <n> --save-model -lsub <rate> [-e|-ee]`
- `topo-model study run <path> --lazy -@ <threads> [--time-limit <minutes>]`
- `topo-model study summary <path> -o <results.csv>`
- `topo-model study retrain <path> <trial_num> <output.pkl> --lazy -@ <threads>`

## Python API

### G-Tensor operations (`mu.gt.*`)
```python
data = mu.gt.lazy_load("file.nc")                          # Load with samples on disk
data = mu.gt.load_dataset("file.nc", with_samples=False)   # Load without samples
data = mu.gt.annot_empirical_marginal(data)                 # Annotate average counts per locus/context
features = mu.gt.fetch_features(data, "Gene*")              # Fetch features by glob pattern
data = mu.gt.slice_regions(data, "chr1:start-end")          # Slice by genomic region
data = mu.gt.slice_samples(data, ...)                       # Slice by samples
mu.gt.write_dataset(data, "out.nc", write_samples=False)    # Save to disk
components = mu.gt.list_components(data)                    # List model components
data = mu.gt.rename_components(data, new_names)             # Rename components
explanation = mu.gt.get_explanation(data, component="name")  # Get SHAP Explanation object
```

G-Tensors are extended xarray Datasets. Access sections via `data.sections["Features"]`, `data["Features/H3K27ac"]`, or `data["Regions/length"]`. Samples are loaded lazily: `data.list_samples()`, `data.fetch_sample("id")`.

### Model operations
```python
model_cls = mu.make_model_cls(train_data)
model = model_cls(num_components=15, seed=42, locus_subsample=1/8, threads=5, eval_every=10, num_epochs=1000)
model = model.fit(train, test)
model.save("model.pkl")
model = mu.load_model("model.pkl")

# Annotate data with model results (all-in-one):
data = model.annot_data(data, threads=5, calc_shap=True, subset_regions="chr2")

# Or individually:
data = model.annot_contributions(data)
data = model.annot_component_distributions(data)
data = model.annot_marginal_prediction(data)
data = model.annot_components(data)
data = model.annot_SHAP_values(data, threads=5)
```

### Plotting (`mu.pl.*` and `mutopia.plot.track_plot as tr`)
```python
# Signature plots
mu.pl.plot_signature_panel(data)
mu.pl.plot_component(data, "component_name")
mu.pl.plot_component(data, "component_name", "GeneStrand")  # With strand condition
mu.pl.plot_interaction_matrix(data, "component_name")
mu.pl.plot_shap_summary(data, scale=40, max_size=20, figsize=(10,6))

# Track plots - composable genome browser views
view = tr.make_view(data, "chr1:start-end")
config = lambda view, scalebar_bp: (
    tr.scale_bar(scalebar_bp, scale="mb"),
    tr.ideogram("cytoBand.txt"),
    tr.scatterplot(tr.select("variable"), s=0.5),
    tr.line_plot(tr.pipeline(tr.select("var"), view.smooth(20)), linewidth=0.5),
    tr.fill_plot(tr.select("Features/H3K27ac"), height=0.75),
    tr.heatmap_plot(tr.feature_matrix("feat1", "feat2"), height=1.25),
    tr.stack_plots(plot1, plot2, height=1.5, label="Label"),
    tr.spacer(0.2),
)
tr.plot_view(config, view, scalebar_bp=1_000_000, width=10)

# Data selectors for track plots
tr.select("variable_name")                    # Select a variable
tr.feature_matrix("feat1", "feat2")           # Select feature matrix
tr.pipeline(tr.select("var"), fn1, fn2)       # Chain transformations
tr.apply_rows(tr.renorm)                      # Per-row transformation

# Model-specific track plots
topography = tr.TopographyTransformer().fit(data)
tr.plot_topography(topography, height=2)
tr.tracks.plot_marginal_observed_vs_expected(view, pred_smooth=7, smooth=7, height=1.25)
tr.plot_component_rates(view, *tr.order_components(data))
```

### Tuning
```python
study, *_ = mu.tuning.load_study("studies/liver/01")
```

### Utilities
```python
from mutopia.utils import fill_jinja_template
config_str = fill_jinja_template(template, name="x", chromsizes="path", fasta="path", ...)
```

## G-Tensor YAML config format

```yaml
name: <name>
dtype: SBS
chromsizes: <path>
blacklist: <path>
fasta: <path>
region_size: 10000

features:
  <FeatureName>:
    normalization: <standardize|log1p_cpm|quantile|power|robust|categorical|mesoscale|strand|gex>
    column: <int, default=4>        # bed column to extract
    null: <int>                     # null sentinel value
    group: <str, default="all">     # feature interaction group
    classes: [<str>, ...]           # class ordering for categorical
    sources:
      - <path or URL>              # bed, bedGraph, or bigWig files
```

Normalization notes:
- `gex`: Required for expression data; normalizes per gene not per locus
- `categorical`: Discrete regions modeled with GBT estimators
- `mesoscale`: Discrete regions that influence signature distributions
- `strand`: Features with "+" or "-" values (gene/replication strand)

## Training best practices

- Always use subsampling (`locus_subsample` preferred); batch training overfits
- Target ~20s iteration time via `threads`, `locus_subsample`, `batch_subsample`
- Train until convergence (don't set `num_epochs` unless testing)
- Use `init_components` with COSMIC names for small datasets (e.g., `["SBS1", "SBS3"]`)
- Higher test scores = better models
- Use chr1 as test set by convention
- Use `--lazy` flag for CLI training to keep samples on disk
- For model selection: choose fewest components near maximum score (elbow method)
