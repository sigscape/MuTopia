Topography UMAP API
===================

Project components from a newly-trained model into the published pan-cancer
topography UMAP — a shared space built from each component's normalized
mutation-rate profile across chr2 10 kb bins.

Requires the optional ``umap-learn`` dependency::

    pip install mutopia[umap]

Quick start
-----------

.. code-block:: python

    import mutopia.analysis as mu

    # Layout only: ~15 kB, pandas alone, no umap-learn needed.
    coords = mu.load_reference_coordinates()

    # The full projector.
    ref = mu.load_reference_umap()

    data = mu.gt.load_dataset("my_cohort.annotated.nc", with_samples=False)
    ref.transform(data)        # coordinates, topo-cluster, novelty score
    ref.kneighbors(data, k=5)  # nearest reference components

Projecting a model trained on a different grid
----------------------------------------------

The reference feature space is defined on a specific ordered locus axis — the
same regions bed and the same mesoscale features. A gTensor built differently
cannot be matched to it, because mesoscale-split loci share ``(chrom, start)``
and no stored column separates them.

A model, however, is a function of genomic features, so evaluate it on a
reference-axis gTensor instead and the projection stays on the strict path:

.. code-block:: python

    model = mu.load_model("my_cohort.model.pkl")
    grid = mu.gt.load_dataset(reference_nc, with_samples=False, with_state=False)

    ref.transform_model(model, grid)
    ref.kneighbors_model(model, grid, k=5)

    # The annotated gTensor itself, e.g. for mutopia.plot.track_plot
    annotated = mu.annot_component_rates(model, grid)

Pass ``with_state=False``: a previous model's corpus state must be shed, and the
pan-cancer models convolve each feature over neighbouring bins, leaving a
51-long ``feature`` coordinate that a different model will reject.

Interpreting the output
-----------------------

The 2D coordinates are for display. ``transform`` is an approximate
re-optimization of new points against a frozen layout of a few hundred points.
The **nearest-neighbour table** and the **cluster vote** — both computed in the
full-dimensional cosine space — are the defensible readouts.

Two failure modes are reported explicitly, because neither is visible in the
coordinates:

``is_outlier`` / ``outlier_score``
    A local outlier factor over the reference. Structureless input (noise,
    shuffled loci) still lands somewhere plausible and receives a confident
    cluster call; this is what catches it. Nearest-neighbour distance does not,
    because a bland profile sits near the centroid where the reference is densest.

``set_is_degenerate`` / ``set_spread``
    A *set*-level check. Components that are near-identical to one another each
    land beside whatever reference point is closest to the centroid, and every
    per-component statistic looks normal. This fires when the projected set's
    median pairwise cosine distance falls below that of the least diverse
    reference cohort.

Limitations
-----------

- The reference space is **chr2 only**; a new gTensor must span it.
- Cluster labels are hand-curated, so the k-NN vote's in-sample ceiling is
  ~0.82, not 1.0.
- ``on_grid_mismatch="aggregate"`` exists for gTensors that cannot match the
  axis, but is lossy — pairwise cosine distances correlate r = 0.89 with the raw
  axis and only 61% of components keep the same nearest neighbour. It warns at
  runtime. Prefer the model route above.

Reference
---------

.. autoclass:: mutopia.analysis.topography_umap.TopographyUMAP
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__

.. autofunction:: mutopia.analysis.topography_umap.load_reference_umap

.. autofunction:: mutopia.analysis.topography_umap.load_reference_coordinates

.. autofunction:: mutopia.analysis.topography_umap.annot_component_rates
