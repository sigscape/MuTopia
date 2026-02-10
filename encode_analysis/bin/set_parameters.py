#!/usr/bin/env python

ADJUSTMENTS = {
    "ColoRect-AdenoCA" : (
        ("SBS94",), ("SBS4",),
    ),
    "Eso-AdenoCA" : (
        ("SBS37",), ("SBS28",),
    ),
    "Head-SCC" : (
        (), ("SBS33",),
    ),
    "Liver-HCC" : (
        (), ("SBS17b",),
    ),
    "Kidney-All" : (
        (), ("SBS40a", "SBS40b", "SBS40c", "SBS17b",),
    ),
    "Prost-AdenoCA" : (
        (), ("SBS33", "SBS3"),
    ),
}

def list_components(tumor_type: str, max_components: int = 20):
    import pandas as pd

    # Load and normalize exposures
    exposures = pd.read_excel("annotations/41588_2024_1659_MOESM5_ESM.xlsx", sheet_name="Supplementary_Table_4")\
        .query("MMRD_POLE_category == 'MMRP'")\
        .set_index(["sample_id", "tumor_type", "MMRD_POLE_category"])
    
    exposures = exposures.div(exposures.sum(axis=1), axis=0)

    # Find matching tumor types
    tumor_type_search = tumor_type.removesuffix("-All").replace("-", ".")
    matching_types = [tt for tt in exposures.index.get_level_values("tumor_type").unique() if tumor_type_search in tt]
    
    if not matching_types:
        raise ValueError(f"No matching tumor types found for {tumor_type}")
    
    # Filter exposures to matching tumor types, then calculate 95th quantile occurrence
    matched_exposures = exposures[exposures.index.get_level_values("tumor_type").isin(matching_types)]
    occurrence = matched_exposures.quantile(0.90)
    
    # Sort components by occurrence (descending) and filter
    components_sorted = occurrence.sort_values(ascending=False)
    
    # Get components with occurrence > 0.1 for min_components calculation
    high_occurrence = components_sorted[components_sorted > 0.1]
    min_components = max(4, len(high_occurrence))
    
    # Get components with occurrence > 0.025 for initialization
    active_components = components_sorted[components_sorted > 0.025]
    components = [comp for comp in active_components.index.tolist() if not (comp.startswith("SBS40") or comp == "SBS34")]

    max_components = max(min(max_components, len(components) * 2), len(components) + 5, 10)

    return components, min_components, max_components


def set_parameters(tumor_type, max_components: int = 20):
    components, min_components, max_components = list_components(tumor_type, max_components)
    print(
        f"-min {min_components} -max {max_components} ",
        *[f"-init {comp}" for comp in components],
        sep=" ",
    )

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("TUMOR_TYPE", type=str)
    parser.add_argument("MAX_COMPONENTS", type=int, default=20)
    args = parser.parse_args()

    set_parameters(args.TUMOR_TYPE, args.MAX_COMPONENTS)

if __name__ == "__main__":
    main()
