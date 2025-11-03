#!/usr/bin/env python

def set_parameters(tumor_type, max_components: int = 20):

    import pandas as pd

    exposures = pd.read_excel("annotations/41588_2024_1659_MOESM5_ESM.xlsx", sheet_name="Supplementary_Table_4")\
        .query("MMRD_POLE_category == 'MMRP'")\
        .set_index(["sample_id", "tumor_type", "MMRD_POLE_category"])
    
    exposures = exposures.div(exposures.sum(axis=1), axis=0)

    active_components = exposures.groupby("tumor_type")\
        .quantile(0.95).melt(ignore_index=False)\
        .query("value > 0.025")\
        .reset_index()\
        .groupby("tumor_type")\
        ["variable"].apply(list)

    matching_types = [tt for tt in active_components.index if tumor_type.removesuffix("-All").replace("-", ".") in tt]
    components = active_components[matching_types].explode().unique().tolist()
    components = [comp for comp in components if not comp.startswith("SBS40")]  # Exclude SBS40 as it is often uninformative
    
    max_components = max(min(max_components, len(components) * 2), len(components) + 1)
    min_components = max(5, len(components))

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
