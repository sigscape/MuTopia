
from download_encode import EncodeExperimentConfig
from typing import Any, Dict
from collections import Counter

def get_term_matches(
    encode_confgs: list[EncodeExperimentConfig],
    search_term: str,
) -> Dict[tuple[str, str], int]:

    term_matches = Counter()
    for config in encode_confgs:
        biosample_desc = config.biosample_term_name.lower()
        if search_term.lower() in biosample_desc:
            term_matches[(biosample_desc, config.biosample_term_id)] += 1 
    return term_matches

def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Fetch UBERON term IDs from ENCODE biosample descriptions"
    )
    parser.add_argument(
        "ENCODE_CONFIGS",
        type=str,
        help="Path to the ENCODE experiment configuration YAML file",
    )
    parser.add_argument(
        "SEARCH_TERM",
        type=str,
        help="Search term to match in biosample descriptions",
    )
    args = parser.parse_args()

    with open(args.ENCODE_CONFIGS, "r") as f:
        encode_configs_data: Dict[str, Any] = yaml.safe_load(f)
    
    encode_configs = [
        EncodeExperimentConfig(**data)
        for data in encode_configs_data.values()
    ]
    
    term_matches = get_term_matches(
        encode_configs,
        args.SEARCH_TERM
    )

    term_matches = sorted(term_matches.items(), key=lambda x: len(x[0][0]))

    for (desc, term_id), count in term_matches:
        print(f"{desc}: {term_id} ({count})")

if __name__ == "__main__":
    main()