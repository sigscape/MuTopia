#!/usr/bin/env python3

import subprocess
import tempfile
from numpy import array
import numpy as np
from ..genome_utils.bed12_utils import check_regions_file
from ..utils import safe_read


def make_continous_features_bigwig(
    bigwig_file,
    regions_file,
    *,
    extend=None,
    **kw,
):
    check_regions_file(regions_file)

    with tempfile.NamedTemporaryFile() as bed, tempfile.NamedTemporaryFile() as regions:

        with open(regions.name, "w") as r:
            subprocess.check_call(["cut", "-f", "1-4", regions_file], stdout=r)

        subprocess.check_output(
            ["bigWigAverageOverBed", bigwig_file, regions.name, bed.name]
        )

        with open(bed.name, "r") as bed:
            data = map(lambda s: s.strip().split("\t"), bed)
            data = map(lambda s: (int(s[0]), float(s[5])), data)
            data = sorted(data, key=lambda x: x[0])

            vals = array(list(map(lambda x: x[1], data)))

    if not len(vals) > 1:
        raise RuntimeError(f"No values found in {bigwig_file} for {regions_file}")

    return vals


def make_continuous_features_bed(
    bed_file,
    regions_file,
    *,
    null="nan",
    column: int = 4,
    **kw,
):
    check_regions_file(regions_file)

    map_out = subprocess.check_output(
        [
            "bedtools",
            "map",
            "-a",
            regions_file,
            "-b",
            bed_file,
            "-c",
            str(column),
            "-o",
            "mean",
            "-null",
            null,
        ]
    )

    vals = []
    for line in map_out.decode().strip().split("\n"):
        vals.append(float(line.strip().split("\t")[-1]))
    vals = array(vals)

    if not len(vals) > 1:
        raise RuntimeError(f"No values found in {bed_file} for {regions_file}")

    return vals


def make_continous_features_bedgraph(
    bedgraph_file,
    regions_file,
    *,
    null="nan",
    **kw,
):
    return make_continuous_features_bed(
        bedgraph_file,
        regions_file,
        null=null,
        column=4,
    )


def make_distance_features(
    bedfile,
    regions_file,
):
    ##
    # TODO: Handle gzipped files!
    ##
    check_regions_file(regions_file)

    def _find_stranded_closest_feature(strand):

        strand_process = subprocess.Popen(
            [
                "awk",
                "-v",
                "OFS=\t",
                f'{{print $1,$2,$3,NR-1,0,"{strand}"}}',
                regions_file,
            ],
            stdout=subprocess.PIPE,
        )

        closest_out = subprocess.check_output(
            [
                "bedtools",
                "closest",
                "-a",
                "-",
                "-b",
                bedfile,
                "-d",
                "-id",
                "-D",
                "a",
                "-t",
                "first",
            ],
            stdin=strand_process.stdout,
        )

        strand_process.wait()

        return -array(
            list(
                map(
                    lambda x: x.split("\t")[-1],
                    closest_out.decode().strip().split("\n"),
                )
            )
        ).astype(float)

    upstream = _find_stranded_closest_feature("+")
    downstream = _find_stranded_closest_feature("-")

    nan_mask = (upstream < 0.0) | (downstream < 0.0) | (upstream + downstream <= 0.0)

    progress = upstream / (upstream + downstream + 1)
    progress = np.minimum(progress, 1 - progress)

    # progress = 1. - progress if reverse else progress

    total_distance = upstream + downstream

    progress[nan_mask] = 0.0
    total_distance[nan_mask] = 0.0

    return progress, total_distance


def make_discrete_features(
    bed_file,
    regions_file,
    *,
    column=4,
    null="None",
    class_priority=None,
):
    check_regions_file(regions_file)

    def _resolve_class_priority(vals, _class_priority):
        vals = set(vals).difference({null})

        if len(vals) == 0:
            return null
        elif len(vals) == 1:
            return vals.pop()
        else:
            for _class in _class_priority:
                if _class in vals:
                    return _class
            else:
                raise RuntimeError(
                    f"Could not resolve class priority for {vals} using {class_priority}"
                )

    # check that the bedfile has 4 columns
    with safe_read(bed_file) as f:
        for line in f:
            if line.startswith("#"):
                continue

            cols = line.strip().split("\t")
            if len(cols) < column:
                raise ValueError(
                    f"Bedfile {bed_file} must have at least {column} columns."
                    "The fourth column should be the name of the class for that region."
                )
            break

    cmd = [
        "bedtools",
        "map",
        "-a",
        regions_file,
        "-b",
        bed_file,
        "-o",
        "distinct",
        "-c",
        str(column),
        "-null",
        str(null),
        "-delim",
        "|",
        "-sorted",
        "-split",
    ]

    map_out = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
    )

    awk_out = subprocess.check_output(
        ["awk", " {print $NF}"],
        stdin=map_out.stdout,
    )

    mappings = [x.strip() for x in awk_out.decode().strip().split("\n")]
    vals = [m.split("|") for m in mappings]
    classes = set([_v for v in vals for _v in v]).difference({null})

    if class_priority is None:
        class_priority = sorted(list(classes))
    else:
        assert (
            set(class_priority) == classes
        ), f"Class priority must contain all classes in {classes}, non including the null class: {null}"

    vals = array([_resolve_class_priority(v, class_priority) for v in vals])

    return (vals, list(reversed(list(class_priority) + [null])))


def make_strand_features(
    bed_file,
    regions_file,
    *,
    column=4,
):
    vals, _ = make_discrete_features(
        bed_file,
        regions_file,
        column=column,
        null=".",
        class_priority=["-", "+"],
    )

    VAL_MAP = {
        "-": -1,
        ".": 0,
        "+": 1,
    }
    vals = np.array([VAL_MAP[v] for v in vals])

    return vals
