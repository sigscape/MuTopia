#!/usr/bin/env python3

import subprocess
import tempfile
import os
import tqdm
from contextlib import contextmanager
import sys
from mutopia.utils import logger, close_process


@contextmanager
def make_genome_file(vcf_file, chr_prefix="", require_length=False):

    if not any(os.path.exists(vcf_file + ext) for ext in [".tbi", ".csi"]):
        subprocess.check_call(["bcftools", "index", vcf_file])

    contigs = subprocess.check_output(
        ["bcftools", "index", "--all", "--stats", vcf_file],
        universal_newlines=True,
        stderr=sys.stderr,
    )

    with tempfile.NamedTemporaryFile(delete=False) as f:
        for line in contigs.split("\n"):
            chrom, length, *_ = line.split("\t")

            if length == ".":
                if require_length:
                    raise ValueError(
                        f"Chromosome {chrom} does not have a length in the VCF file. Convert to BCF format."
                    )
                length = int(1e9)

            try:
                length = int(length)
            except ValueError:
                raise ValueError(
                    f"Chromosome {chrom} has a non-integer length ({length}) in the VCF file."
                )

            print(chr_prefix + chrom, length, sep="\t", file=f)

    try:
        yield f.name
    finally:
        os.remove(f.name)


@contextmanager
def stream_passed_SNVs(
    vcf_file,
    query_string,
    output=subprocess.PIPE,
    filter_string=None,
    pass_only=True,
    sample=None,
    chr_prefix="",
    sorted=True,
):

    filter_basecmd = ["bcftools", "view", "-v", "snps"]

    if pass_only:
        filter_basecmd += ["-f", "PASS"]

    if not sample is None:
        filter_basecmd += ["-s", sample]

    if not filter_string is None:
        filter_basecmd += ["-i", filter_string]

    # Suppress bcftools view output by redirecting stderr to DEVNULL
    filter_process = subprocess.Popen(
        filter_basecmd + [vcf_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        universal_newlines=True,
        bufsize=10000,
    )

    query_process = subprocess.Popen(
        ["bcftools", "query", "-f", chr_prefix + query_string],
        stdin=filter_process.stdout,
        stdout=subprocess.PIPE if sorted else output,
        stderr=subprocess.PIPE,  # Capture stderr to detect errors
        universal_newlines=True,
        bufsize=10000,
    )
    
    # Close filter_process stdout so query_process can detect EOF
    if filter_process.stdout:
        filter_process.stdout.close()

    sort_process = None
    if sorted:
        # ahh so annoying - it would be nice to delegate sorting to the user
        # but it can be difficult to make sure the VCFs are in lexigraphical order.
        # Instead, I'll bite the bullet and sort here.
        sort_process = subprocess.Popen(
            "LC_COLLATE=C sort -k1,1 -k2,2n",
            stdin=query_process.stdout,
            stdout=output,
            stderr=subprocess.PIPE,  # Capture stderr to detect errors
            universal_newlines=True,
            bufsize=10000,
            shell=True,
        )
        
        # Close query_process stdout so sort_process can detect EOF
        if query_process.stdout:
            query_process.stdout.close()

    out_process = sort_process if sorted else query_process

    try:
        yield out_process
    finally:
        # Check each process for errors with descriptive messages
        try:
            close_process(out_process)
        except subprocess.CalledProcessError as e:
            if sorted:
                # Check which process failed
                query_returncode = query_process.wait()
                filter_returncode = filter_process.wait()
                
                if filter_returncode != 0:
                    raise RuntimeError(
                        f"bcftools view failed (exit code {filter_returncode}). "
                        f"Command: {' '.join(filter_basecmd + [vcf_file])}"
                    ) from e
                elif query_returncode != 0:
                    stderr_output = query_process.stderr.read() if query_process.stderr else ""
                    raise RuntimeError(
                        f"bcftools query failed (exit code {query_returncode}). "
                        f"Command: bcftools query -f {chr_prefix + query_string}. "
                        f"Error: {stderr_output}"
                    ) from e
                else:
                    raise RuntimeError(
                        f"sort failed (exit code {e.returncode}). "
                        f"Command: LC_COLLATE=C sort -k1,1 -k2,2n"
                    ) from e
            else:
                # query_process is the out_process
                filter_returncode = filter_process.wait()
                
                if filter_returncode != 0:
                    raise RuntimeError(
                        f"bcftools view failed (exit code {filter_returncode}). "
                        f"Command: {' '.join(filter_basecmd + [vcf_file])}"
                    ) from e
                else:
                    stderr_output = query_process.stderr.read() if query_process.stderr else ""
                    raise RuntimeError(
                        f"bcftools query failed (exit code {e.returncode}). "
                        f"Command: bcftools query -f {chr_prefix + query_string}. "
                        f"Error: {stderr_output}"
                    ) from e
        
        # Ensure all processes are terminated
        processes_to_check = [filter_process, query_process]
        if sort_process is not None:
            processes_to_check.append(sort_process)
            
        for proc in processes_to_check:
            if proc.poll() is None:  # Process still running
                proc.terminate()
                proc.wait()


def transfer_annotations_to_vcf(
    annotations_df, *, vcf_file, description, output=subprocess.PIPE, chr_prefix=""
):

    annotations_df = annotations_df.copy()

    assert (
        "CHROM" in annotations_df.columns
    ), 'Annotations must have a column named "CHROM".'
    assert (
        "POS" in annotations_df.columns
    ), 'Annotations must have a column named "POS".'

    annotations_df["CHROM"] = annotations_df.CHROM.str.removeprefix(chr_prefix)
    annotations_df["POS"] = (
        annotations_df.POS + 1
    )  # switch to 1-based from 0-based indexing
    annotations_df = annotations_df.sort_values(["CHROM", "POS"])

    transfer_columns = ",".join(
        ["CHROM", "POS"]
        + ["INFO/" + c for c in annotations_df.columns if not c in ["CHROM", "POS"]]
    )

    with tempfile.NamedTemporaryFile() as header, tempfile.NamedTemporaryFile(
        delete=False
    ) as dataframe:

        with open(header.name, "w") as f:
            for col in annotations_df.columns:

                if col in ["CHROM", "POS"]:
                    continue

                dtype = str(annotations_df[col].dtype)
                if dtype.startswith("int"):
                    dtype = "Integer"
                elif dtype.startswith("float"):
                    dtype = "Float"
                else:
                    dtype = "String"

                print(
                    f'##INFO=<ID={col},Number=1,Type={dtype},Description="{description.setdefault(col, col)}">',
                    file=f,
                    sep="\n",
                )

            annotations_df.to_csv(dataframe.name, index=None, sep="\t", header=None)

        try:
            subprocess.check_output(["bgzip", "-f", dataframe.name])
            subprocess.check_output(
                ["tabix", "-s1", "-b2", "-e2", "-f", dataframe.name + ".gz"]
            )

            subprocess.check_call(
                [
                    "bcftools",
                    "annotate",
                    "-a",
                    dataframe.name + ".gz",
                    "-h",
                    header.name,
                    "-c",
                    transfer_columns,
                    vcf_file,
                ],
                stdout=output,
                universal_newlines=True,
                stderr=sys.stderr,
            )
        finally:
            os.remove(dataframe.name + ".gz")
            os.remove(dataframe.name + ".gz.tbi")


def get_marginal_mutation_rate(
    genome_file,
    *vcf_files,
    output=sys.stdout,
    chr_prefix="",
    pass_only=True,
):

    query_str = f"{chr_prefix}%CHROM\t%POS0\t%POS\n"

    with tempfile.NamedTemporaryFile() as coverage_file, tempfile.NamedTemporaryFile() as regions_file:

        with open(regions_file.name, "w") as f:
            subprocess.check_call(
                [
                    "bedtools",
                    "makewindows",
                    "-g",
                    genome_file,
                    "-w",
                    "50000",
                ],
                stdout=f,
            )

        with tempfile.TemporaryDirectory() as tempdir:

            for vcf_file in tqdm.tqdm(vcf_files, desc="Filtering VCFs", ncols=100):
                with open(os.path.join(tempdir, os.path.basename(vcf_file)), "w") as f:
                    with stream_passed_SNVs(
                        vcf_file, query_str, pass_only=pass_only, output=f
                    ) as st:
                        st.communicate()

            processed_vcfs = [os.path.join(tempdir, v) for v in os.listdir(tempdir)]

            logger.info("Computing coverage statistics...")
            with open(coverage_file.name, "w") as f:
                subprocess.check_call(
                    [
                        "bedtools",
                        "coverage",
                        "-a",
                        regions_file.name,
                        "-b",
                        *processed_vcfs,
                        "-sorted",
                        "-split",
                        "-counts",
                    ],
                    stdout=f,
                    universal_newlines=True,
                )

        logger.info("Calculating total mutations...")

        total_mutations, num_regions = (
            subprocess.check_output(
                ["awk", "{sum += $NF} END {print sum, NR}", coverage_file.name]
            )
            .decode("utf-8")
            .strip()
            .split(" ")
        )

        total_mutations = int(total_mutations)
        num_regions = int(num_regions)
        total_mutations += num_regions

        logger.info(
            f"Piled up {total_mutations} total mutations across {num_regions} regions."
        )
        logger.info("Writing output ...")

        subprocess.check_call(
            [
                "awk",
                "-v",
                "OFS=\t",
                f"{{print $1,$2,$3,($4+1)/{total_mutations}}}",
                coverage_file.name,
            ],
            stdout=output,
        )


def _get_local_mutation_rate(
    mutation_rate_bedgraph,
    query_fn,
    output,
    smoothing_distance=25000,
):
    """
    bedfile : a bed file of genomic regions, and the score column should be the *normalized* mutation rate
    """

    # 1. get SNP positions that passed QC
    with query_fn(
        "%CHROM\t%POS0\t%POS\n", sorted=True
    ) as query_process, tempfile.NamedTemporaryFile() as temp_file:

        # 2. define a window around each SNV
        slop_process = subprocess.Popen(
            [
                "awk",
                "-v",
                "OFS=\t",
                f"{{start=($2-{smoothing_distance} > 0) ? $2-{smoothing_distance} : 0 ; print $1,start,$2+{smoothing_distance},$1,$2,$3}}",
            ],
            stdin=query_process.stdout,
            stdout=subprocess.PIPE,
        )

        # 3. intersect the window with the mutation rate bedgraph
        intersect_process = subprocess.Popen(
            [
                "bedtools",
                "intersect",
                "-a",
                "-",
                "-b",
                mutation_rate_bedgraph,
                "-sorted",
                "-wa",
                "-wb",
            ],
            stdin=slop_process.stdout,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            bufsize=10000,
        )

        # 4. get the mutation location (col 1-3),
        #   the local relative mutation rate (col 4),
        #   and the size of the local interval (col 5)
        awk_process = subprocess.Popen(
            ["awk", "-v", "OFS=\t", "{print $4,$5,$6,$10,$9-$8}"],
            stdin=intersect_process.stdout,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

        # 5. For each unique mutation location,
        #   sum the local mutation rate and the size of the intersection interval
        #   Save this to a temporary file
        with open(temp_file.name, "w") as f:
            subprocess.check_call(
                [
                    "bedtools",
                    "groupby",
                    "-g",
                    "1,2,3",
                    "-c",
                    "4,5",
                    "-o",
                    "sum,sum",
                ],
                stdin=awk_process.stdout,
                stdout=f,
                universal_newlines=True,
                bufsize=10000,
            )

        total_mutations = int(
            subprocess.check_output(
                f"cat {temp_file.name} | wc -l",
                shell=True,
            )
            .decode("utf-8")
            .strip()
        )

        # 6. Divide the sum of the local mutation rate by the sum of the size of the intersection interval
        #   to get the average local mutation rate, them multiply by the total number of mutations
        #   in the sample to get the poisson process parameter
        write_process = subprocess.check_call(
            [
                "awk",
                "-v",
                "OFS=\t",
                f"{{print $1,$2,$3,$4/$5*{total_mutations}}}",
                temp_file.name,
            ],
            stdout=output,
            universal_newlines=True,
        )

    return write_process


def _get_rainfall_statistic(
    mutation_rate_bedgraph,
    query_fn,
    output,
    smoothing_distance=15000,
):
    with tempfile.NamedTemporaryFile("w") as snp_file:

        # 1. get local mutation rate parameter about each mutation,
        #   This parameter gives the average number of mutations in a window of size `smoothing_distance`
        _get_local_mutation_rate(
            mutation_rate_bedgraph,
            query_fn,
            snp_file,
            smoothing_distance=smoothing_distance,
        )
        snp_file.flush()

        # 2. Compute the distance to the nearest mutation for each mutation
        closest_process = subprocess.Popen(
            [
                "bedtools",
                "closest",
                "-a",
                snp_file.name,
                "-b",
                snp_file.name,
                "-io",
                "-d",
            ],
            stdout=subprocess.PIPE,
            universal_newlines=True,
            bufsize=10000,
        )

        # 3. Select columns 1-3, 8, 9
        #   1-3: mutation location
        #   8: local mutation rate
        #   9: distance to nearest mutation
        subprocess.check_call(
            ["cut", "-f1-3,8,9"],
            stdin=closest_process.stdout,
            stdout=output,
            universal_newlines=True,
        )


def _cluster_mutations(
    mutations_df,
    alpha=0.005,
):
    from scipy.stats import expon
    import numpy as np

    def get_mutclusters(df):

        df = df.sort_values(["CHROM", "POS"])
        # if any of these clauses are true, then we start a new cluster
        # 1. we are on a new chromosome
        # 2. the distance to the previous mutation is greater than the critical distance
        # 4. the allele frequency is different - this means the mutations occured in different cells or at different times
        ## TODO - clean up problem where a cluster is split by a variant with a different VAF? ##

        return (
            (df.CHROM.shift(1) != df.CHROM)
            | (
                df.POS - df.POS.shift(1)
                > np.minimum(10000, df.criticalDistance.shift(1))
            )
            # | ~vaf_is_similar \
        ).cumsum()

    mutations_df["criticalDistance"] = expon.ppf(
        alpha, scale=1 / mutations_df.localMutationRate
    )
    clusters = get_mutclusters(mutations_df).rename("cluster")
    mutations_df = mutations_df.join(clusters, how="left")
    cluster_size = mutations_df.cluster.value_counts().rename("clusterSize")
    mutations_df["negLog10interMutationDistanceRatio"] = -np.log10(
        mutations_df.rainfallDistance / mutations_df.criticalDistance
    )

    mutations_df = mutations_df.set_index("cluster").join(cluster_size)

    mutations_df.index.name = "clusterID"

    return mutations_df.reset_index()[
        [
            "CHROM",
            "POS",
            "negLog10interMutationDistanceRatio",
            "clusterSize",
            "clusterID",
        ]
    ]


def cluster_vcf(
    *,
    mutation_rate_bedgraph,
    query_fn,
    vcf_file,
    output=sys.stdout,
    chr_prefix="",
    smoothing_distance=25000,
    alpha=0.005,
    AF_tol=0.1,
    use_mutation_type=False,
):
    """
    A "cluster" of mutations should be
    1. Contiguously statistically-significantly close to each other.
    2. Of the same type (e.g. C>A) - nope
    3. Of the same allele frequency - nope
    """
    import pandas as pd

    # num_samples = int( subprocess.check_output(f'bcftools query -l {vcf_file} | wc -l | cut -f1', shell=True)\
    #                  .decode('utf-8').strip() )

    # if num_samples > 1:
    #    assert not sample is None, 'The VCF file contains multiple samples. Please specify a sample to analyze.'

    with tempfile.NamedTemporaryFile("w") as rainfall_file:

        # 2. Calculate rainfall statistics
        #    from the VCF file
        _get_rainfall_statistic(
            mutation_rate_bedgraph,
            query_fn,
            rainfall_file,
            smoothing_distance=smoothing_distance,
        )
        rainfall_file.flush()
        mutations_df = pd.read_csv(rainfall_file.name, sep="\t", header=None)

    mutations_df.columns = [
        "CHROM",
        "POS",
        "POS1",
        "localMutationRate",
        "rainfallDistance",
    ]

    mutations_df = mutations_df[mutations_df.localMutationRate != "."]
    mutations_df["localMutationRate"] = mutations_df.localMutationRate.astype(float)

    mutations_df = mutations_df[mutations_df.localMutationRate > 0]

    mutations_df = _cluster_mutations(
        mutations_df,
        alpha=alpha,
    )

    transfer_annotations_to_vcf(
        mutations_df,
        vcf_file=vcf_file,
        description={
            "negLog10interMutationDistanceRatio": "The negative log10 of the ratio of the inter-mutation distance to the critical distance",
            "clusterSize": "The number of mutations in the cluster",
            "clusterID": "The mutation's cluster ID",
        },
        output=output,
        chr_prefix=chr_prefix,
    )
