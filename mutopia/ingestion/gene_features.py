def download_gtf(output):
    import urllib.request
    import os
    import gzip
    import shutil

    if not os.path.exists(output):
        # Download the GTF file to a temporary file
        temp_gz_file = output + ".gz"
        urllib.request.urlretrieve(
            "https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/release_1.3/MANE.GRCh38.v1.3.ensembl_genomic.gtf.gz",
            temp_gz_file,
        )

        # Unzip the file using gzip module
        with gzip.open(temp_gz_file, "rb") as f_in:
            with open(output, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove the temporary gzip file
        os.remove(temp_gz_file)


def make_annotation(gtf, output):

    import subprocess

    subprocess.check_call(
        (
            "gtensor utils query-gtf "
            f"-i {gtf} "
            "-type transcript -attr gene_id --zero-based "
            '-f "{chrom}\t{start}\t{end}\t{attributes[gene_name]}\t{attributes[gene_id]}\t{strand}\n" '
            f"| sort -k1,1 -k2,2n > {output}",
        ),
        shell=True,
    )


def join_quantitation(
    annotation_file,
    quantitation_file,
    join_on="gene_id",
):

    import pandas as pd

    # read and reformate the quantification file
    gene_annotation = pd.read_csv(
        annotation_file,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "gene_name", "gene_id", "strand"],
    )
    gene_annotation.columns = [x.strip("#") for x in gene_annotation.columns]
    gene_annotation["gene_id"] = gene_annotation["gene_id"].str.rsplit(".", n=1).str[0]
    gene_annotation["start"] = gene_annotation["start"].astype(int)
    gene_annotation["end"] = gene_annotation["end"].astype(int)

    bed_features = ["chrom", "start", "end"]
    gene_annotation.sort_values(bed_features, inplace=True)

    gene_annotation.drop_duplicates(
        subset="gene_id",
        keep="first",
        inplace=True,
    )

    quantifications = pd.read_csv(
        quantitation_file,
        sep=None,
        header=None,
        engine="python",
        names=[join_on, "expression_est"],
    )

    quantifications = quantifications.groupby(join_on).mean().reset_index()

    if join_on == "gene_id":
        quantifications[join_on] = quantifications[join_on].str.rsplit(".", n=1).str[0]

    # Calculate number of shared elements in the join_on columns
    shared_elements = set(quantifications[join_on]) & set(gene_annotation[join_on])
    shared_count = len(shared_elements)

    # Raise ValueError if there are no shared elements
    if shared_count == 0:
        raise ValueError(
            f"No matching {join_on} values found between gene annotation and quantification files."
        )

    gene_annotation = gene_annotation.merge(
        quantifications,
        on=join_on,
        how="left",
    )

    gene_annotation.fillna({"expression_est": 0.0}, inplace=True)

    gene_annotation = gene_annotation.sort_values(bed_features)
    # save the gene expression
    return gene_annotation[[*bed_features, join_on, "expression_est", "strand"]]
