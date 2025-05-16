import sys
import re

GFF_COLS = [
    "chrom",
    "source",
    "type",
    "start",
    "end",
    "score",
    "strand",
    "phase",
    "attributes",
]
GFF_KEYS = dict(zip(GFF_COLS, range(len(GFF_COLS))))


def parse_GTF(
    gff_file,
    delim="\t",
):

    def format_kv(k, v):
        return k, v.strip('"')

    for line in gff_file:
        if line.startswith("#"):
            continue

        record = dict(zip(GFF_COLS, line.strip().strip(";").split(delim)))

        attributes = record["attributes"]
        record["attributes"] = dict(
            [format_kv(*x.split(" ", 1)) for x in attributes.split("; ")]
        )  # split attributes into key-value pairs
        record["attributes"][
            "all"
        ] = attributes  # add back the original attributes string for reference with "attributes[all]"

        yield record


def parse_GFF(
    gff_file,
    delim="\t",
):
    for line in gff_file:
        if line.startswith("#"):
            continue

        record = dict(zip(GFF_COLS, line.strip().split(delim)))

        attributes = record["attributes"]
        record["attributes"] = dict(
            [x.split("=") for x in attributes.split(";")]
        )  # split attributes into key-value pairs
        record["attributes"][
            "all"
        ] = attributes  # add back the original attributes string for reference with "attributes[all]"

        yield record


def filter_gff(
    gff_parser,
    type_filter=None,
    attribute_key=None,
    attribute_values=(),
):

    if attribute_key and attribute_values:
        attribute_values = tuple(map(str, attribute_values))

    for record in gff_parser:

        if type_filter is None or record["type"] == type_filter:

            if attribute_key is None or (
                attribute_key in record["attributes"]
                and (
                    (attribute_values is None)
                    or len(attribute_values) == 0
                    or record["attributes"][attribute_key] in attribute_values
                )
            ):
                yield record


def format_records(
    gff_parser,
    format_str="{chrom}\t{start}\t{end}\t{attributes[ID]}\n",
    outfile=sys.stdout,
):

    accepted_records = 0
    for record in gff_parser:
        try:
            print(format_str.format(**record), end="", file=outfile)
            accepted_records += 1
        except KeyError as err:
            print(
                "ERROR: The following attribute was not found in the GFF record: "
                + str(err),
                file=sys.stderr,
            )
            sys.exit(1)

    if accepted_records == 0:
        print(
            "ERROR: No records were found matching the specified criteria",
            file=sys.stderr,
        )
        sys.exit(1)


def print_format_str(format_str, outfile=sys.stdout):
    contained_columns = re.findall(r"{(.*?)}", format_str)
    print(
        "#"
        + "\t".join(
            map(
                lambda x: x.strip().removeprefix("attributes[").strip("]"),
                contained_columns,
            )
        ),
        file=outfile,
    )


def query_gtf(
    input,
    output,
    type_filter=None,
    attribute_key=None,
    attribute_values=None,
    is_gff=False,
    header=False,
    format_str=None,
    as_regions=False,
    as_gtf=False,
):
    """Parse and filter GTF/GFF files."""
    # Determine format string based on options
    if format_str is None:
        if as_regions:
            format_str = "{chrom}:{start}-{end}\n"
        elif as_gtf:
            format_str = (
                "{chrom}\t{source}\t{type}\t{start}\t{end}\t{score}\t{strand}\t{phase}\t{attributes[all]}\n"
            )
        else:
            format_str = "{chrom}\t{start}\t{end}\t{attributes[ID]}\n"  # default format
    else:
        format_str = format_str.encode().decode('unicode-escape')
        
    # Choose parser based on file format
    parser_fn = (
        parse_GFF
        if is_gff or getattr(input, "name", "").lower().endswith(".gff")
        else parse_GTF
    )

    # Parse input file
    gff_parser = parser_fn(input)

    # Apply filters if specified
    if type_filter or attribute_key:
        gff_parser = filter_gff(
            gff_parser,
            type_filter=type_filter,
            attribute_key=attribute_key,
            attribute_values=attribute_values,
        )

    # Print header if requested
    if header:
        print_format_str(format_str, outfile=output)

    # Format and print records
    format_records(gff_parser, format_str=format_str, outfile=output)
