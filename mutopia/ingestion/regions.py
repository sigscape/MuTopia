import subprocess
import tempfile
import sys
from collections import Counter
import typing
from functools import partial, wraps
from dataclasses import dataclass
from gzip import open as gzopen
from itertools import chain
from ..genome_utils.fancy_iterators import *
from ..utils import logger, str_wrapped_list


def _make_fixed_size_windows(
    *, genome_file, window_size, blacklist_file=None, output=sys.stdout
):

    process_kw = dict(
        universal_newlines=True,
        bufsize=10000,
    )

    makewindows_process = subprocess.Popen(
        ["bedtools", "makewindows", "-g", genome_file, "-w", str(window_size)],
        stdout=subprocess.PIPE,
        **process_kw,
    )

    sort_process = subprocess.Popen(
        ["sort", "-k1,1", "-k2,2n"],
        stdin=makewindows_process.stdout,
        stdout=subprocess.PIPE,
        **process_kw,
    )

    if blacklist_file is not None:
        subract_process = subprocess.Popen(
            ["bedtools", "intersect", "-a", "-", "-b", blacklist_file, "-v"],
            stdin=sort_process.stdout,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            bufsize=10000,
        )
        sort_process = subract_process

    add_id_process = subprocess.Popen(
        ["awk", "-v", "OFS=\t", '{print $0,NR-1,"0","+",$2,$3,"0,0,0","1",$3-$2,"0"}'],
        stdin=sort_process.stdout,
        stdout=output,
        **process_kw,
    )

    add_id_process.wait()


def stream_bedfile(bedfile):

    try:
        opener = open if not bedfile.endswith(".gz") else gzopen
        with opener(bedfile, "rt") as f:
            has_data_line = False
            for line in f:
                if line.startswith("#"):
                    continue
                # We saw a non-comment line
                has_data_line = True
                cols = line.strip().split("\t")
                if len(cols) < 3:
                    raise ValueError(f"Bedfile {bedfile} must have at least 3 columns")
                feature = "1" if len(cols) == 3 else cols[3]
                chrom, start, end = cols[:3]
                start = int(start)
                end = int(end)
                yield chrom, start, end, feature

            # If there were no non-comment lines, raise a clear error so callers
            # don't silently proceed with empty inputs.
            if not has_data_line:
                raise ValueError(f"Bedfile {bedfile} is empty (no data lines)")
    except Exception as e:
        raise ValueError(f"Error reading bedfile {bedfile}: {str(e)}") from e

def sorted_bedfile(bedfile):
    return sorted(list(stream_bedfile(bedfile)), key=lambda x: (x[0], x[1]))


@dataclass
class Region:
    chrom: str
    start: int
    end: int

    def __hash__(self):
        return hash((self.chrom, self.start, self.end))

    def __str__(self):
        return f"{self.chrom}:{self.start}-{self.end}"


@dataclass
class Endpoint:
    chrom: str
    start: int
    end: int
    track_id: str
    feature: typing.Any
    is_start: bool


@dataclass
class Segment:
    chrom: str
    start: int
    end: int
    parent_region: Region
    feature_combination: typing.Any
    active_features: typing.Any

    def __len__(self):
        return self.end - self.start

    def __str__(self):
        return f"{self.chrom}:{self.start}-{self.end}"


def _get_endpoints(
    *bedfiles,
    blacklist=None,
    base_regions=None,
    sort=False,
):
    key = lambda x: (x.chrom, x.start)

    def _iter_base_regions():
        for chrom, start, end, *_ in stream_bedfile(base_regions):
            yield Endpoint(
                chrom, start, end, "__base__", Region(chrom, start, end), True
            )
            yield Endpoint(
                chrom, end, end, "__base__", Region(chrom, start, end), False
            )

    def _iter_blacklist_regions():
        for chrom, start, end, i in stream_bedfile(blacklist):
            yield Endpoint(chrom, start, end, "__blacklist__", i, True)
            yield Endpoint(chrom, end, end, "__blacklist__", i, False)

    def _iter_endpoints_bedfile(bedfile, track_id):
        # okay the problem is that this is not current sorted ...
        for chrom, start, end, feature in (stream_bedfile if not sort else sorted_bedfile)(bedfile):
            yield Endpoint(chrom, start, end, track_id, feature, True)
            yield Endpoint(chrom, end, end, track_id, feature, False)

    def order_endpoints(endpoints):
        return sorted_iterator(
            streaming_local_sort(
                endpoints,
                has_lapsed=lambda curr, buffval: curr.chrom != buffval.chrom
                or (curr.start - buffval.end) > 0,
                key=key,
            ),
            key=key,
        )

    def wrap_error(msg):
        @wraps(order_endpoints)
        def _f(*args, **kwargs):
            try:
                return order_endpoints(*args, **kwargs)
            except Exception as e:
                raise ValueError(f"{msg}: {str(e)}") from e

        return _f

    endpoints = [
        wrap_error(f"Error raised when processing {bedfile}")(
            _iter_endpoints_bedfile(bedfile, feature_name),
        )
        for feature_name, bedfile in bedfiles
    ]

    if not blacklist is None:
        endpoints.append(
            wrap_error(f"Error raised when processing {blacklist}")(
                _iter_blacklist_regions()
            )
        )

    if not base_regions is None:
        endpoints.append(
            wrap_error(f"Error raised when processing {base_regions}")(
                _iter_base_regions()
            )
        )

    return interleave_streams(
        *endpoints,
        key=key,
    )


def _endpoints_to_segments(
    endpoints, has_base_regions=True
):  # change default min_windowsize 3 to 4

    active_features = Counter()
    feature_combination_ids = dict()
    prev_chrom = None
    prev_pos = None

    for endpoint in endpoints:
        (chrom, pos, track_id, feature, is_start) = (
            endpoint.chrom,
            endpoint.start,
            endpoint.track_id,
            endpoint.feature,
            endpoint.is_start,
        )

        pos = int(pos)

        if prev_chrom is None:
            prev_chrom = chrom
            prev_pos = pos
        elif chrom != prev_chrom:
            active_features = Counter()
            prev_chrom = chrom
            prev_pos = pos
        elif pos > prev_pos and len(active_features) > 0:

            is_nested_start = active_features[(track_id, feature)] > 0 and is_start
            is_nested_end = active_features[(track_id, feature)] > 1 and not is_start

            if is_nested_start or is_nested_end:
                pass
            else:
                feature_combination = tuple(
                    sorted(active_features.keys(), key=lambda x: (x[0], str(x[1])))
                )

                if not feature_combination in feature_combination_ids:
                    feature_combination_ids[feature_combination] = len(
                        feature_combination_ids
                    )

                base_region = next(
                    (f for t, f in active_features.keys() if t == "__base__"), None
                )

                if (not has_base_regions or not base_region is None) and not any(
                    t == "__blacklist__" for t, _ in active_features.keys()
                ):
                    yield Segment(
                        chrom,
                        prev_pos,
                        pos,
                        base_region,
                        feature_combination_ids[feature_combination],
                        feature_combination,
                    )

        if is_start:
            active_features[(track_id, feature)] += 1
        else:
            if active_features[(track_id, feature)] > 1:
                active_features[(track_id, feature)] -= 1
            else:
                active_features.pop((track_id, feature))

        prev_pos = pos
        prev_chrom = chrom


def format_bed12_record(region_id, segments):

    starts = list(map(lambda s: s.start, segments))
    ends = list(map(lambda s: s.end, segments))

    parent = segments[0].parent_region
    num_blocks = len(segments)

    thick_start = min(starts)
    thick_end = max(ends)

    block_sizes = ",".join(map(lambda x: str(x[0] - x[1]), zip(ends, starts)))
    block_starts = ",".join(map(lambda s: str(s - parent.start), starts))

    return (
        parent.chrom,  # chr
        parent.start,  # start
        parent.end,  # end
        region_id,  # name
        "0",
        "+",  # value, strand
        thick_start,  # thickStart
        thick_end,  # thickEnd
        "0,0,0",  # itemRgb,
        num_blocks,  # blockCount
        block_sizes,  # blockSizes
        block_starts,  # blockStarts
    )


def linearize_beds(
    *bedfiles,
    output=sys.stdout,
    max_region_size=25000,
):

    def chop_if_too_large(segment):

        if len(segment) / max_region_size < 1.5:
            yield segment

        else:
            n_cuts = len(segment) // max_region_size
            cut_size = len(segment) // n_cuts + 1

            for i in range(n_cuts):
                yield Segment(
                    segment.chrom,
                    segment.start + i * cut_size,
                    min(segment.end, segment.start + (i + 1) * cut_size),
                    segment.parent_region,
                    segment.feature_combination,
                    segment.active_features,
                )

    # 1. get the endpoints from the bedfiles
    data = _get_endpoints(*bedfiles)

    # 3. convert the endpoints to segments
    data = _endpoints_to_segments(data, has_base_regions=False)

    data = chain.from_iterable(map(chop_if_too_large, data))

    for segment in data:
        print(
            segment.chrom,
            segment.start,
            segment.end,
            "|".join(
                f"{track_id}:{feature}" for track_id, feature in segment.active_features
            ),
            sep="\t",
            file=output,
        )


def make_regions(
    *bedfiles,
    genome_file,
    blacklist_file,
    base_regions=None,
    output=sys.stdout,
    window_size=10000,
    min_windowsize=25,
    sort=False,
):
    allowed_chroms = []
    with open(genome_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            allowed_chroms.append(line.strip().split("\t")[0].strip())

    logger.info(f"Using chromosomes: {str_wrapped_list(allowed_chroms)}")

    window_sizes = []
    n_windows_written = [0]

    def accumulate_windowsizes(segments):
        window_sizes.append(sum(map(len, segments)))
        n_windows_written[0] += 1
        if n_windows_written[0] % 25000 == 0:
            logger.info(f"Wrote {n_windows_written[0]} windows ...")
        return segments

    def group_has_lapsed(curr, group):
        return (
            curr.chrom != group[0].chrom
            or (curr.start - group[0].start) > 2 * window_size
        )

    with tempfile.NamedTemporaryFile("w") as windows_file:

        if base_regions is None:
            logger.info(f"Making initial coarse-grained regions ...")
            _make_fixed_size_windows(
                genome_file=genome_file,
                window_size=window_size,
                output=windows_file,
            )
            windows_file.flush()
            base_regions = windows_file.name

        logger.info(f"Building regions ...")
        # 1. get the endpoints from the bedfiles
        data = _get_endpoints(
            *bedfiles, blacklist=blacklist_file, base_regions=base_regions,
            sort=sort,
        )

        # 2. filter out the endpoints that are not on the allowed chromosomes
        data = filter(lambda x: x.chrom in allowed_chroms, data)

        # 3. convert the endpoints to segments
        data = _endpoints_to_segments(data)

        # 5. group the segments by feature combination
        data = streaming_groupby(
            data,
            groupby_key=lambda segment: segment.feature_combination,
            has_lapsed=group_has_lapsed,
        )
        data = map(lambda x: x[1], data)

        data = streaming_local_sort(
            data,
            key=lambda s: (s[0].chrom, s[0].start),
            has_lapsed=lambda curr_group, buff_group: group_has_lapsed(
                curr_group[0], buff_group
            ),
        )

        # 6. double-check that things are still sorted after the groupby.
        data = sorted_iterator(data, key=lambda s: (s[0].chrom, s[0].start))

        # 6b. sort the segments within each group - just to be sure.
        data = map(partial(sorted, key=lambda s: s.start), data)

        # data = map(trace, data)
        # 7. filter out the groups that are too small
        data = filter(lambda segments: sum(map(len, segments)) > min_windowsize, data)

        # 8. collect some stats on the window sizes
        data = enumerate(map(accumulate_windowsizes, data))

        # 9. format the segments as bed12 records
        data = map(expand_args(format_bed12_record), data)

        data = streaming_local_sort(
            data,
            key=lambda x : (x[0], x[1], str(x[3])), # weird issues with lexical sorting
            has_lapsed=lambda curr, buffval: curr[0] != buffval[0] or (curr[1] > buffval[1])
        )

        # 10. write the bed12 records to the output
        data = map(expand_args(partial(print, sep="\t", file=output)), data)

        list(data)  # force evaluation - returns nothing

    q = (0.1, 0.25, 0.5, 0.75, 0.9)

    from numpy import quantile

    windowsize_dist = quantile(window_sizes, q)

    print(
        f"""Window size report
----------------------
Num windows   | {len(window_sizes)}
Smallest      | {min(window_sizes)}
Largest       | {max(window_sizes)}    
"""
        + "\n".join(
            (
                "Quantile={: <4} | {}".format(str(k), str(int(v)))
                for k, v in zip(q, windowsize_dist)
            )
        ),
        file=sys.stderr,
    )
