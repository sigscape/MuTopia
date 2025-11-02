from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import starmap, chain
from numpy import array
from contextlib import contextmanager
from .fancy_iterators import streaming_local_sort, sorted_iterator


@contextmanager
def safe_read(filename):
    from gzip import open as gzopen

    with (
        gzopen(filename, "rt") if filename.endswith(".gz") else open(filename, "r")
    ) as f:
        yield f


def stream_sort_bed(data, buffer_len=1000):
    return streaming_local_sort(
        data,
        key=lambda x: (x[0], x[1]),
        has_lapsed=lambda curr, buffval: curr[0] != buffval[0]
        or curr[1] - buffval[1] > buffer_len,
    )


@dataclass
class BED12Record:
    chromosome: str
    start: int
    end: int
    name: str
    block_count: int
    score: int = 0
    strand: str = "+"
    thick_start: int = 0
    thick_end: int = 0
    item_rgb: str = "0,0,0"
    block_sizes: list[int] = field(
        default_factory=lambda: [
            0,
        ]
    )
    block_starts: list[int] = field(
        default_factory=lambda: [
            0,
        ]
    )

    def segments(self):
        for start, size in zip(self.block_starts, self.block_sizes):
            yield self.chromosome, self.start + start, self.start + start + size

    def __len__(self):
        return sum(self.block_sizes)

    def __str__(self):
        return "\t".join(
            map(
                str,
                [
                    self.chromosome,
                    self.start,
                    self.end,
                    self.name,
                    self.score,
                    self.strand,
                    self.thick_start,
                    self.thick_end,
                    self.item_rgb,
                    self.block_count,
                    ",".join(map(str, self.block_sizes)),
                    ",".join(map(str, self.block_starts)),
                ],
            )
        )


def parse_bed12_line(line) -> BED12Record:
    (
        chromosome,
        start,
        end,
        name,
        score,
        strand,
        thick_start,
        thick_end,
        item_rgb,
        block_count,
        block_sizes,
        block_starts,
    ) = line.strip().split("\t")
    block_sizes = list(map(int, block_sizes.split(",")))
    block_starts = list(map(int, block_starts.split(",")))

    return BED12Record(
        chromosome=chromosome,
        start=int(start),
        end=int(end),
        name=name,
        score=float(score),
        strand=strand,
        thick_start=int(thick_start),
        thick_end=int(thick_end),
        item_rgb=item_rgb,
        block_count=int(block_count),
        block_sizes=block_sizes,
        block_starts=block_starts,
    )


def stream_bed12(bed12_file, sep="\t") -> Iterable[BED12Record]:

    with safe_read(bed12_file) as f:

        for lineno, txt in enumerate(f):
            if not txt[0] == "#":
                line = txt.strip().split(sep)
                assert (
                    len(line) >= 12
                ), "Expected BED12 file type with at least 12 columns"
            try:
                yield parse_bed12_line(txt)
            except ValueError as err:
                raise ValueError(
                    "Could not ingest line {}: {}".format(lineno, txt)
                ) from err


def check_regions_file(regions_file):

    encountered_idx = defaultdict(lambda: False)

    with safe_read(regions_file) as f:

        for i, line in enumerate(f):

            if line.startswith("#"):
                continue

            cols = line.strip().split("\t")
            assert (
                len(cols) >= 12
            ), f"Expected 12 or more columns (in BED12 format) in {regions_file}, with the fourth column being an integer ID.\n"

            try:
                _, start, end, idx = cols[:4]
                idx = int(idx)
                start = int(start)
                end = int(end)
            except TypeError:
                raise TypeError(
                    f"Count not parse line {i} in {regions_file}: {line}.\n"
                    "Make sure the regions file is tab-delimited with columns chr<str>, start<int>, end<int>, idx<int>.\n"
                )

            encountered_idx[idx] = True

    assert (
        len(encountered_idx) > 0
    ), f"Expected regions file to have at least one region."

    largest_bin = max(encountered_idx.keys())
    assert all(
        [encountered_idx[i] for i in range(largest_bin + 1)]
    ), f"Expected regions file to have a contiguous set of indices from 0 to {largest_bin}.\n"


def unstack_regions(
    region_names,
    regions_file,
    values,
):

    assert len(region_names) == len(
        values
    ), "Expected region_names and values to have the same length."

    region_lookup = set(list(map(str, region_names)))

    data = stream_bed12(regions_file)
    data = zip(filter(lambda x: str(x.name) in region_lookup, data), values)
    data = starmap(
        lambda region, vals: [(segment, vals) for segment in region.segments()], data
    )
    data = chain.from_iterable(data)

    data = streaming_local_sort(
        data,
        key=lambda x: (x[0][0], x[0][1]),
        has_lapsed=lambda curr, buffval: curr[0][0] != buffval[0][0]
        or curr[0][1] - buffval[0][1] > 100000,
    )

    data = sorted_iterator(
        data,
        key=lambda x: (x[0][0], x[0][1]),
    )

    data = map(lambda x: (*x[0], x[1]), data)

    data = list(map(array, (zip(*data))))

    return data
