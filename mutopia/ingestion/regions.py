
import os
import subprocess
import tempfile
import sys
from numpy import quantile
from collections import Counter
import typing
from functools import partial
from dataclasses import dataclass
from ..genome_utils.fancy_iterators import *
from ..utils import logger, str_wrapped_list

def _make_fixed_size_windows(
    *, 
    genome_file, 
    window_size,
    blacklist_file=None,
    output=sys.stdout
):
    
    process_kw = dict(
        universal_newlines=True,
        bufsize=10000,
    )

    makewindows_process = subprocess.Popen(
        ['bedtools', 'makewindows', '-g', genome_file, '-w', str(window_size)],
        stdout = subprocess.PIPE,
        **process_kw,
    )

    sort_process = subprocess.Popen(
        ['sort', '-k1,1', '-k2,2n'],
        stdin = makewindows_process.stdout,
        stdout = subprocess.PIPE,
        **process_kw,
    )

    if blacklist_file is not None:
        subract_process = subprocess.Popen(
            ['bedtools', 'intersect', '-a', '-', '-b', blacklist_file, '-v'],
            stdin = sort_process.stdout,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            bufsize=10000,
        )
        sort_process = subract_process

    add_id_process = subprocess.Popen(
        ['awk','-v','OFS=\t', '{print $0,NR-1,"0","+",$2,$3,"0,0,0","1",$3-$2,"0"}'],
        stdin = sort_process.stdout,
        stdout = output,
        **process_kw,
    )

    add_id_process.wait()


def stream_bedfile(bedfile):
    with open(bedfile, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            cols = line.strip().split('\t')
            if len(cols) < 3:
                raise ValueError(f'Bedfile {bedfile} must have at least 3 columns')
            feature = '1' if len(cols) == 3 else cols[3]
            chrom, start, end = cols[:3]
            start = int(start); end = int(end)
            yield chrom, start, end, feature


@dataclass
class Endpoint:
    chrom : str
    start : int
    end : int
    track_id : str
    feature : typing.Any
    is_start : bool


@dataclass
class Segment:
    chrom : str
    start : int
    end : int
    feature_combination : typing.Any

    def __len__(self):
        return self.end - self.start


def _get_endpoints(*bedfiles):

    def _iter_endpoints_bedfile(bedfile, track_id):
        # okay the problem is that this is not current sorted ...
        for chrom, start, end, feature in stream_bedfile(bedfile):
            yield Endpoint(chrom, start, end, track_id, feature, True)
            yield Endpoint(chrom, end, end, track_id, feature, False)


    '''def ordered_endpoints(bedfile, track_id):
        return map(lambda x : (x[0],x[1],x[3],x[4],x[5]),
            chain.from_iterable(
                buffered_aggregator(
                    _iter_endpoints_bedfile(bedfile, track_id),
                    has_lapsed = lambda x, y : x[0] != y[0] or ( (x[1] - (y[1] + y[2])) > 0 and x[-1]),
                    key = lambda x : (x[0], x[1]),
                    order_key = lambda x : (x[0], x[1]),
                )
            )
        )'''
    
    def order_endpoints(bedfile, track_id):
        return streaming_local_sort(
            _iter_endpoints_bedfile(bedfile, track_id),
            has_lapsed = lambda curr, buffval : \
                curr.chrom != buffval.chrom or \
                (curr.start - buffval.end) > 0,
            key = lambda x : (x.chrom, x.start),
        )
    
    #for bedfile in bedfiles:
    #    for a in sorted_iterator( ordered_endpoints(bedfile, 'yeah'), key = lambda x : (x[0],x[1]) ):
    #        pass
    #assert False

    endpoints = [
        order_endpoints(bedfile, os.path.basename(bedfile)) 
        for bedfile in bedfiles
    ]
    
    return interleave_streams(*endpoints, key = lambda x : (x.chrom, x.start))


def _endpoints_to_segments(endpoints): # change default min_windowsize 3 to 4

    active_features = Counter()
    feature_combination_ids = dict()
    prev_chrom = None; prev_pos = None

    for endpoint in endpoints:
        (chrom, pos, track_id, feature, is_start) = \
            endpoint.chrom, endpoint.start, endpoint.track_id, endpoint.feature, endpoint.is_start

        pos = int(pos)

        if prev_chrom is None:
            prev_chrom = chrom; prev_pos = pos
        elif chrom != prev_chrom:
            active_features = Counter()
            prev_chrom = chrom; prev_pos = pos
        elif pos > prev_pos and len(active_features) > 0:
            
            is_nested_start = active_features[(track_id, feature)] > 0 and is_start
            is_nested_end = active_features[(track_id, feature)] > 1 and not is_start

            if is_nested_start or is_nested_end:
                pass
            else:
                feature_combination = tuple(sorted(active_features.keys()))

                if not feature_combination in feature_combination_ids:
                    feature_combination_ids[feature_combination] = len(feature_combination_ids)

                yield Segment(chrom, prev_pos, pos, feature_combination_ids[feature_combination])

        if is_start:
            active_features[(track_id,feature)] += 1
        else:
            if active_features[(track_id,feature)] > 1:
                active_features[(track_id,feature)]-=1
            else:
                active_features.pop((track_id,feature))

        prev_pos = pos; prev_chrom = chrom


def format_bed12_record(region_id, segments):

    chrs = list(map(lambda s : s.chrom, segments))
    starts = list(map(lambda s : s.start, segments))
    ends = list(map(lambda s : s.end, segments))
            
    region_start=min(starts); region_end=max(ends)
    region_chr = chrs[0]
    num_blocks = len(segments)
    
    block_sizes = ','.join(map(lambda x : str(x[0] - x[1]), zip(ends, starts)))
    block_starts = ','.join(map(lambda s : str(s - region_start), starts))

    return (
        region_chr,    # chr
        region_start,  # start
        region_end,    # end
        region_id,     # name
        '0','+',       # value, strand
        region_start,  # thickStart
        region_start,  # thickEnd
        '0,0,0',       # itemRgb,
        num_blocks,    # blockCount
        block_sizes,   # blockSizes
        block_starts,  # blockStarts
    )


def make_regions(
    *bedfiles,
    genome_file,  
    blacklist_file, 
    output=sys.stdout, 
    window_size=10000,
    min_windowsize=25,
):
    allowed_chroms=[]
    with open(genome_file,'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            allowed_chroms.append(line.strip().split('\t')[0].strip())

    logger.info(f'Using chromosomes: {str_wrapped_list(allowed_chroms)}')
    
    window_sizes=[]
    n_windows_written=[0]

    def accumulate_windowsizes(segments):
        window_sizes.append(sum(map(len, segments)))
        n_windows_written[0]+=1
        if n_windows_written[0] % 25000 == 0:
            logger.info(f'Wrote {n_windows_written[0]} windows ...')
        return segments
    
    def group_has_lapsed(curr, group):
        return curr.chrom != group[0].chrom or (curr.start - group[0].start) > 2*window_size


    with tempfile.NamedTemporaryFile('w') as windows_file:

        logger.info(f'Making initial coarse-grained regions ...')
        _make_fixed_size_windows(
            genome_file=genome_file,
            window_size=window_size,
            output=windows_file,
        )
        windows_file.flush()
        
        logger.info(f'Building regions ...')
        # 1. get the endpoints from the bedfiles
        data = _get_endpoints(windows_file.name, *bedfiles)

        # 2. filter out the endpoints that are not on the allowed chromosomes
        data = filter(lambda x : x.chrom in allowed_chroms, data)
        
        # 3. convert the endpoints to segments
        data = _endpoints_to_segments(data)

        # 4. filter out the segments that are in the blacklist
        data = filter_intersection(
            data, 
            map(lambda v : Segment(*v[:3], None), stream_bedfile(blacklist_file)),
            blacklist=True,
            key = RegionOverlapComparitor
        )

        # 5. group the segments by feature combination
        data = streaming_groupby(
            data,
            groupby_key = lambda segment : segment.feature_combination,
            has_lapsed = group_has_lapsed
        )
        data = map(lambda x : x[1], data)

        # 6. double-check that things are still sorted after the groupby
        data = streaming_local_sort(
            data,
            key = lambda s : (s[0].chrom, s[0].start),
            has_lapsed = group_has_lapsed
        )

        def trace(x):
            print(x)
            return x
        
        #data = map(trace, data)
        # 7. filter out the segments that are too small
        data = filter(lambda segments : sum(map(len, segments)) > min_windowsize, data)

        # 8. collect some stats on the window sizes
        data = enumerate(map(accumulate_windowsizes, data))

        # 9. format the segments as bed12 records
        data = map(expand_args(format_bed12_record), data)

        # 10. write the bed12 records to the output
        data = map(expand_args(partial(print, sep='\t', file=output)), data)

        list(data) # force evaluation - returns nothing

    q=(0.1, 0.25, 0.5, 0.75, 0.9)
    windowsize_dist=quantile(window_sizes, q)
    print(
f'''Window size report
------------------
Num windows   | {len(window_sizes)}
Smallest      | {min(window_sizes)}
Largest       | {max(window_sizes)}    
''' + '\n'.join(('Quantile={: <4} | {}'.format(str(k),str(int(v))) for k,v in zip(q, windowsize_dist)))
    )
