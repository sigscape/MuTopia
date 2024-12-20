#!/usr/bin/env python3

import subprocess
import tempfile
import os
import tqdm
import pandas as pd
import logging
from scipy.stats import expon
import numpy as np
import sys
from ..utils import logger


def get_passed_SNVs(
    vcf_file, 
    query_string, 
    output=subprocess.PIPE,
    filter_string=None,
    pass_only=True,
    sample=None,
    chr_prefix='',
):
    
    filter_basecmd = [
        'bcftools','view',
        '-v','snps'
    ]

    if pass_only:
        filter_basecmd += ['-f','PASS']
    
    if not sample is None:
        filter_basecmd += ['-s', sample]

    if not filter_string is None:
        filter_basecmd += ['-i', filter_string]

    filter_process = subprocess.Popen(
            filter_basecmd + [vcf_file],
            stdout = subprocess.PIPE,
            universal_newlines=True,
            bufsize=10000,
            stderr = sys.stderr,
        )

    query_process = subprocess.Popen(
        ['bcftools','query','-f', chr_prefix + query_string],
        stdin = filter_process.stdout,
        stdout = subprocess.PIPE,
        stderr = sys.stderr,
        universal_newlines=True,
        bufsize=10000,
    )

    # ahh so annoying - it would be nice to delegate sorting to the user
    # but it can be difficult to make sure the VCFs are in lexigraphical order.
    # Instead, I'll bite the bullet and sort here.
    sort_process = subprocess.Popen(
        ['sort','-k1,1','-k2,2n'],
        stdin=query_process.stdout,
        stdout=output,
        stderr = sys.stderr,
        universal_newlines=True,
        bufsize=10000,
    )

    return sort_process


def transfer_annotations_to_vcf(
        annotations_df,*,
        vcf_file,
        description, 
        output=subprocess.PIPE, 
        chr_prefix=''
    ):

    annotations_df = annotations_df.copy()

    assert 'CHROM' in annotations_df.columns, 'Annotations must have a column named "CHROM".'
    assert 'POS' in annotations_df.columns, 'Annotations must have a column named "POS".'

    annotations_df['CHROM'] = annotations_df.CHROM.str.removeprefix(chr_prefix)
    annotations_df['POS'] = annotations_df.POS + 1 #switch to 1-based from 0-based indexing
    annotations_df = annotations_df.sort_values(['CHROM','POS'])

    transfer_columns = ','.join(['CHROM','POS'] + ['INFO/' + c for c in annotations_df.columns if not c in ['CHROM','POS']])

    with tempfile.NamedTemporaryFile() as header, \
        tempfile.NamedTemporaryFile(delete=False) as dataframe:

        with open(header.name, 'w') as f:
            for col in annotations_df.columns:
                
                if col in ['CHROM','POS']:
                    continue
                
                dtype = str(annotations_df[col].dtype)
                if dtype.startswith('int'):
                    dtype = 'Integer'
                elif dtype.startswith('float'):
                    dtype = 'Float'
                else:
                    dtype = 'String'

                print(
                    f'##INFO=<ID={col},Number=1,Type={dtype},Description="{description.setdefault(col, col)}">',
                    file = f,
                    sep = '\n',
                )

            annotations_df.to_csv(dataframe.name, index = None, sep = '\t', header = None)

        try:    
            subprocess.check_output(['bgzip','-f',dataframe.name])
            subprocess.check_output(['tabix','-s1','-b2','-e2', '-f', dataframe.name + '.gz'])

            subprocess.check_call(
                ['bcftools','annotate',
                '-a',  dataframe.name + '.gz',
                '-h', header.name,
                '-c', transfer_columns,
                vcf_file,
                ],
                stdout = output,
                universal_newlines=True,
                stderr = sys.stderr,
            )
        finally:
            os.remove(dataframe.name + '.gz')
            os.remove(dataframe.name + '.gz.tbi')


'''def unstack_bed12_file(
    regions_file, 
    output,
):

    regions = read_windows(regions_file)

    segments = []
    for region in regions:
        
        total_region_length = sum([end-start for (_, start, end) in region.segments()])
        
        for chr, start, end in region.segments():
            segments.append((chr, start, end, region.score/total_region_length*(end-start)))
    
    segments = sorted(segments, key=lambda x: (x[0], x[1]))

    for chrom, start, end, score in segments:
        print(
            chrom, start, end, score, 
            sep='\t', 
            file=output
        )'''


def get_marginal_mutation_rate(
    genome_file, 
    *vcf_files,
    output=sys.stdout,
    chr_prefix='',
    pass_only=True,
):
    
    query_str = f'{chr_prefix}%CHROM\t%POS0\t%POS0\n'

    with tempfile.NamedTemporaryFile() as coverage_file, \
        tempfile.NamedTemporaryFile() as regions_file:

        with open(regions_file.name, 'w') as f:
            subprocess.check_call(
                [
                    'bedtools','makewindows',
                    '-g', genome_file,
                    '-w', '5000',
                ], 
                stdout=f
            )

        with tempfile.TemporaryDirectory() as tempdir:
            for vcf_file in tqdm.tqdm(vcf_files, desc='Filtering VCFs', ncols = 100):
                with open(os.path.join(tempdir, os.path.basename(vcf_file)), 'w') as f:
                        get_passed_SNVs(vcf_file, query_str, pass_only=pass_only, output=f).communicate()

            processed_vcfs = [os.path.join(tempdir, v) for v in os.listdir(tempdir)]

            logger.info('Computing coverage statistics...')
            with open(coverage_file.name, 'w') as f:
                subprocess.check_call(
                    ['bedtools','coverage',
                    '-a',regions_file.name,
                    '-b', *processed_vcfs,
                    '-sorted',
                    '-split',
                    '-counts',
                    ],
                    stdout=f,
                    universal_newlines=True,
                )

        logger.info('Calculating total mutations...')
        
        total_mutations, num_regions = \
            subprocess.check_output(
                ['awk','{sum += $NF} END {print sum, NR}', coverage_file.name]
            ).decode('utf-8').strip().split(' ')
        
        total_mutations=int(total_mutations); num_regions=int(num_regions)
        total_mutations+=num_regions

        logger.info(f'Piled up {total_mutations} total mutations across {num_regions} regions.')
        logger.info('Writing output ...')
        
        subprocess.check_call(
            ['awk','-v','OFS=\t', f'{{print $1,$2,$3,($4+1)/{total_mutations}}}', coverage_file.name],
            stdout = output,
        )

 
def _get_local_mutation_rate(
    mutation_rate_bedgraph, 
    query_fn,
    output, 
    smoothing_distance=15000,
):
    '''
    bedfile : a bed file of genomic regions, and the score column should be the *normalized* mutation rate
    '''

    #1. get SNP positions that passed QC
    query_process = query_fn('%CHROM\t%POS0\t%POS0\n')

    #2. define a window around each SNV
    slop_process = subprocess.Popen(
        ['awk','-v','OFS=\t', 
        f'{{start=($2-{smoothing_distance} > 0) ? $2-{smoothing_distance} : 0 ; print $1,start,$2+{smoothing_distance}, $1,$2,$3}}'],
        stdin = query_process.stdout,
        stdout = subprocess.PIPE,
    )

    #3. intersect the window with the mutation rate bedgraph
    intersect_process = subprocess.Popen(
        ['bedtools','intersect',
         '-a','-', 
         '-b', mutation_rate_bedgraph,
         '-sorted',
         '-wa','-wb',
        ],
        stdin = slop_process.stdout,
        stdout = subprocess.PIPE,
        universal_newlines=True,
        bufsize=10000,
    )

    #4. get the mutation location (col 1-3), 
    #   the local relative mutation rate (col 4), 
    #   and the size of the local interval (col 5)
    awk_process = subprocess.Popen(
        ['awk','-v','OFS=\t','{print $4,$5,$6,$10,$9-$8}'],
        stdin = intersect_process.stdout,
        stdout = subprocess.PIPE,
        universal_newlines=True,
    )

    with tempfile.NamedTemporaryFile() as temp_file:

        #5. For each unique mutation location, 
        #   sum the local mutation rate and the size of the intersection interval
        #   Save this to a temporary file
        with open(temp_file.name, 'w') as f:
            subprocess.check_call(
                ['bedtools','groupby',
                '-g','1,2,3','-c','4,5','-o','sum,sum',
                ],
                stdin = awk_process.stdout,
                stdout = f,
                universal_newlines=True,
                bufsize=10000,
            )

        total_mutations = int(
            subprocess.check_output(
                f'cat {temp_file.name} | wc -l',
                shell=True,
            ).decode('utf-8').strip()
        )

        #6. Divide the sum of the local mutation rate by the sum of the size of the intersection interval
        #   to get the average local mutation rate, them multiply by the total number of mutations
        #   in the sample to get the poisson process parameter
        awk_process = subprocess.check_call(
            ['awk','-v','OFS=\t', f'{{print $1,$2,$3,$4/$5*{total_mutations}}}', temp_file.name],
            stdout = output,
            universal_newlines=True,
        )

        return awk_process



def _get_rainfall_statistic(
    mutation_rate_bedgraph, 
    query_fn, 
    output, 
    smoothing_distance=15000,                       
):


    with tempfile.NamedTemporaryFile() as snp_file:
        
        #1. get local mutation rate parameter about each mutation,
        #   This parameter gives the average number of mutations in a window of size `smoothing_distance`
        with open(snp_file.name, 'w') as f:
            _get_local_mutation_rate(
                mutation_rate_bedgraph, 
                query_fn, 
                f, 
                smoothing_distance=smoothing_distance,
            )

        #2. Compute the distance to the nearest mutation for each mutation
        closest_process = subprocess.Popen(
            ['bedtools',
             'closest',
             '-a', snp_file.name,
             '-b', snp_file.name,
             '-io',
             '-d',
            ],
            stdout = subprocess.PIPE,
            universal_newlines=True,
            bufsize=10000,
        )

        #3. Select columns 1-3, 8, 9
        #   1-3: mutation location
        #   8: local mutation rate
        #   9: distance to nearest mutation
        subprocess.check_call(
            ['cut','-f1-3,8,9'],
            stdin = closest_process.stdout,
            stdout = output,
            universal_newlines=True,
        )


def _cluster_mutations(
    mutations_df, 
    alpha = 0.005, 
    AF_tol = 0.1,
    use_mutation_type=True,
):

    def get_mutclusters(df):
        
        df = df.sort_values(['CHROM','POS'])
        # if any of these clauses are true, then we start a new cluster
        # 1. we are on a new chromosome
        # 2. the distance to the previous mutation is greater than the critical distance
        # 4. the allele frequency is different - this means the mutations occured in different cells or at different times

        '''def similar_VAF(rd1, vrd1, rd2, vrd2):
            #return np.abs(vrd1/rd1 - vrd2/rd2) < AF_tol
            return chi2_contingency([[rd1-vrd1, vrd1], [rd2-vrd2, vrd2]])[1] > 0.01
        
        vaf_is_similar = np.array([
            similar_VAF(rd1, vrd1, rd2, vrd2)
            for rd1, vrd1, rd2, vrd2 in zip(df.readDepth, df.variantReadDepth, df.readDepth.shift(1), df.variantReadDepth.shift(1))
        ])'''
        ## TODO - clean up problem where a cluster is split by a variant with a different VAF? ##
        return (
            (df.CHROM.shift(1) != df.CHROM) | \
            (df.POS - df.POS.shift(1) > np.minimum(10000, df.criticalDistance.shift(1))) 
            #| ~vaf_is_similar \
        ).cumsum()


    mutation_type_map = {
        'G>A' : 'C>T',
        'G>T' : 'C>A',
        'G>C' : 'C>G',
        'A>G' : 'T>C',
        'A>T' : 'T>A',
        'A>C' : 'T>G',
    }

    mutations_df['mutationType'] = mutations_df.mutationType.apply(lambda x : mutation_type_map.setdefault(x,x))
    mutations_df['criticalDistance'] = expon.ppf(alpha, scale=1/mutations_df.localMutationRate)

    if use_mutation_type:
        clusters = mutations_df.groupby('mutationType').apply(get_mutclusters, include_groups=False)
        clusters = (clusters.index.get_level_values(0) + '_' + clusters.astype(str))\
            .droplevel(0)\
            .rename('cluster')
    else:
        clusters = get_mutclusters(mutations_df).rename('cluster')      

    mutations_df = mutations_df.join(clusters, how = 'left')
    cluster_size = mutations_df.cluster.value_counts().rename('clusterSize')
    mutations_df['negLog10interMutationDistanceRatio'] = -np.log10(mutations_df.rainfallDistance/mutations_df.criticalDistance)

    mutations_df = mutations_df\
        .set_index('cluster')\
        .join(cluster_size)
    
    mutations_df.index.name = 'cluster'

    return mutations_df.reset_index()\
        [['CHROM','POS','mutationType','negLog10interMutationDistanceRatio','clusterSize', 'cluster']]



def cluster_vcf(
    *,
    mutation_rate_bedgraph, 
    query_fn,
    vcf_file,
    output = sys.stdout,
    chr_prefix='',
    smoothing_distance=15000,
    alpha = 0.005,
    AF_tol = 0.1,
    use_mutation_type=False,
):
    '''
    A "cluster" of mutations should be 
    1. Contiguously statistically-significantly close to each other.
    2. Of the same type (e.g. C>A)
    3. Of the same allele frequency
    '''

    #num_samples = int( subprocess.check_output(f'bcftools query -l {vcf_file} | wc -l | cut -f1', shell=True)\
    #                  .decode('utf-8').strip() )

    #if num_samples > 1:
    #    assert not sample is None, 'The VCF file contains multiple samples. Please specify a sample to analyze.'


    with tempfile.NamedTemporaryFile() as rainfall_file, \
        tempfile.NamedTemporaryFile() as df:
        
        # 2. Calculate rainfall statistics
        #    from the VCF file
        with open(rainfall_file.name, 'w') as f:
            _get_rainfall_statistic(
                mutation_rate_bedgraph, 
                query_fn, 
                f, 
                smoothing_distance=smoothing_distance,
            )

        # 2. get the mutation type and allele frequency
        #    from the VCF file
        query_process = query_fn('%CHROM\t%POS0\t%POS0\t%REF>%ALT\n')
        
        # 3. intersect the rainfall statistics with the mutation type and allele frequency
        #    The output columns will be 1) chr 2) start 3) end 4) local mutation rate 5) rainfall distance
        #                               6) chr 7) start 8) end 9) mutation type 10) read depth 11) variant read depth
        with open(df.name, 'w') as f:
            subprocess.check_call(
                [
                'bedtools','intersect',
                '-a', rainfall_file.name,
                '-b', '-',
                '-sorted',
                '-wa','-wb',
                ], 
                stdin = query_process.stdout,
                stdout = f,
                universal_newlines=True,
                bufsize=10000,
            )

        mutations_df = pd.read_csv(df.name, sep='\t', header=None)\
                .iloc[:, [0,1,3,4,8]]
        

    mutations_df.columns = ['CHROM','POS','localMutationRate','rainfallDistance','mutationType']

    mutations_df = mutations_df[ mutations_df.localMutationRate != '.' ]
    mutations_df['localMutationRate'] = mutations_df.localMutationRate.astype(float)

    mutations_df = mutations_df[mutations_df.localMutationRate > 0]

    mutations_df = _cluster_mutations(
        mutations_df, 
        alpha=alpha, 
        AF_tol=AF_tol, 
        use_mutation_type=use_mutation_type
    )
        
    transfer_annotations_to_vcf(
        mutations_df,
        vcf_file=vcf_file,
        description = {
            'mutationType' : 'The type of mutation (e.g. C>A)',
            'negLog10interMutationDistanceRatio' : 'The negative log10 of the ratio of the inter-mutation distance to the critical distance',
            'clusterSize' : 'The number of mutations in the cluster',
            'cluster' : 'The mutation\'s cluster ID',
        },
        output=output,
        chr_prefix=chr_prefix,
    )

