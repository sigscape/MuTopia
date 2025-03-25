
import subprocess
from dataclasses import dataclass
import tempfile
from functools import partial
from pyfaidx import Fasta
import numpy as np
from ._sbs_nucdata import *
from ._sbs_clustering import *
from ..utils import logger, stream_subprocess_output

@dataclass
class QueryRecord:
    CHROM: str
    POS0: int
    POS: int
    REF: str
    ALT: str
    CLUSTER_SIZE: int
    WEIGHT: float

    @classmethod
    def get_format_str(self, weight_col=None, cluster=True):
        return f'%CHROM\t%POS0\t%POS\t%REF\t%ALT\t' \
            + ('%INFO/clusterSize\t' if cluster else '1\t') \
            + ('1.\n' if weight_col is None else f'%INFO/{weight_col}\n')


class WeirdMutationError(Exception):
    pass

class BadWeightError(Exception):
    pass

CONTEXT_IDX_MAP = dict(zip(CONTEXTS, range(len(CONTEXTS))))


def _convert_to_bases(context, alt):
    configuration = 0
    if not context[1] in 'CT': 
        configuration = 1
        context, alt = revcomp(context), COMPLEMENT[alt]
    
    mut = 'N->' + (alt if alt in 'AG' else 'T/C')
    return context, mut, configuration

def _extract_mutation_info(
    chrom : str, 
    pos : int, 
    ref : str, 
    alt : str, 
    check_ref=True,
    *, 
    fasta_object
):

    start = pos - 1; end = pos + 2
    try:
        context = fasta_object[chrom][start : end].seq.upper()
    except KeyError as err:
        raise KeyError('\tChromosome {} found in VCF file is not in the FASTA reference file'\
            .format(chrom)) from err

    oldnuc = context[1]

    if not ref.upper() in 'ATCG' or not alt.upper() in 'ATCG':
        raise WeirdMutationError('\tWeird mutation found at {}:{} {}->{}'.format(
            chrom, str(pos), ref, alt
        ))

    if check_ref:
        if not oldnuc == ref:
            raise ValueError(
                'Looks like the vcf file was constructed with a different reference genome:'
                ' different ref allele found at {}:{}, found {} instead of {}'.format(
                    chrom, str(pos), oldnuc, ref 
                )
            )

    return _convert_to_bases(context, alt)


def _process_query_line(line, fasta_object):

    fields=line.strip().split('\t')
    locus_idx = int(fields[3])
    
    try:
        query = QueryRecord(*fields[12:])
        query.POS = int(query.POS)
        query.POS0 = int(query.POS0)
        query.CLUSTER_SIZE = int(query.CLUSTER_SIZE)
    except (TypeError, ValueError) as err:
        raise WeirdMutationError(
            f'Could not parse the following line: {line}'
        ) from err

    try:
        query.WEIGHT = min(float(query.WEIGHT), 1.)
    except ValueError as err:
        raise BadWeightError(
            f'Could not parse the following weight: {query.WEIGHT}'
        ) from err
    
    (context, alt, configuration_idx) = \
        _extract_mutation_info(
            query.CHROM, query.POS0, query.REF, query.ALT,
            fasta_object=fasta_object
        )

    try:
        context_idx = MUTOPIA_IDX_MAP[(context, alt)]
    except KeyError as err:
        raise WeirdMutationError(
            f'Could not parse the following context: {context}'
        ) from err

    adjusted_weight = query.WEIGHT if (query.CLUSTER_SIZE <= 3) else (query.WEIGHT/query.CLUSTER_SIZE)

    return (
        (query.CHROM, query.POS0),
        (int(query.CLUSTER_SIZE > 3), configuration_idx, context_idx, locus_idx),
        adjusted_weight
    )
    

def featurize_mutations(
    vcf_file, 
    regions_file, 
    fasta_file,
    chr_prefix = '', 
    weight_col = None, 
    mutation_rate_file=None,
    sample_weight=None,
    sample_name=None,
    pass_only=True,
    cluster=True,
    skip_sort=False,
):
    
    if weight_col is None and sample_name is None:
        num_samples=len(subprocess.check_output(f'bcftools query -l {vcf_file}', shell=True).decode().strip().split('\n'))
        assert num_samples <=1, "Multiple samples were found in this vcf file. You must specify a `sample_name` in order to extract mutation weights."

    if weight_col is None and sample_weight is None:
        logger.warning(
            'If no manually-defined mutation weight is provided, you should use the *inverse* tumor purity as the `sample_weight`.'
        )

    query_fn = partial(
        stream_passed_SNVs,
        sample=sample_name,
        pass_only=pass_only,
        chr_prefix=chr_prefix,
    )

    with tempfile.NamedTemporaryFile('w') as processed_vcf:
        
        if cluster:
            if mutation_rate_file is None:
                raise ValueError('You must provide a mutation rate bedgraph file in order to cluster mutations.')
            
            logger.info('Clustering mutations ...')

            cluster_vcf(
                mutation_rate_bedgraph=mutation_rate_file,
                query_fn=partial(query_fn, vcf_file, sorted=True),
                vcf_file=vcf_file,
                output=processed_vcf,
                chr_prefix=chr_prefix,
            )
            processed_vcf.flush()
            input_vcf = processed_vcf.name
        else:
            input_vcf = vcf_file

        logger.info('Parsing mutations ...')

        query_fn = partial(
            query_fn, 
            input_vcf, 
            QueryRecord.get_format_str(
                weight_col=weight_col,
                cluster=cluster
            ),
            sorted=not skip_sort,
        )
        
        with query_fn() as query, \
            Fasta(fasta_file) as fa:
            
            intersect_process = subprocess.Popen(
                [
                'bedtools',
                'intersect',
                '-a', regions_file, 
                '-b', '-', 
                '-sorted',
                '-wa',
                '-wb',
                '-split'
                ],
                stdin=query.stdout,
                stdout=subprocess.PIPE,
                universal_newlines=True,
                bufsize=10000,
            )

            coords=[]
            weights=[]
            mut_ids=[]
            n_ingested=0
            n_weird=0

            for line in stream_subprocess_output(intersect_process):
                
                try:
                    mut_id, coo, weight = _process_query_line(line, fa)
                    coords.append(coo)
                    weights.append(weight)
                    mut_ids.append(mut_id)
                    n_ingested+=1

                except (WeirdMutationError, BadWeightError) as err:
                    logger.warning(err)
                    n_weird+=1
                    continue

                if n_ingested % 5000 == 0:
                    logger.info(f'Ingested {n_ingested} mutations ...')

    if len(weights) == 0:
        raise ValueError(
            'No mutations were ingested! Check for error messages above.\n'
            'This could be due to a couple reasons:\n'
            '\t* You set `pass_only=True`, and none of the SNVs in the VCF file passed the filters, or this is not annotated.\n'
            '\t* The `chr_prefix` is wrong, e.g. the VCF file uses "1","2", etc., but the FASTA file uses "chr1","chr2", etc.\n'
            '\t* The weight column you specified is not present in the VCF, or it contains no numeric values.\n'
            '\t* The VCF file is empty, or does not contain any SNVs.\n'
            '\t* None of the SNVs intersect with any of the regions supplied.\n'
        )

    coords = np.array(coords).T.astype(np.int32)
    weights = np.array(weights).astype(np.float32)

    if not sample_weight is None:
        weights*=sample_weight

    logger.info(f'Successfully ingested {n_ingested} mutations, {n_weird} mutations could not be parsed.')

    return mut_ids, coords, weights
