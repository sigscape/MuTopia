
import subprocess
from dataclasses import dataclass
import tempfile
from functools import partial as curry
from ._sbs_nucdata import *
from ._sbs_clustering import *
from ..utils import logger
from pyfaidx import Fasta


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
    def get_format_str(self, weight_col=None):
        return f'%CHROM\t%POS0\t%POS\t%REF\t%ALT\t%INFO/clusterSize\t' \
            + ('[%AF]\n' if weight_col is None else f'%INFO/{weight_col}\n')


class WeirdMutationError(Exception):
    pass

class BadWeightError(Exception):
    pass


CONTEXT_IDX_MAP = dict(zip(CONTEXTS, range(len(CONTEXTS))))
COMPLEMENT = {'A' : 'T','T' : 'A','G' : 'C','C' : 'G'}
def _revcomp(seq):
    return ''.join(reversed([COMPLEMENT[nuc] for nuc in seq]))


def _convert_to_bases(context, alt):
    configuration = 0
    if not context[1] in 'CT': 
        configuration = 1
        context, alt = _revcomp(context), COMPLEMENT[alt]

    return context, alt, configuration


def _convert_to_idx(context, alt):
    mutation = f'N->{alt}' if not alt in 'CT' else 'N->T/C'
    return CONTEXT_IDX_MAP[context], MUTATIONS.index(mutation)


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
    except TypeError as err:
        raise WeirdMutationError(
            f'Could not parse the following line: {line}'
        ) from err

    try:
        query.WEIGHT = float(query.WEIGHT)
    except ValueError as err:
        raise BadWeightError(
            f'Could not parse the following weight: {query.WEIGHT}'
        ) from err
    
    (context, alt, configuration_idx) = \
        _extract_mutation_info(
            query.CHROM, query.POS0, query.REF, query.ALT,
            fasta_object=fasta_object
        )
    
    (context_idx, mutation_idx) = _convert_to_idx(context, alt)
    
    adjusted_weight = query.WEIGHT if query.CLUSTER_SIZE <= 3 else query.WEIGHT/query.CLUSTER_SIZE

    return (
        (int(query.CLUSTER_SIZE > 3), configuration_idx, context_idx, mutation_idx, locus_idx),
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
):
    
    if weight_col is None and sample_name is None:
        num_samples=len(subprocess.check_output(f'bcftools query -l {vcf_file}', shell=True).decode().strip().split('\n'))
        assert num_samples <=1, "Multiple samples were found in this vcf file. You must specify a `sample_name` in order to extract mutation weights."

    if weight_col is None and sample_weight is None:
        logger.warning(
            'If no manually-defined mutation weight is provided, you should use the *inverse* tumor purity as the `sample_weight`.'
        )

    query_fn = curry(
        get_passed_SNVs,
        sample=sample_name,
        pass_only=pass_only,
        chr_prefix=chr_prefix,
    )

    with tempfile.NamedTemporaryFile() as processed_vcf, \
        tempfile.NamedTemporaryFile() as query_file:
        
        with open(processed_vcf.name, 'w') as out:
            cluster_vcf(
                mutation_rate_bedgraph=mutation_rate_file,
                query_fn=curry(query_fn, vcf_file),
                vcf_file=vcf_file,
                output=out,
                chr_prefix=chr_prefix,
            )
        
        with open('query.txt', 'w') as out:
            query_fn(
                processed_vcf.name, 
                QueryRecord.get_format_str(weight_col),
                output=out,
            ).communicate()

        intersect_process = subprocess.Popen(
            ['bedtools',
            'intersect',
            '-a', regions_file, 
            '-b', 'query.txt', 
            '-sorted',
            '-wa','-wb',
            '-split'],
            stdout=subprocess.PIPE,
            universal_newlines=True,
            bufsize=10000,
        )

        coords=[]
        weights=[]
        n_ingested=0
        n_weird=0

        with Fasta(fasta_file) as fa:

            while True:
                line = intersect_process.stdout.readline()
                if not line:
                    break
                
                try:
                    
                    c,w = _process_query_line(line, fa)
                    coords.append(c)
                    weights.append(w)
                    n_ingested+=1

                except (WeirdMutationError, BadWeightError) as err:
                    logger.warning(err)
                    n_weird+=1
                    continue

        intersect_process.communicate()

    if len(weights) == 0:
        raise ValueError(
            'No mutations were ingested!\n'
            'This could be due to a couple reasons:\n'
            '\t* You set `pass_only=True`, and none of the SNVs in the VCF file passed the filters, or this is not annotated.\n'
            '\t* The `chr_prefix` is wrong, e.g. the VCF file uses "1","2", etc., but the FASTA file uses "chr1","chr2", etc.\n'
            '\t* The weight column you specified is not present in the VCF, or it contains no numeric values.\n'
            '\t* The VCF file is empty, or does not contain any SNVs.\n'
            '\t* None of the SNVs intersect with any of the regions supplied.\n'
        )

    coords = np.array(coords).T
    weights = np.array(weights)

    if not sample_weight is None:
        weights*=sample_weight

    return (
        coords.astype(np.int32),
        weights,
    )
