from collections import defaultdict
import bisect
from functools import wraps

def expand_args(func):
    @wraps(func)
    def wrapper(x):
        return func(*x)
    return wrapper

def tack_on(val, iter):
    for x in iter:
        yield val, x


class RegionOverlapComparitor:
    '''
    Why does this class exist?

    This class is used to wrap a region and change the comparison behavior of the region.
    When wrapped, a region is equal to another region if the two regions overlap.
    '''

    def __init__(self, wraps):
        self._wraps = wraps

    @property
    def chrom(self):
        return self._wraps.chrom
    
    @property
    def start(self):
        return self._wraps.start
    
    @property
    def end(self):
        return self._wraps.end
    
    def __gt__(self, other):
        return self.chrom > other.chrom or \
            (self.chrom == other.chrom and self.start >= other.end)
    
    def __eq__(self, other):
        '''
        self  ____
        other   ____

        self      ____
        other  ____
        '''
        return self.chrom == other.chrom and \
                    (self.end >= other.start and self.start < other.end)
    
    def __ge__(self, other):
        return (self > other) or (self == other)
    


def sorted_iterator(iter, key = lambda x : x):
    # this is a generator that yields the next item in the iterator, 
    # but raises a ValueError if the items are not in order
    for curr in iter:

        try:
            _prev
        except UnboundLocalError:
            _prev = curr
        else:
            if not key(curr) >= key(_prev):
                raise ValueError('Items are out of order, {} should be greater than {}'\
                        .format(str(key(curr)), str(key(_prev)))
                    )
            
            _prev = curr

        yield curr


class PeekIterator:
    '''
    A class that wraps an iterator and allows you to peek at the next value,
    without advancing the iterator. It implements the Iterator trait
    but with the "peek" method added.
    '''
    def __init__(self, iterator):
        self._iterator = iterator
        self._depleted = False
        try:
            self._next = self._get_next()
        except StopIteration:
            self._depleted = True

    def _get_next(self):
        return next(self._iterator)
        
    def __next__(self):

        if self._depleted:
            raise StopIteration()            

        ret_value = self._next 
        
        try:
            self._next = self._get_next()
        except StopIteration:
            self._depleted = True
        
        return ret_value

    def peek(self):
        if self._depleted:
            raise StopIteration()

        return self._next

    def has_next(self):
        return not self._depleted

    def __eq__(self, other):
        return self.peek() == other.peek()

    def __gt__(self, other):
        return self.peek() > other.peek()
    

def interleave_streams(*iterators, key = lambda x : x):

    streams = [
        PeekIterator(sorted_iterator(stream, key=key))
        for stream in iterators
    ]

    while True:

        streams = [stream for stream in streams if stream.has_next()]
        
        if len(streams) == 0:
            break
        else:
            yield next(min(streams, key = lambda x : key(x.peek())))


def filter_intersection(
    filter_iter, 
    compare_iter, 
    blacklist=True,
    key=lambda x : x,
):

    iter1 = PeekIterator(filter_iter)
    iter2 = PeekIterator(compare_iter)

    while iter1.has_next():

        while iter2.has_next() and key(iter1.peek()) > key(iter2.peek()):
            next(iter2)

        if iter2.has_next():
            val = next(iter1) #advance iter1

            #overlaps and whitelist => yield
            #overlaps and blacklist => skip
            #no overlap and whitelist => skip
            #no overlap and blacklist => yield
            #this is just XOR operation
            if (key(val) == key(iter2.peek())) ^ blacklist:
                yield val

        elif blacklist:
            yield next(iter1) # if there are no more values in iter2, 
                              # yield the value from iter1
        else:
            break


def streaming_local_sort(
        iter,
        key = lambda x : x,
        has_lapsed = lambda curr, buffval : buffval < curr,
):
    
    iter = PeekIterator(iter)
    buffer = []

    while iter.has_next():

        # pop the next value from the iterator
        val = next(iter)
        
        # put the new value in the sorted buffer
        bisect.insort_left(buffer, val, key = key)

        # yield all the values that have lapsed
        while has_lapsed(val, buffer[0]):
            yield buffer.pop(0)

    # yield the remaining values
    for val in buffer:
        yield val


def streaming_groupby(
    iterator,
    groupby_key = lambda x : x,
    has_lapsed = lambda currval, group : False,
):
    groups = defaultdict(list)

    for val in iterator:

        group = groupby_key(val)

        its = list(groups.items())
        for key, buffer in its:
            if has_lapsed(val, buffer):
                yield key, groups.pop(key)

        groups[group].append(val)

    for group in groups:
        yield group, groups[group]