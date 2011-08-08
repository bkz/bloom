from math import ceil, exp, log, pow

import murmur3
import bitarray

try:
    import redis
except ImportError:
    pass


##############################################################################
# Statistics.
##############################################################################

class Stats(list):
    def sample(self, x):
        self.append(x)

    def min(self):
        return min(self)

    def max(self):
        return max(self)

    def sum(self):
        return sum(self)

    def avg(self):
        return self.sum() / len(self)

    def median(self):
        self.sort()
        return self[len(self) / 2]

    def stddev(self):
        a = self.avg()
        s = sum([(x - a) ** 2 for x in self])
        return (s / (len(self) -1)) ** 0.5

    def percentile(self, p):
        self.sort()
        return self[int(len(self) * float(p) / 100.0)]


##############################################################################
# Bloom filter calculations.
##############################################################################

# Bloom Filters Math
# http://pages.cs.wisc.edu/~cao/papers/summary-cache/node8.html
#
# Almeida, P. S., Baquero, C., Preguica, N., and Hutchison, D. 2007. Scalable
# Bloom Filters. Inf. Process. Lett. 101, 6 (Mar. 2007), 255-261. DOI=
# http://dx.doi.org/10.1016/j.ipl.2006.10.007
#
# Kirsh, A., and Mitzenmacher, M. 2006. Less Hashing, Same Performance:
# Building a Better Bloom Filter. In Proceedings of the 14th Annual European
# Symposium on Algorithms (ESA), pp. 456-467.
#
# NOTE: Use the code below to generate and/or tweak the following tables.

# m/n |   k    |    p     |   k=1    |   k=2    |   k=3    |   k=4    |   k=5    |   k=6    |   k=7    |   k=8    |   k=9    |   k=10   |   k=11   |   k=12
#   2 |  1.386 | 0.382546 | 0.393469 | 0.399576 | 0.468862 |    -     |    -     |    -     |    -     |    -     |    -     |    -     |    -     |    -
#   3 |  2.079 | 0.236606 | 0.283469 | 0.236763 | 0.252580 | 0.294078 |    -     |    -     |    -     |    -     |    -     |    -     |    -     |    -
#   4 |  2.773 | 0.146342 | 0.221199 | 0.154818 | 0.146892 | 0.159661 |    -     |    -     |    -     |    -     |    -     |    -     |    -     |    -
#   5 |  3.466 | 0.090513 | 0.181269 | 0.108689 | 0.091849 | 0.091954 | 0.100925 |    -     |    -     |    -     |    -     |    -     |    -     |    -
#   6 |  4.159 | 0.055982 | 0.153518 | 0.080354 | 0.060916 | 0.056057 | 0.057781 | 0.063797 |    -     |    -     |    -     |    -     |    -     |    -
#   7 |  4.852 | 0.034625 | 0.133122 | 0.061764 | 0.042348 | 0.035899 | 0.034658 | 0.036379 |    -     |    -     |    -     |    -     |    -     |    -
#   8 |  5.545 | 0.021416 | 0.117503 | 0.048929 | 0.030579 | 0.023969 | 0.021679 | 0.021577 | 0.022930 |    -     |    -     |    -     |    -     |    -
#   9 |  6.238 | 0.013246 | 0.105161 | 0.039706 | 0.022778 | 0.016577 | 0.014070 | 0.013272 | 0.013489 | 0.014463 |    -     |    -     |    -     |    -
#  10 |  6.931 | 0.008193 | 0.095163 | 0.032859 | 0.017411 | 0.011813 | 0.009431 | 0.008436 | 0.008194 | 0.008455 |    -     |    -     |    -     |    -
#  11 |  7.625 | 0.005067 | 0.086899 | 0.027638 | 0.013601 | 0.008637 | 0.006502 | 0.005522 | 0.005126 | 0.005086 | 0.005310 |    -     |    -     |    -
#  12 |  8.318 | 0.003134 | 0.079956 | 0.023568 | 0.010823 | 0.006457 | 0.004595 | 0.003711 | 0.003294 | 0.003142 | 0.003170 | 0.003339 |    -     |    -
#  13 |  9.011 | 0.001938 | 0.074039 | 0.020334 | 0.008752 | 0.004921 | 0.003318 | 0.002553 | 0.002169 | 0.001990 | 0.001938 | 0.001980 | 0.002101 |    -
#  14 |  9.704 | 0.001199 | 0.068937 | 0.017721 | 0.007176 | 0.003815 | 0.002443 | 0.001793 | 0.001460 | 0.001289 | 0.001213 | 0.001201 | 0.001240 |    -
#  15 | 10.397 | 0.000742 | 0.064493 | 0.015582 | 0.005956 | 0.003002 | 0.001830 | 0.001284 | 0.001003 | 0.000852 | 0.000775 | 0.000744 | 0.000747 | 0.000778
#  16 | 11.090 | 0.000459 | 0.060587 | 0.013807 | 0.004998 | 0.002394 | 0.001392 | 0.000935 | 0.000702 | 0.000574 | 0.000505 | 0.000470 | 0.000459 | 0.000466
#  17 | 11.784 | 0.000284 | 0.057127 | 0.012319 | 0.004234 | 0.001932 | 0.001074 | 0.000692 | 0.000499 | 0.000394 | 0.000335 | 0.000302 | 0.000287 | 0.000284
#  18 | 12.477 | 0.000175 | 0.054041 | 0.011059 | 0.003618 | 0.001577 | 0.000839 | 0.000519 | 0.000360 | 0.000275 | 0.000226 | 0.000198 | 0.000183 | 0.000176
#  19 | 13.170 | 0.000109 | 0.051271 | 0.009982 | 0.003116 | 0.001299 | 0.000663 | 0.000394 | 0.000264 | 0.000194 | 0.000155 | 0.000132 | 0.000118 | 0.000111
#  20 | 13.863 | 0.000067 | 0.048771 | 0.009056 | 0.002703 | 0.001080 | 0.000530 | 0.000303 | 0.000196 | 0.000140 | 0.000108 | 0.000089 | 0.000078 | 0.000071
#  21 | 14.556 | 0.000042 | 0.046503 | 0.008253 | 0.002359 | 0.000905 | 0.000427 | 0.000236 | 0.000147 | 0.000101 | 0.000076 | 0.000061 | 0.000052 | 0.000046
#  22 | 15.249 | 0.000026 | 0.044437 | 0.007551 | 0.002071 | 0.000764 | 0.000347 | 0.000185 | 0.000112 | 0.000075 | 0.000054 | 0.000042 | 0.000035 | 0.000030
#  23 | 15.942 | 0.000016 | 0.042547 | 0.006936 | 0.001829 | 0.000649 | 0.000285 | 0.000147 | 0.000086 | 0.000055 | 0.000039 | 0.000030 | 0.000024 | 0.000020
#  24 | 16.636 | 0.000010 | 0.040811 | 0.006393 | 0.001622 | 0.000555 | 0.000235 | 0.000117 | 0.000066 | 0.000042 | 0.000029 | 0.000021 | 0.000017 | 0.000014
#
# Optimal bloom filter for capacity and max false positive rate combinations:
#
# |--------------|--------------|--------------|--------------|--------------|
# |   capacity   |    max p     |   mem (kB)   |   bits/key   |   hash/key   |
# |--------------|--------------|--------------|--------------|--------------|
# |        1000  |         0.1  |        0.59  |        4.79  |           4  |
# |        1000  |        0.05  |        0.76  |        6.24  |           5  |
# |        1000  |        0.01  |        1.17  |        9.59  |           7  |
# |        1000  |       0.005  |        1.35  |       11.03  |           8  |
# |        1000  |       0.001  |        1.76  |       14.38  |          10  |
# |        1000  |      0.0005  |        1.93  |       15.82  |          11  |
# |        1000  |      0.0001  |        2.34  |       19.17  |          14  |
# |--------------|--------------|--------------|--------------|--------------|
#
# Optimal bloom filter for a desired max false positive rate:
#
# |--------------|--------------|--------------|--------------|
# |    max p     |  expected p  |   bits/key   |   hash/key   |
# |--------------|--------------|--------------|--------------|
# |       0.100  |      0.0918  |           5  |           3  |
# |       0.050  |      0.0423  |           7  |           3  |
# |       0.010  |      0.0094  |          10  |           5  |
# |       0.005  |      0.0046  |          12  |           5  |
# |       0.001  |      0.0009  |          15  |           8  |
# |       0.001  |      0.0005  |          16  |          10  |
# |       0.000  |      0.0001  |          20  |          10  |
# |--------------|--------------|--------------|--------------|
#
# Bloom filter capacities for memory size +  max false positive combinations:
#
# |--------------|--------------|--------------|--------------|--------------|
# |    max p     |   mem (kB)   |   capacity   |   bits/key   |   hash/key   |
# |--------------|--------------|--------------|--------------|--------------|
# |         0.1  |           1  |        1709  |        4.79  |           4  |
# |        0.01  |           1  |         855  |        9.58  |           7  |
# |       0.001  |           1  |         570  |       14.37  |          10  |
# |         0.1  |           4  |        6837  |        4.79  |           4  |
# |        0.01  |           4  |        3419  |        9.58  |           7  |
# |       0.001  |           4  |        2279  |       14.38  |          10  |
# |         0.1  |          16  |       27349  |        4.79  |           4  |
# |        0.01  |          16  |       13675  |        9.58  |           7  |
# |       0.001  |          16  |        9116  |       14.38  |          10  |
# |         0.1  |          32  |       54698  |        4.79  |           4  |
# |        0.01  |          32  |       27349  |        9.59  |           7  |
# |       0.001  |          32  |       18233  |       14.38  |          10  |
# |         0.1  |          64  |      109397  |        4.79  |           4  |
# |        0.01  |          64  |       54698  |        9.59  |           7  |
# |       0.001  |          64  |       36466  |       14.38  |          10  |
# |         0.1  |         128  |      218794  |        4.79  |           4  |
# |        0.01  |         128  |      109397  |        9.59  |           7  |
# |       0.001  |         128  |       72931  |       14.38  |          10  |
# |         0.1  |         512  |      875175  |        4.79  |           4  |
# |        0.01  |         512  |      437588  |        9.59  |           7  |
# |       0.001  |         512  |      291725  |       14.38  |          10  |
# |         0.1  |        1024  |     1750351  |        4.79  |           4  |
# |        0.01  |        1024  |      875175  |        9.59  |           7  |
# |       0.001  |        1024  |      583450  |       14.38  |          10  |
# |--------------|--------------|--------------|--------------|--------------|

def make_bloomfilter_lut(limit_mn=24, limit_k=12):
    """
    Generate a lookup table for bloom filter calculations:
    [(mn=2, optimal p, optimal k, [(k1, p), (k2, p), ...]),
     (mn=3, optimal p, optimal k, [(k1, p), (k2, p), ...]),
     ...]
    """
    tab = []
    for mn in range(2, limit_mn+1):
        opt_k = log(2) * float(mn)
        opt_p = pow(1 - exp(-1 * opt_k * 1.0/mn), opt_k)
        max_k = int(ceil(opt_k))+2
        pset = []
        for k in range(1, limit_k+1):
            if k < max_k:
                p = pow(1 - exp(-1 * k * 1.0/mn), k)
                pset.append((k, p))
            else:
                break
        tab.append((mn, opt_p, opt_k, pset))
    return tab


def optimal_bloomfilter(n, p):
    """
    Calculate optimal m (vector bits) and k (hash count) for a bloom filter
    with capacity `n` and an expected positive failure rate of `p`. Returns a
    tuple (m, k).  Note: calculations are a direct translation from Wikipedia.
    """
    n, p = float(n), float(p)
    m = -1 * (n * log(p)) / pow(log(2), 2)
    k = (m / n) * log(2)
    return int(ceil(m)), int(ceil(k))


def optimal_bloomfilter_spacetime(max_p, lut=make_bloomfilter_lut()):
    """
    Calculate optimal mn (bits/key) and k (hash count) values for a bloom
    filter with expected `max_p` false positive rate which minimize space
    requirement and the amount hashing needed per key. Returns tuple (mn, k).
    """
    best_mn, best_p, best_k = (None, None, None)
    for (mn, opt_p, opt_k, pset) in lut:
        if opt_p > max_p:
            continue
        for (k, p) in pset:
            if p < max_p:
                if (not best_p) or (mn < best_mn and k < best_k):
                    best_mn, best_p, best_k = mn, p, k
        break
    return (best_mn, best_p, best_k)


def bloomfilter_capacity(m, p):
    """
    Calculate max capacity n for bloom filter with size `m` bits and expected
    `p` false positive rate. (Note: in practice this means that when n is
    reached about half the bits in m will be set and p will start go up).
    """
    n = m * (pow(log(2), 2) / abs(log(p)))
    k = log(1/p, 2)
    return (int(round(n)), int(ceil(k)))


def print_bloomfilter_lut(lut=make_bloomfilter_lut()):
    """
    Generate pretty looking tables for manual bloom filter calculations.
    """
    k_max = max([len(pset) for (mn, opt_p, opt_k, pset) in lut])
    print "m/n".center(3), "|", "k".center(6), "|", "p".center(8), "|", " | ".join([("k=%d" % k).center(8) for k in range(1, k_max+1)])
    for (mn, opt_p, opt_k, pset) in lut:
        print str(mn).rjust(3), "|", str("%2.3f" % opt_k).rjust(6), "|", str("%0.6f" % opt_p).rjust(8),
        for (k,p) in pset:
            print "|", str("%0.6f" % p).rjust(8),
        for n in range(k_max - len(pset)):
            print "|", "-".center(8),
        print
    print


def print_table(row_width, columns, rows):
    """
    Helper for generating pretty look tables, each column is max `row_width` in
    which we format N `columns` which is expected to be a list of tuple
    (header, fmt-string) and `rows`` which also a list of tuples with matching
    values.

    Example:
      print_table(8, [('n', '%d'), ('c', '%s')], [(1, 'a'), (2, 'b'), ...])

    """
    def write_separator():
        output = ""
        for (header, fmt) in columns:
            output += "|" + (row_width+4) * '-'
        return output + "|"
    def write_headers():
        output = ""
        for (header, fmt) in columns:
            output += "|  " + header.center(row_width) + "  "
        return output + "|"
    def write_rows():
        output = ""
        for (n, row) in enumerate(rows):
            for ((header, fmt), value) in zip(columns, row):
                output += "|  " + (fmt % value).rjust(row_width) + "  "
            output += "|"
            if n < len(rows) - 1:
                output += "\n"
        return output
    print write_separator()
    print write_headers()
    print write_separator()
    print write_rows()
    print write_separator()
    print


def print_bloomfilter_stats():
    """
    Generate pretty looking tables with bloom filter statistics data.
    """
    values = []
    for max_p in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
        n = 1000
        m, k = optimal_bloomfilter(n, max_p)
        values.append((n, max_p, float(m)/8/1024, float(m)/n, k))
    print "Optimal bloom filter for capacity and max false positive rate combinations:"
    print
    print_table(10, [("capacity", "%d"), ("max p", "%g"), ("mem (kB)", "%.2f"), ("bits/key", "%.2f"), ("hash/key", "%d")], values)

    values = []
    for max_p in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
        mn, p, k = optimal_bloomfilter_spacetime(max_p)
        values.append((max_p, p, mn, k))
    print "Optimal bloom filter for a desired max false positive rate:"
    print
    print_table(10, [("max p", "%0.3f"), ("expected p", "%0.4f"), ("bits/key", "%d"), ("hash/key", "%d")], values)

    values = []
    for kb in [1, 4, 16, 32, 64, 128, 512, 1024]:
        m = kb * 1024 * 8
        for max_p in [0.1, 0.01, 0.001]:
            n, k = bloomfilter_capacity(m, max_p)
            values.append((max_p, float(m)/8/1024, n, float(m)/n, k))
    print "Bloom filter capacities for memory size +  max false positive combinations:"
    print
    print_table(10, [("max p", "%g"), ("mem (kB)", "%g"), ("capacity", "%d"), ("bits/key", "%.2f"), ("hash/key", "%d")], values)


##############################################################################
# BloomFilter.
##############################################################################

class BloomFilter(object):
    """
    Uses a variant presented in the Scalable Bloom Filters papers which slices
    up m into k slices. For every k hash we operate on a slice which gives
    better distribution and results in predictable lookup patterns.

    For hashing the key into k slices we use a 32-bit murmur3 hash which is
    used to produce two seeds h1 and h2 which are then combined to produce the
    remaning hashes (as described in "Less Hashing, Same Performance: Building
    a Better Bloom Filter", see above).

    Note: depends on the bitarray module as storage.
    """
    def __init__(self, m, k):
        self.b = bitarray.bitarray(m)
        self.b.setall(0)
        self.m = m
        self.k = k
        self.bits_per_slice = self.m / self.k
        self.opcount = 0
    def set(self, key):
        self.opcount += 1
        offset = 0
        for bit in self.get_bits(key):
            self.b[offset + bit] = 1
            offset += self.bits_per_slice
    def get_bits(self, key):
        h1 = murmur3.hash32(key, 0)
        h2 = murmur3.hash32(key, h1)
        return [(h1 + i * h2)  % self.bits_per_slice for i in range(self.k)]
    def __contains__(self, key):
        self.opcount += 1
        offset = 0
        for bit in self.get_bits(key):
            if not self.b[offset + bit]:
                return False
            offset += self.bits_per_slice
        return True


##############################################################################
# RedisBloomFilter.
##############################################################################

class RedisBloomFilter(object):
    """
    Bloom filter which uses Redis as storage, otherwise similar in design to
    the bitarray backed filter above.
    """
    def __init__(self, redis, name, m, k):
        self.redis = redis
        self.name = name
        self.redis.setbit(self.name, m, 0) # Pre-allocate bloom filter bitvector.
        self.pipe = self.redis.pipeline()
        self.m = m
        self.k = k
        self.bits_per_slice = self.m / self.k
        self.opcount = 0
    def set(self, key):
        self.opcount += 1
        offset = 0
        for bit in self.get_bits(key):
            self.pipe.setbit(self.name, offset + bit, 1)
            offset += self.bits_per_slice
        self.pipe.execute()
    def get_bits(self, key):
        h1 = murmur3.hash32(key, 0)
        h2 = murmur3.hash32(key, h1)
        return [(h1 + i * h2)  % self.bits_per_slice for i in range(self.k)]
    def __contains__(self, key):
        self.opcount += 1
        offset = 0
        for bit in self.get_bits(key):
            self.pipe.getbit(self.name, offset + bit)
            offset += self.bits_per_slice
        return all(self.pipe.execute())


##############################################################################
# Unit-test.
##############################################################################

if __name__ == '__main__':
    print_bloomfilter_lut()
    print_bloomfilter_stats()

    n = 18232
    p = 0.001

    m, k = optimal_bloomfilter(n, p)
    assert m == 262133
    assert k == 10

    n, k = bloomfilter_capacity(m, p)
    assert n == 18232
    assert k == 10

    # r = redis.StrictRedis(host='localhost', port=6379, db=0)
    # bf = RedisBloomFilter(r, 'bloom', m, k)
    bf = BloomFilter(m, k)

    from random import sample
    from string import ascii_letters

    import time
    s = time.time()

    words = ["key" + str(i) for i in xrange(int(n*0.99))] # Target 99% capacity.
    for w in words:
        bf.set(w)
    for w in words:
        assert w in words

    trials = 30

    stats = Stats()
    for _ in range(trials):
        fp = sum("".join(sample(ascii_letters, 8))  in bf for i in xrange(n))
        stats.sample(float(fp) / n)
    print "Stats: false positive avg = %g, med = %g, stddev = %g, max = %g, 90th = %g" % (
        stats.avg(), stats.median(), stats.stddev(), stats.max(), stats.percentile(90))

    t = time.time() - s
    print "Stats: runtime = %.2fs, bitvector = %.3f%%, %.2f ops/s" % (
        t, 100.0 * float(bf.b.count()) / len(bf.b), bf.opcount / t)


##############################################################################
# The End.
##############################################################################
