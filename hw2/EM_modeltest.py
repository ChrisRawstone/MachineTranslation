#!/usr/bin/env python
import optparse
import random
import sys
from collections import defaultdict
import IBM_dont_save

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float",
                     help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=1000, type="int",
                     help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)



from collections import defaultdict
from itertools   import product
from msgpack     import pack,unpack
from sys         import stdout

import math
import numpy as np
import time


class IBM:
    """Class containing the IBM2 Model"""

    @classmethod
    def random(cls, corpus):
        """Model initialized using a normalized random initialization on the corpus"""
        return cls.with_generator(
            corpus, lambda n: np.random.dirichlet(np.ones(n), size=1)[0])


    @classmethod
    def uniform(cls, corpus):
        """Model initialized using a normalized uniform initialization on the corpus"""
        return cls.with_generator(corpus, lambda n: [1 / float(n)] * n)


    @classmethod
    def with_generator(cls, corpus, g):
        """Initialize the model with normalized parameters using generator g"""

        start = time.time()
        lens = set()
        aligns = defaultdict(set)

        # "Compute all possible alignments..."
        for k, (f, e) in enumerate(corpus):

            if k % 1000 == 0:
                stdout.write("\rInit    %6.2f%%" % ((33 * k) / float(len(corpus))))
                stdout.flush()

            e = [None] + e
            lens.add((len(e), len(f) + 1))

            for (f, e) in product(f, e):
                aligns[e].add((f, e))

        # "Compute initial probabilities for each ali½nment..."
        k = 0
        t = dict()
        for e, aligns_to_e in aligns.iteritems():

            if k % 1000 == 0:
                stdout.write("\rInit    %6.2f%%" % (33 + ((33 * k) / float(len(aligns)))))
                stdout.flush()
            k += 1

            p_values = g(len(aligns_to_e))
            t.update(zip(aligns_to_e, p_values))

        # "Compute initial probabilities for each distortion..."
        q = dict()
        for k, (l, m) in enumerate(lens):

            if k % 1000 == 0:
                stdout.write("\rInit    %6.2f%%" % (66 + ((33 * k) / float(len(lens)))))
                stdout.flush()

            for i in range(1, m):
                p_values = g(l)
                for j in range(0, l):
                    q[(j, i, l, m)] = p_values[j]

        print("\rInit     100.00%% (Elapsed: %.2fs)" % (time.time() - start))
        return cls(defaultdict(float, t), defaultdict(float, q))


    @classmethod
    def load(cls, stream):
        """Load model from a pack file"""
        (t, q) = unpack(stream,use_list=False)
        return cls(defaultdict(float,t), defaultdict(float,q))


    def dump(self, stream):
        """Dump model to a pack file, warning this file could be several 100 MBs large"""
        pack((self.t, self.q), stream)


    def __init__(self, t, q):
        self.t = t
        self.q = q


    def em_train(self, corpus, n=3, s=1):
        """Run several iterations of the EM algorithm on the model"""
        for k in range(s,n+s):
            self.em_iter(corpus,passnum=k)


    def em_iter(self, corpus, passnum=1):
        """Run a single iteration of the EM algorithm on the model"""

        start = time.time()
        likelihood = 0.0
        c1 = defaultdict(float) # ei aligned with fj
        c2 = defaultdict(float) # ei aligned with anything
        c3 = defaultdict(float) # wj aligned with wi
        c4 = defaultdict(float) # wi aligned with anything

        # The E-Step
        for k, (f, e) in enumerate(corpus):

            if k % 1000 == 0:
                sys.stderr.write("\rPass %2d: %6.2f%%" % (passnum, (100*k) / float(len(corpus))))
                sys.stderr.flush()

            l = len(e) + 1
            m = len(f) + 1
            e = [None] + e

            for i in range(1, m):

                num = [ self.q[(j, i, l, m)] * self.t[(f[i - 1], e[j])]
                        for j in range(0,l) ]
                den = float(sum(num))
                likelihood += math.log(den)

                for j in range(0, l):

                    delta = num[j] / den

                    c1[(f[i - 1], e[j])] += delta
                    c2[(e[j],)]          += delta
                    c3[(j, i, l, m)]        += delta
                    c4[(i, l, m)]          += delta

        # The M-Step
        self.t = defaultdict(float, {k: v / c2[k[1:]] for k, v in c1.items() if v > 0.0})
        self.q = defaultdict(float, {k: v / c4[k[1:]] for k, v in c3.items() if v > 0.0})

        duration = (time.time() - start)
        sys.stderr.write("\rPass %2d: 100.00%% (Elapsed: %.2fs) (Log-likelihood: %.5f)" % (passnum, duration, likelihood))
        return likelihood, duration


    def viterbi_alignment(self, f, e):
        """Returns an alignment from the provided french sentence to the english sentence"""

        l = len(e) + 1
        m = len(f) + 1
        e = [None] + e

        # for each french word:
        #  - compute a list of indices j of words in the english sentence,
        #    together with the probability of e[j] being aligned with f[i-1]
        #  - take the index j for the word with the _highest_ probability;

        def maximum_alignment(i):
            possible_alignments = [(j, self.t[(f[i - 1], e[j])] * self.q[(j, i, l, m)]) for j in range(0, l)]
            return max(possible_alignments, key=lambda x: x[1])[0]

        return [
            maximum_alignment(i)
            for i in range(1, m)]

if __name__ == "__main__":
    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][
             :opts.num_sents]



    fe_count = defaultdict(int)
    for (n, (f, e)) in enumerate(bitext):
        for f_i in set(f):
            for e_j in set(e):
                fe_count[(e_j, f_i)] += 1

    t_ef = defaultdict(float)
    q = defaultdict(float)
    NumberOfCombinations = len(fe_count)

    for (k, (f, e)) in enumerate(bitext):  # for k = 1..n

        l_k = len(e) + 1
        m_k = len(f)+1
        e = [None] + e
        for i in range(1, m_k):
            for j in range(0, l_k):
                q[(j, i, l_k, m_k)] = random.uniform(0,1)
                t_ef[(f[i-1] , e[j])] = 1 / NumberOfCombinations

    IBM_model1 = IBM_dont_save.IBM(t_ef)
    t_ef=IBM_model1.t


    OurIBMModel = IBM(t_ef,q)
    OurIBMModel.em_train(bitext)

    for (f, e) in bitext:
        # f = f
        for (i, f_i) in enumerate(f):
            for (j, e_j) in enumerate(e):
                if OurIBMModel.t[(f_i, e_j)] >= 0.3:
                    sys.stdout.write("%i-%i " % (i, j))
        sys.stdout.write("\n")