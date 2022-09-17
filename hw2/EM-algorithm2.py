#!/usr/bin/env python
import optparse
import random
import sys
from collections import defaultdict

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

bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]
f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)
t_ef = defaultdict(float)

for (n, (f, e)) in enumerate(bitext):
    for f_i in set(f):
        f_count[f_i] += 1
        for e_j in set(e):
            fe_count[(e_j, f_i)] += 1
    for e_j in set(e):
        e_count[e_j] += 1
    if n % 500 == 0:
        sys.stderr.write(".")

# initialisation
NumberOfCombinations = len(fe_count)

for key in fe_count:
    t_ef[key] = 1 / NumberOfCombinations

sys.stderr.write(".")

not_converged = True


# initialize t(e|f) uniformly
num_converged = 0
NumberOfCombinations = len(fe_count)

for key in fe_count:
    t_ef[key] = random.uniform(0, 1)

s_total = {}
prev = 0
convergenceSum = 0
numProbabilities = 0



# def q(j,i,l,m):
#     if flag:
#         return random.randint(10)
#     return c3[(j,i,l,m)]/c4[(i,l,m)]


q = dict()
for (k, (f, e)) in enumerate(bitext):  # for k = 1..n
    l = len(e)
    m = len(f)
    for i in range(0, m):
        for j in range(0, l):
            q[(j, i, l, m)] = random.uniform(0, 1)

def delta(k,i,j,e,f):
    l_k=len(e)
    m_k=len(f)
    numerator = q[(j,i,l_k,m_k)]*t_ef[(e[j],f[i])]
    denominator = sum([(q[(j,i,l_k,m_k)]*t_ef[(e[j],f[i])]) for j in range(0,l_k)])
    return numerator/denominator

c3 = defaultdict(float)

iterations = 0



for s in range(5):

    c1 = defaultdict(float)
    c2 = defaultdict(float)
    c3 = defaultdict(float)
    c4 = defaultdict(float)

    for (k, (f, e)) in enumerate(bitext):  # for k = 1..n
        l_k = len(e)
        m_k = len(f)
        for j, e_j in enumerate(e): # for
            for i,f_i in enumerate(f):

                c1[(e_j, f_i)] += delta(k,i,j,e,f)
                c2[f_i] += delta(k,i,j,e,f)
                c3[(j, i, l_k, m_k)] += delta(k,i,j,e,f)
                c4[(i, l_k, m_k)] += delta(k,i,j,e,f)

    for (k, (f, e)) in enumerate(bitext):  # for k = 1..n
        l_k = len(e)
        m_k = len(f)
        for j, e_j in enumerate(e): # for
            for i, f_i in enumerate(f): # for j = 0..l
                q[(j,i,l_k,m_k)] = c3[(j, i, l_k, m_k)] / c4[(i, l_k, m_k)]
                t_ef[(e_j,f_i)] = c1[(e_j, f_i)]/c2[f_i]

    # print(t_ef[("the", "les")])

for (f, e) in bitext:
  for (i, f_i) in enumerate(f):
    for (j, e_j) in enumerate(e):
      if t_ef[(e_j, f_i)] >= 0.5:
        sys.stdout.write("%i-%i " % (i,j))
  sys.stdout.write("\n")








