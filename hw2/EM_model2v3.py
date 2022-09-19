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
    # if n % 500 == 0:
        # sys.stderr.write(".")

# initialisation

# sys.stderr.write(".")


s_total = {}

NumberOfCombinations = len(fe_count)

q = dict()
t = dict()
for (k, (f, e)) in enumerate(bitext):  # for k = 1..n
    f = [None] + f
    l_k = len(e) + 1
    m_k = len(f)
    for i in range(0, m_k):
        for j in range(1, l_k):
            q[(i, j, l_k, m_k)] = 1 / NumberOfCombinations
            t_ef[(e[j-1],f[i])] = 1 / NumberOfCombinations

# def delta(k,j,i,e,f):
#     l_k=len(e)+1
#     m_k=len(f)
#
#
#     numerator = q[(i,j,l_k,m_k)]*t_ef[(e[j-1],f[i])]
#
#     denominator = sum([(q[(i,j,l_k,m_k)]*t_ef[(e[j-1],f[i])]) for i in range(0,m_k)])
#     return numerator/denominator

iterations = 0



for s in range(5):


    c_ef = defaultdict(float)
    c_f = defaultdict(float)
    c_ij = defaultdict(float)
    c_j = defaultdict(float)



    for (k, (f, e)) in enumerate(bitext):  # for k = 1..n
        f = [None] + f
        l_k = len(e)+1
        m_k = len(f)


        for j in range(1,l_k):
            numerator=[]

            for z in range(0, m_k):
                numerator.append( t_ef[(e[j - 1], f[z])])

            denominator = sum(numerator)

            for i in range(0,m_k):

                d = numerator[i] / denominator

                c_ef[(e[j - 1], f[i])] += d
                c_f[(f[i],)] += d
                c_ij[(i, j, l_k, m_k)] += d
                c_j[(j, l_k, m_k)] += d


    for (k, (f, e)) in enumerate(bitext):  # for k = 1..n
        f = [None] + f
        l_k = len(e)+1
        m_k = len(f)
        for j in range(1, l_k): # for
            for i in range(m_k): # for j = 0..l
                q[(j,i,l_k,m_k)] = c_ij[(i, j, l_k, m_k)] / c_j[(j, l_k, m_k)]
                t_ef[(e[j-1],f[i])] = c_ef[(e[j - 1], f[i])] / c_f[(f[i],)] # looks out for c2!!!




for s in range(5):

    c_ef = defaultdict(float)
    c_f = defaultdict(float)
    c_ij = defaultdict(float)
    c_j = defaultdict(float)



    for (k, (f, e)) in enumerate(bitext):  # for k = 1..n
        f = [None] + f
        l_k = len(e)+1
        m_k = len(f)


        for j in range(1,l_k):
            numerator=[]

            for z in range(0, m_k):
                numerator.append(q[(z, j, l_k, m_k)] * t_ef[(e[j - 1], f[z])])

            denominator = sum(numerator)

            for i in range(0,m_k):

                d = numerator[i] / denominator

                c_ef[(e[j - 1], f[i])] += d
                c_f[(f[i],)] += d
                c_ij[(i, j, l_k, m_k)] += d
                c_j[(j, l_k, m_k)] += d


    for (k, (f, e)) in enumerate(bitext):  # for k = 1..n
        f = [None] + f
        l_k = len(e)+1
        m_k = len(f)
        for j in range(1, l_k): # for
            for i in range(m_k): # for j = 0..l
                q[(j,i,l_k,m_k)] = c_ij[(i, j, l_k, m_k)] / c_j[(j, l_k, m_k)]
                t_ef[(e[j-1],f[i])] = c_ef[(e[j - 1], f[i])] / c_f[(f[i],)] # looks out for c2!!!

    sys.stderr.write("{}\n".format(s))
    sys.stderr.write("the,les, {} \n".format(t_ef[("the","les")]))
    sys.stderr.write("the,la  {} \n".format(t_ef[("the", "le")]))
    sys.stderr.write("the, chacun  {} \n".format(t_ef[("the","chacun")]))
    sys.stderr.write("the, none  {} \n".format(t_ef[("the", None)]))


for (f, e) in bitext:
  # f = f
  for (i, f_i) in enumerate(f):
    for (j, e_j) in enumerate(e):
      if t_ef[(e_j, f_i)] >= 0.3:
        sys.stdout.write("%i-%i " % (i,j))
  sys.stdout.write("\n")








