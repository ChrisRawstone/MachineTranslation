#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float",
                     help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=10000, type="int",
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
    for f_i in f:
        f_count[f_i] += 1
        for e_j in e:
            fe_count[(e_j, f_i)] += 1
    for e_j in e:
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
    t_ef[key] = 1 / NumberOfCombinations

s_total = {}
prev = 0
convergenceSum = 0
numProbabilities = 0

while not_converged:
    num_converged = 0
    count = defaultdict(float)
    total = defaultdict(float)

    for (n, (f, e)) in enumerate(bitext):
        for e_j in e:
            s_total[e_j] = 0
            for f_i in f:
                s_total[e_j] += t_ef[(e_j, f_i)]
        for e_j in e:
            for f_i in f:
                count[(e_j, f_i)] += t_ef[(e_j, f_i)] / s_total[e_j]
                total[f_i] += t_ef[(e_j, f_i)] / s_total[e_j]
        for f_i in f:
            for e_j in e:
                prev = t_ef[(e_j, f_i)]
                t_ef[(e_j, f_i)] = count[(e_j, f_i)] / total[f_i]
                convergenceSum += abs(prev - t_ef[(e_j, f_i)])
                numProbabilities += 1
    temp = convergenceSum / numProbabilities
    if (convergenceSum / numProbabilities) < 0.002:
        not_converged = False

for (f, e) in bitext:
  for (i, f_i) in enumerate(f):
    for (j, e_j) in enumerate(e):
      if t_ef[(e_j, f_i)] >= 0.4:
        sys.stdout.write("%i-%i " % (i,j))
  sys.stdout.write("\n")

