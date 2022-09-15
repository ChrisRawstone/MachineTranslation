#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=1000, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)


bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]
f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)
t_ef = defaultdict(int)


for (n, (f, e)) in enumerate(bitext):
  for f_i in set(f):
    f_count[f_i] += 1
    for e_j in set(e):
        fe_count[(e_j,f_i)] += 1
  for e_j in set(e):
    e_count[e_j] += 1
  if n % 500 == 0:
      sys.stderr.write(".")

#initialisation
NumberOfCombinations=len(fe_count)

for key in fe_count:
    t_ef[key]=1/NumberOfCombinations


sys.stderr.write(".")

not_converged=True

iter=0
stop_iter=5

# initialize t(e|f) uniformly
num_converged = 0
NumberOfCombinations=len(fe_count)

for key in fe_count:
    t_ef[key]=1/NumberOfCombinations


s_total={}
prev=0

while not_converged:
    num_converged = 0
    count = defaultdict(int)
    total = defaultdict(int)



    for (n, (f, e)) in enumerate(bitext):
        for e_j in set(e):
            s_total[e_j]= 0
            for f_i in set(f):
                s_total[e_j] += t_ef[(e_j,f_i)]
        for e_j in set(e):
            for f_i in set(f):
                count[e_j, f_i] += t_ef[(e_j,f_i)] / s_total[e_j]
                total[f_i] += t_ef[(e_j,f_i)] / s_total[e_j]
        for f_i in set(f):
            for e_j in set(e):
                prev =
                t_ef[(e_j, f_i)] = count[e_j, f_i] / total[f_i]



    if abs(prev-t_ef[("the", "le")]) < 0.001:
        not_converged=False

    prev=t_ef[("the", "le")]

    # // initialize
    # count(e|f ) = 0 for a   ll e
    # f total(f) = 0 for all f
    # for all sentence pairs (e,f) do
    #     // compute normalization
    #     for all words e in e do
    #         s-total(e) = 0
    #         for all words f in f do
    #             s-total(e) += t(e|f)
    #         end for
    #     end for












    # if iter==stop_iter:
    #     print("stopped")
    #     break
    iter+=1

print(iter)


