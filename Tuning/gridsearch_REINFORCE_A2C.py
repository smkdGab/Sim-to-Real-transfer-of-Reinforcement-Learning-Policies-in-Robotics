import itertools
from json import load
import os
import pickle
from multiprocessing import Pool
from tqdm import tqdm
from pprint import pprint

from utils_REINFORCE_A2C import train, test


# A2C
params = {
	'lra': [3e-4, 1e-3],
	'lrc': [3e-4, 1e-3],
	'batch_size': [32, 64],
	'use_entropy': [True, False]
}

# # REINFORCE
# params = {
# 	'lra': [1e-3, 1e-4, 1e-5],
# 	'lrc': [1e-3, 1e-4, 1e-5]
# }

MULTIPLE_STARTS = 4

def pool_tt(args:dict):
	agent = train(**args)
	return test(agent)

results = []

keys = list(params.keys())
for p in tqdm(itertools.product(*params.values())):
	kw = dict(zip(keys, p))
	pool = Pool(processes=MULTIPLE_STARTS)
	scores = pool.map(pool_tt, [kw]*MULTIPLE_STARTS)
	score = sum(scores)/len(scores)
	results.append([score, kw])

print(max(results))

with open(f'log_pickle', 'wb') as outfile:
	pickle.dump(obj=results, file=outfile)
