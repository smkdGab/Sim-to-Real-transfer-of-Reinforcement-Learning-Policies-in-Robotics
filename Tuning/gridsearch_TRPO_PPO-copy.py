import itertools
import os
import pickle
from multiprocessing import Pool
from tqdm import tqdm
from pprint import pprint

# Choose one:
from PPO_train_test import train, test
# from TRPO_train_test import train, test

os.makedirs('outputs')

# PPO:
params = {
	'learning_rate': [3e-4, 2.5e-4, 1e-3],
	'batch_size': [64, 128],
    'ent_coef': [0, 0.01],
	'n_steps': [1024, 2048]
}

# TRPO:
# params = {
# 	'learning_rate': [3e-4, 2.5e-4, 1e-3],
# 	'batch_size': [64, 128],
#	'n_steps': [1024, 2048]
#}


MULTIPLE_STARTS = 4

def pool_tt(args: dict):
	model = train(args)
	return test(model)


results = []
counter = 0 

keys = list(params.keys())
for p in tqdm(itertools.product(*params.values())):
	kw = dict(zip(keys, p))
	pprint(kw)
	pool = Pool(processes=MULTIPLE_STARTS)
	scores = pool.map(pool_tt, [kw]*MULTIPLE_STARTS)
	score = sum(scores)/len(scores)
	results.append([score, kw])
	counter += 1
	with open(f'outputs/log_{counter}.pickle', 'wb') as outfile:
		pickle.dump(obj=results, file=outfile)

    # alternative log
	# np.savez(f'outputs/log_{counter}', results=results)

print(max(results, key=lambda x:x[0]))
