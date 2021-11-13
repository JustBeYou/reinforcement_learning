from itertools import product
from multiprocessing import Pool
import pandas as pd

"""
params = {param1: [... values ...], param2: [... values ...], ...}
func = (**kwargs) -> value
"""
def gridsearch(params, func, workers=1, print_every=50000):
    sizes = [len(params[k]) for k in params]
    split_index = _find_min_split_index(sizes, workers)

    values = [params[k] for k in params]
    prefixes = list(product(*values[:split_index]))
    prefixes_parts = _split_n_chunks(prefixes, workers)
    
    worker_args = [(i, 
        prefix_part, 
        values[split_index:], 
        list(params.keys()), 
        func,
        print_every) for i,prefix_part in enumerate(prefixes_parts)]

    with Pool(workers) as p:
        result = p.map(_worker_main, worker_args)

    all_results = []
    for x in result:
        all_results.extend(x)
    all_results.sort(key=lambda x: -x[0])
    
    d = {"result": []}
    for i in range(min(5, len(all_results))):
        d["result"].append(all_results[i][0])
        for j, name in enumerate(params.keys()):
            if name in d:
                d[name].append(all_results[i][1][j])
            else:
                d[name] = [all_results[i][1][j]]
    return pd.DataFrame(d)

def _worker_main(args):
    idx = args[0]
    prefixes = args[1]
    values = args[2]
    param_names = args[3]
    f = args[4]
    print_every = args[5]

    params_to_pass = {}
    results = []
    sufixes = product(*values)
    to_process = len(prefixes)
    for v in values:
        to_process *= len(v)

    cnt = 0

    print(f"Worker {idx} has to process {to_process} values.")
    for prefix in prefixes:
        for sufix in sufixes:
            complete = [*prefix, *sufix]
            for param_name, value in zip(param_names, complete):
                params_to_pass[param_name] = value

            cnt += 1
            if cnt % print_every == 0 and cnt > 1:
                max_val = max(results, key=lambda x: x[0])
                print(f"Worker {idx} processed {cnt}/{to_process} with best result {max_val}")

            _keep_best_n(results, (f(**params_to_pass), complete), 5, key=lambda x: x[0])

    print(f"Worker {idx} processed {cnt}/{to_process} with best result {max_val}. Exit")
    return results

# keep n largest values
def _keep_best_n(v, x, n=5, key=lambda x: x):
    if len(v) < n:
        v.append(x)
        return

    min_val, min_pos = v[0], 0
    for i,val in enumerate(v):
        if key(val) < key(min_val):
            min_val = val
            min_pos = i
    v[min_pos] = x

def _split_n_chunks(v, n):
    sz = int(len(v) / n)
    rem = len(v) % n

    if rem == 0:
        return [v[i:i+sz] for i in range(0, len(v), sz)]

    return [v[:sz+rem]] + [v[i:i+sz] for i in range(sz+rem, len(v), sz)]

def _find_min_split_index(sizes, workers):
    p = 1
    for i, size in enumerate(sizes):
        p *= size
        if p > workers:
            return i+1

    raise Exception("Too little work, reduce the number of workers")