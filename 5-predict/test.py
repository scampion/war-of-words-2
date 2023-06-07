import sys
import json
import predict
from tqdm import tqdm

if __name__ == '__main__':
    filepath = sys.argv[1]
    mat = list()	
    for l in tqdm(open(filepath).readlines()):
        data = json.loads(l)
        if '607910' in data[0]['source']:
            mat.append(data[0])
    import IPython; IPython.embed()
    r = predict.main4test_with_zenodo_data(data, sys.argv[2])
    print(list(r))
