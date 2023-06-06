import sys
import json
import warofwords

import numpy as np

import predict
from tqdm import tqdm


def get_featmat(ams, trained):
    featmat = list()
    for datum in ams:
        test = predict.get_features(datum, trained.features)
        featmat.append(test)
    vec_dossier = predict.get_features_dossiers(datum, trained.features)
    featmat.append(vec_dossier)
    return np.array(featmat)


if __name__ == '__main__':
    filepath = sys.argv[1]
    amendements = []
    for l in tqdm(open(filepath).readlines()):
        data = json.loads(l)
        if '607910' in data[0]['source']:
            amendements.append(data[0])
    model = 'WarOfWords'
    TrainedModel = getattr(warofwords, 'Trained' + model)
    trained = TrainedModel.load(sys.argv[2])

    featmat = get_featmat(amendements, trained)

    scores = [score for datum, score in predict.main4test_with_zenodo_data(featmat, sys.argv[2])]
    for datum, score in zip(amendements, scores):
        print(datum['text_amended'], score)
    import IPython; IPython.embed()
    # break
