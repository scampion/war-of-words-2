import json
import sys

import numpy as np
from warofwords import Features
import numpy as np
import warofwords
from warofwords.utils import build_name, get_base_dir, parse_definition

import am2json

def docx2json(docxfile):
    return json.dumps(docxfile)


def get_text_features(add_embeddings):
    # TODO: Implement this.
    return np.zeros_like(10, 128)


def get_features(datum):
    featmats = list()
    # Initialize features.
    features = Features()
    features.add('bias', group='bias')
    features.add('rapporteur', group='rapporteur')
    features.add('insert-length', group='edit-length')
    features.add('delete-length', group='edit-length')
    features.add('justification', group='justification')
    features.add('outsider', group='outsider')
    features.add(datum['article_type'], group='article-type')
    features.add(datum['edit_type'], group='edit-type')
    # Dossier features.
    features.add(datum['dossier_ref'], group='dossier')
    features.add(datum['committee'], group='committee')
    features.add(datum['dossier_type'], group='dossier-type')
    # Add legal act.
    features.add(datum['legal_act'], group='legal-act')
    # MEP features.
    for a in datum['authors']:
        features.add(a['id'], group='mep')
        features.add(a['nationality'], group='nationality')
        features.add(a['group'], group='political-group')
        features.add(a['gender'], group='gender')

    dim = add_embeddings(datum)

    for d in range(dim):
        features.add(f'edit-dim-{d}', group='edit-embedding')
        features.add(f'title-dim-{d}', group='title-embedding')

    vec = features.new_vector()
    for d, emb in enumerate(datum['edit-embedding']):
        vec[f'edit-dim-{d}'] = emb
    for d, emb in enumerate(datum['title-embedding']):
        vec[f'title-dim-{d}'] = emb

    vec['outsider'] = 1
    vec[datum['article_type']] = 1
    vec[datum['edit_type']] = 1
    just = datum['justification']
    # With text features, the justification is either None or the whole
    # text, without text it is True or False.
    if type(just) is not bool:
        just = (just is not None) and (just != '')
    if just:
        vec['justification'] = 1

    i1, i2 = datum['edit_indices']['i1'], datum['edit_indices']['i2']
    j1, j2 = datum['edit_indices']['j1'], datum['edit_indices']['j2']
    vec['insert-length'] = np.log(1 + j2 - j1)
    vec['delete-length'] = np.log(1 + i2 - i1)

    for a in datum['authors']:
        vec[a['id']] = 1
        vec[a['nationality']] = 1
        vec[a['group']] = 1
        vec[a['gender']] = 1
        if a['rapporteur']:
            vec['rapporteur'] = 1
    featmat.append(vec.as_sparse_list())

    dossier = data[0]
    vec[dossier['dossier_ref']] = 1
    vec[dossier['dossier_type']] = 1
    vec[dossier['legal_act']] = 1
    vec[dossier['committee']] = 1
    vec['bias'] = 1
    if args.text_features:
        # Add title embedding.
        add_title_embedding(vec, datum)
    featmat.append(vec.as_sparse_list())

    return featmat


def main(docxfile, model_path, model='WarOfWords'):
    TrainedModel = getattr(warofwords, 'Trained' + model)
    trained = TrainedModel.load(model_path)

    for datum in am2json.extract_amendments(docxfile):
        test = get_features(datum)
        acc = trained.accuracy(test)
        los = trained.log_loss(test)
        print(datum)
        print(f'  Accuracy: {acc * 100:.2f}%')
        print(f'  Log-loss: {los:.4f}')
        print("#" * 80)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
