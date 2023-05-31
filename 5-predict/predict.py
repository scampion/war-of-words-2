import json
import re
import sys

import numpy as np
from warofwords import Features
import numpy as np
import warofwords
from warofwords.utils import build_name, get_base_dir, parse_definition
from nltk.tokenize import word_tokenize
import fasttext

import am2json


def docx2json(docxfile):
    return json.dumps(docxfile)


legislature = str(8)
task = 'new_edit-full'
model_edit = fasttext.load_model('../data/ep' + legislature + '-' + task + '-edit.bin')
model_title = fasttext.load_model('../data/ep' + legislature + '-' + task + '-title.bin')


def get_text_features(datum):
    global model_edit, model_title
    i1 = datum['edit_indices']['i1']
    i2 = datum['edit_indices']['i2']
    j1 = datum['edit_indices']['j1']
    j2 = datum['edit_indices']['j2']

    text_del = datum['text_original'][i1:i2]
    text_ins = datum['text_amended'][j1:j2]
    text_context_l = datum['text_original'][:i1]
    text_context_r = datum['text_original'][i2:]

    text_datum = '<con>' + ' <con>'.join(text_context_l) + ' <del>' + ' <del>'.join(
        text_del) + ' <con>' + ' <con>'.join(text_context_r) + ' <ins>' + ' <ins>'.join(text_ins)
    feats_edit = model_edit.get_sentence_vector(text_datum)

    text_datum_title = ' '.join([re.sub('\d', 'D', w.lower()) for w in word_tokenize(datum['dossier_title'])])
    feats_title = model_title.get_sentence_vector(text_datum_title)
    return feats_edit, feats_title


def get_title_features(datum):
    global model_title
    text_datum_title = ' '.join([re.sub('\d', 'D', w.lower()) for w in word_tokenize(datum['dossier_title'])])
    return model_title.get_sentence_vector(text_datum_title)

def add_title_embedding(vec, datum):
    # Title text features.
    for d, emb in enumerate(datum['title-embedding']):
        vec[f'title-dim-{d}'] = emb


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

    datum['edit-embedding'] = get_text_features(datum)
    dim = len(datum['edit-embedding'])

    datum['title-embedding'] = get_text_features(datum)

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

    dossier = datum
    vec[dossier['dossier_ref']] = 1
    vec[dossier['dossier_type']] = 1
    vec[dossier['legal_act']] = 1
    vec[dossier['committee']] = 1
    vec['bias'] = 1
    add_title_embedding(vec, datum)
    return vec


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
