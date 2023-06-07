import logging
import json
import logging
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


def add_text_features(features, dim):
    # Edit text features.
    for d in range(dim):
        features.add(f'edit-dim-{d}', group='edit-embedding')
    # Title text features.
    for d in range(dim):
        features.add(f'title-dim-{d}', group='title-embedding')


def add_edit_embedding(vec, datum):
    # Edit text features.
    for d, emb in enumerate(datum['edit-embedding']):
        vec[f'edit-dim-{d}'] = emb


def add_title_embedding(vec, datum):
    # Title text features.
    for d, emb in enumerate(datum['title-embedding']):
        vec[f'title-dim-{d}'] = emb


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

    if 'dossier_title' in datum:
        text_datum_title = ' '.join([re.sub('\d', 'D', w.lower()) for w in word_tokenize(datum['dossier_title'])])
        feats_title = model_title.get_sentence_vector(text_datum_title)
    else:
        feats_title = np.zeros_like(feats_edit)
    return feats_edit.tolist(), feats_title.tolist()


def get_title_features(datum):
    global model_title
    text_datum_title = ' '.join([re.sub('\d', 'D', w.lower()) for w in word_tokenize(datum['dossier_title'])])
    return model_title.get_sentence_vector(text_datum_title)


def get_features(datum, features):
    edit_embedding, title_embedding = get_text_features(datum)
    datum['edit-embedding'] = edit_embedding
    datum['title-embedding'] = title_embedding
    dim = len(edit_embedding)

    add_text_features(features, dim)

    vec = features.new_vector()
    add_edit_embedding(vec, datum)
    add_title_embedding(vec, datum)

    vec['outsider'] = 1
    if datum['article_type'] and datum['article_type'] in features._idx.keys():
        vec[datum['article_type']] = 1
    else:
        logging.warning(f'Article type {datum["article_type"]} not in features.')
    if datum['edit_type'] and datum['edit_type'] in features._idx.keys():
        vec[datum['edit_type']] = 1
    else:
        logging.warning(f'Edit type {datum["edit_type"]} not in features or None.')
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
        if a['id'] in features._idx.keys():  # MEP of leg 9 not in leg 8 training data
            vec[a['id']] = 1
        vec[a['nationality']] = 1
        if a['group'] in features._idx.keys():
            vec[a['group']] = 1
        vec[a['gender']] = 1
        if a['rapporteur']:
            vec['rapporteur'] = 1

    return vec.as_array()


def get_features_dossier(datum, features):
    vec = features.new_vector()

    dossier = datum
    if dossier['dossier_ref'] in features._idx.keys():
        vec[dossier['dossier_ref']] = 1
    if dossier['dossier_type'] in features._idx.keys():
        vec[dossier['dossier_type']] = 1
    vec[dossier['legal_act']] = 1
    vec[dossier['committee']] = 1  # TODO check if committee is in features am2json
    vec['bias'] = 1
    return vec.as_array()




def get_featmat(docxfile, trained):
    am_per_article = {}
    for i, datum in enumerate(am2json.extract_amendments(docxfile)):
        article = datum['article'] if 'article' in datum else f"article not found {i}"
        if article not in am_per_article:
            am_per_article[article] = []
        am_per_article[article].append(datum)

    for article, am_list in am_per_article.items():
        featmat = [get_features(datum, trained.features) for datum in am_list]
        vec_dossier = get_features_dossier(datum, trained.features)
        featmat.append(vec_dossier)
        am_list.append('statuquo')
        yield article, am_list, np.array(featmat)


def main(docxfile, model_path, model='WarOfWords'):
    TrainedModel = getattr(warofwords, 'Trained' + model)
    trained = TrainedModel.load(model_path)
    for article, am_list, featmat in get_featmat(docxfile, trained):
        scores = trained.probabilities(featmat)
        for datum, score in zip(am_list, scores):
            yield article, datum, score



def main4test_with_zenodo_data(featmat, model_path, model='WarOfWords'):
    TrainedModel = getattr(warofwords, 'Trained' + model)
    trained = TrainedModel.load(model_path)
    scores = trained.probabilities(np.array(featmat))
    for datum, score in zip(featmat, scores):
        yield datum, score


if __name__ == '__main__':
    for article, datum, score in main(sys.argv[1], sys.argv[2]):
        datum = datum if type(datum) == str else datum['amendment_num']
        print(f"Article: {article: <60} | Amendment ID: {datum: <20} | Score: {score}")
