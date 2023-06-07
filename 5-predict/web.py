from io import StringIO
import os
from tempfile import NamedTemporaryFile

import fasttext as fasttext
from annotated_text import annotated_text

import requests
import streamlit as st
import pandas as pd
import numpy as np

import predict


legislature = str(8)
task = 'new_edit-full'

if os.path.exists('ep8-all_features-text-chronological.predict'):
    model_path = 'ep8-all_features-text-chronological.predict'
    predict.model_edit = fasttext.load_model('ep' + legislature + '-' + task + '-edit.bin')
    predict.model_title = fasttext.load_model('ep' + legislature + '-' + task + '-title.bin')

else:
    model_path = '../2-training/trained-models/ep8-all_features-text-chronological.predict'


st.title('🪖 War of Words 💬')
st.subheader('Predicting the success of amendment in the European Parliament')
st.write('Beta version')
st.write('This is a **beta application** based on the War of Words project. '
         'The goal of this project was to study the dynamics of the legislative process in the European Parliament.'
         'We propose to use the machine learning model implemented in that scientific article for predicting the success of amendment')

st.markdown(
    '> Kristof, V., Grossglauser, M., Thiran, P., '
    '[War of Words: The Competitive Dynamics of Legislative Processes](https://infoscience.epfl.ch/record/275473/files/kristof2020war.pdf),'
    ' The Web Conference, April 20-24, 2020, Taipei, Taiwan)')
st.write('A complete documentation will appear at the end of that page after a prediction is made.')




st.header('Upload your amendments')

st.write('Upload your amendments in a .docx format. ')
uploaded_file = st.file_uploader("Choose a file", type=['docx'])

st.write(
    'You can download a sample of amendments [here](https://www.europarl.europa.eu/doceo/document/AGRI-AM-740645_EN.docx)')


def get_filepath(content):
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(content)
        tmp_file.seek(0)
        return tmp_file.name


if uploaded_file is not None:
    docx_content = get_filepath(uploaded_file.getvalue())
else:
    default_file = "https://www.europarl.europa.eu/doceo/document/AGRI-AM-740645_EN.docx"
    st.write('No file uploaded, using default file: ' + default_file)
    docx_content = get_filepath(requests.get(default_file).content)


def structure(docx_content):
    articles = {}
    for i, (article, datum, score) in enumerate(predict.main(docx_content, model_path)):
        if article not in articles:
            articles[article] = []
        articles[article].append((datum, score))
    return articles

def get_stars(score):
    if score < 0.2:
        return '⭐'
    elif score < 0.4:
        return '⭐⭐'
    elif score < 0.6:
        return '⭐⭐⭐'
    elif score < 0.8:
        return '⭐⭐⭐⭐'
    else:
        return '⭐⭐⭐⭐⭐'
def amlist_to_dataframe(amlist):
    df = list()
    for i, (datum, score) in enumerate(amlist):
        am = 'statuquo' if type(datum) == str and datum == 'statuquo' else 'amendment ' + datum.get('amendment_num', '')
        df.append({'amendment': am,
                   'stars': get_stars(score),
                   'probability in %': int(score * 100)})
    return pd.DataFrame(df)

import warofwords

model = 'WarOfWords'
TrainedModel = getattr(warofwords, 'Trained' + model)
trained = TrainedModel.load(model_path)
st.write(len(trained.features))
st.write("---")


st.markdown('## Summary')
md = list(structure(docx_content).values())
md = md[0][0][0]
d = "|   |   |" + "\n"
d += "|---|---|" + "\n"
d += "|__Committee__| " + md['committee'] + " |" + "\n"
d += "|__Dossier__| " + md['dossier_ref'] + " |" + "\n"
d += "|__Id__| " + md['dossier_id'] + " |" + "\n"
d += "|__Date__| " + md['date'] + " |" + "\n"
d += "|__Rapporter__| " + md['rapporteur'] + " |" + "\n"
d += "|__Source__| " + md['source'] + " |" + "\n"
d += "|__Title__| " + md['dossier_title'] + " |" + "\n"
st.markdown(d)
st.write('---')



for article, am_list in structure(docx_content).items():
    st.markdown(f'### Article {article}')
    df = amlist_to_dataframe(am_list)
    styler = df.style.hide().format(subset=['probability in %'], decimal=',', precision=2)\
        .bar(subset=['probability in %'], align="mid", vmax=100, color=['#2841ff', '#d7be00'])
    st.write(styler.to_html(), unsafe_allow_html=True)
    st.write('---')

if st.button('Show details'):

    st.markdown('## Details')
    current_article = None

    for i, (article, datum, score) in enumerate(predict.main(docx_content, model_path)):
        if article != current_article:
            st.write('---')
            st.markdown(f'## Article {article}')
            current_article = article
            df = pd.DataFrame()

        if type(datum) == str and datum == 'statuquo':
            st.markdown('### Statu quo')
        else:
            st.markdown('### Amendment ' + datum.get('amendment_num', ' '))
            try:
                st.markdown("**Author(s):** " + ', '.join([a['name'] + ' (' + a['group']+')' for a in datum.get('authors', ' ')]))
                st.markdown("**Edit type:** " + datum.get('edit_type', ' '))
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Text original")
                    st.markdown(' '.join(datum['text_original']))
                with col2:
                    st.markdown("#### Text amended")
                    st.markdown(' '.join(datum['text_amended']))
                    # annotated_text(
                    #     "This ",
                    #     ("is", "verb"),
                    #     " some ",
                    #     ("annotated", "adj"),
                    #     ("text", "noun"),
                    #     " for those of ",
                    #     ("you", "pronoun"),
                    #     " who ",
                    #     ("like", "verb"),
                    #     " this sort of ",
                    #     ("thing", "noun"),
                    #     "."
                    # )
                # st.json(datum)
            except Exception as e:
                st.markdown('Error ' + str(e))
        st.write(f'Probability of success: {score:.4f}')
        st.progress(int(score * 100), text=str(int(score * 100)))

st.markdown(open('docs.md').read())
