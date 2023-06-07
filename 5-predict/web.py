from io import StringIO
from tempfile import NamedTemporaryFile

import streamlit as st
import pandas as pd
import numpy as np

import predict

model_path = '../2-training/trained-models/ep8-all_features-text-chronological.predict'

st.title('🪖 War of Words 💬')
st.subheader('Predicting the success of amendment in the European Parliament')
st.write('Beta version')
st.write('This is a **beta application** based on the War of Words project. '
         'The goal of this project was to study the dynamics of the legislative process in the European Parliament.'
         'We propose to use the machine learning model implemented in that scientific article for predicting the success of amendment')

st.markdown('> [Kristof, V., Grossglauser, M., Thiran, P., War of Words: The Competitive Dynamics of Legislative Processes, The Web Conference, April 20-24, 2020, Taipei, Taiwan](https://infoscience.epfl.ch/record/275473/files/kristof2020war.pdf)')
st.write('A complete documentation will appear at the end of that page after a prediction is made.')

st.subheader('1. Upload your amendments')

st.write('Upload your amendments in a .docx format. ')
uploaded_file = st.file_uploader("Choose a file", type=['docx'])

st.write('You can download a sample of amendments [here](https://www.europarl.europa.eu/doceo/document/AGRI-AM-740645_EN.docx)')

if uploaded_file is not None:
    with NamedTemporaryFile(delete=True) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.seek(0)
        docx_content = tmp_file.name
        for i, (datum, score) in enumerate(predict.main(docx_content, model_path)):
            st.subheader(f'Amendment ' + datum.get('amendment_num', ' '))
            try:
              st.markdown("**Edit type:** " + datum.get('edit_type', ' '))
              st.markdown('**Text original:** ' +  ' '.join(datum['text_original']))
              st.markdown('**Text amended:** ' + ' '.join(datum['text_amended']))
            except Exception as e:
              st.markdown('Error ' + str(e))
            st.write(f'Probability of success: {score:.4f}')
            st.progress(int(score*100), text=str(int(score*100)))
            st.write('---')


st.markdown(open('docs.md').read())

