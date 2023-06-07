FROM python:3.9-buster
WORKDIR /app
RUN pip install git+https://github.com/scampion/amendements2json.git@main
#RUN pip install git+https://github.com/scampion/war-of-words-2.git@scampion#subdirectory=lib
COPY lib lib
RUN pip install -e lib
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install streamlit
RUN pip install st-annotated-text
RUN pip install torch
COPY 2-training/trained-models/ep8-all_features-text-chronological.predict /app/ep8-all_features-text-chronological.predict
COPY ep8-new_edit-full-edit.bin /app/ep8-new_edit-full-edit.bin
COPY ep8-new_edit-full-title.bin /app/ep8-new_edit-full-title.bin
COPY 5-predict/predict.py predict.py
COPY 5-predict/web.py web.py
COPY 5-predict/docs.md docs.md
RUN pip install git+https://github.com/scampion/amendements2json.git@main
ENTRYPOINT streamlit run web.py --server.port 8080

