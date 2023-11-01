FROM jupyter/scipy-notebook

RUN pip install joblib
RUN pip install nltk



USER root
RUN apt-get update && apt-get install -y jq

COPY Tweets.csv ./Tweets.csv
COPY nlp_processing.py ./nlp_processing.py
COPY nlp_traning.py ./nlp_traning.py
COPY nlp_test.py ./nlp_test.py
