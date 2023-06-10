#############################

FROM python:3.10.6-buster

#############################

COPY api /api

COPY requirements.txt /requirements.txt

COPY ml_logic /ml_logic

COPY raw_data /raw_data

#############################

RUN pip install --upgrade pip
#RUN pip install scikit-learn
RUN pip install -r requirements.txt
#RUN pip install .
# IF we need tp install own packages, clarification needed

#############################

# One of these must be commented out, depending on local/gps deployment

#local
#CMD uvicorn api.api_file.py:app --host 0.0.0.0.

#gps
CMD uvicorn api.api_file:api --host 0.0.0.0.
#--port $PORT
