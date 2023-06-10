#############################

FROM python:3.10.6-buster

#############################

COPY api.api_file.py /api_file.py

COPY requirements.txt /requirements.txt

COPY PACKAGES /PACKAGES

#############################

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

#RUN pip install . # IF we need tp install own packages, clarification needed

#############################

# One of these must be commented out, depending on local/gps deployment

#local
CMD uvicorn api.api_file.py:app --host 0.0.0.0.

#gps
CMD uvicorn API_FOLDER.API_FILE:api --host 0.0.0.0. --port $PORT
