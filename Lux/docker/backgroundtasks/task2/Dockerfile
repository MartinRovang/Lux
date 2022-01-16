FROM python:3.9


WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./lux/src /code/backgroundtasks
COPY ./lux/database /code/backgroundtasks/tools/database

RUN python -m pip install --upgrade pip
RUN pip install -r /code/requirements.txt


CMD python3 /code/backgroundtasks/tools/backgroundprocess2.py