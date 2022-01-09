# 
FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt
COPY ./lux /code/lux

# 
RUN pip install -r /code/requirements.txt

#
# 
CMD ["uvicorn", "lux:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

