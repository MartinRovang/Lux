# 
FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt
COPY ./lux /code/lux
COPY ./startupscript.sh /code/startupscript.sh

# 
RUN python -m pip install --upgrade pip
RUN pip install -r /code/requirements.txt

#
#
CMD bash /code/startupscript.sh

