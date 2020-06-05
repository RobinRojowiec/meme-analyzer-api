FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

# copy requirements descriptor
COPY ./requirements.txt /requirements.txt

# install dependencies
RUN pip install --upgrade pip
RUn pip install --upgrade pipenv
RUN pip install --upgrade -r /requirements.txt

# copy remaining files
COPY . .


# start server
ENTRYPOINT [ "python" ]
CMD [ "server.py" ]