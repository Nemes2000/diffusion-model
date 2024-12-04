FROM python:3.12
RUN apt-get update && apt-get upgrade -y
RUN python -m pip install --upgrade pip

WORKDIR /diffusion-model

COPY /src /diffusion-model/src
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

CMD [ "python",  "./src/main.py"]