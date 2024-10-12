FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel
RUN apt-get update && apt-get upgrade -y
RUN python -m pip install --upgrade pip

WORKDIR /diffusion-model

COPY /src /diffusion-model/src
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

CMD [ "python",  "./src/main.py"]