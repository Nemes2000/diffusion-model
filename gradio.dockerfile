FROM diffusion

RUN pip install gradio

WORKDIR /diffusion-model

COPY ./src/gradio_app /diffusion-model/src/gradio_app
COPY ./src/app.py /diffusion-model/src/app.py

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "src/app.py", "-path", "./model/diffusion-model", "-model", "best-flowers-v1"]