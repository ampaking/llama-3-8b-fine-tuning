# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt\
    && pip install "unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git" \
    && pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes

COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "main.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
