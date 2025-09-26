FROM ubuntu:24.04

RUN apt-get -y update && \
    apt-get install --assume-yes python3.12 python3-pip

COPY requirements.txt /root
RUN --mount=type=cache,target=/root/.cache python3.12 -m pip install --break-system-packages -r /root/requirements.txt
RUN apt-get install --assume-yes poppler-utils tesseract-ocr

WORKDIR /root
COPY conf ./conf
COPY llm_searcher/ ./llm_searcher
COPY ./docs ./docs

ENTRYPOINT python3.12 llm_searcher --config conf/perovscite.json --searcher perovskite --mode gradio 
