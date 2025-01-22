FROM python:3.11-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY ./resnet-50_v1.5 ./resnet-50_v1.5

COPY inference.py inference.py

EXPOSE 8080

CMD ["python", "inference.py"]

