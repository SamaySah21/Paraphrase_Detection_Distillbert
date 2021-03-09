FROM python:3.9-slim-buster

WORKDIR /office_demo

COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 9010

COPY . .

ENTRYPOINT ["python"]

CMD ["app.py"]