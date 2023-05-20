# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

RUN pip3 install MarkupSafe==1.1.1 itsdangerous==2.0.1 Werkzeug==1.0.1

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]