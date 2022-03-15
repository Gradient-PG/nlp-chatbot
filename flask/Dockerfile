FROM python:3

WORKDIR /app

ADD . /app/

RUN pip install --no-cache-dir Flask uwsgi requests flask-cors

CMD ["uwsgi", "uwsgi.ini"]
