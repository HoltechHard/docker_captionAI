FROM python:3.8.16

WORKDIR ./docker_captionAI

ADD . .

RUN pip install -r requirements.txt

CMD ["python3", "./appTIGenAI/appTIGenAI/manage.py", "runserver", "192.168.199.88:5000"]

EXPOSE 5000