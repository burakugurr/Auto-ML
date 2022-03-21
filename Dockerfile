FROM python:3.8-slim-buster

COPY . /app

WORKDIR /app

RUN pip install  --user -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit","run"]

CMD ["app.py"]