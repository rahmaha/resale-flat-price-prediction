FROM python:3.10-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock","./"] 

RUN pipenv install --system --deploy

#copy model files and source code
COPY models/ /app/models/
COPY src/ /app/src/


EXPOSE 9696

ENTRYPOINT [ "waitress-serve", "--port=9696", "predict:app" ]