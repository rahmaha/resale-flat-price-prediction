FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Rest of your Dockerfile remains the same
RUN pip install --no-cache-dir pipenv


WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "model.bin", "dv.bin", "predict.py", "./"] 

RUN pipenv install --system --deploy

EXPOSE 9696

ENTRYPOINT [ "waitress-serve", "--port=9696", "predict:app" ]