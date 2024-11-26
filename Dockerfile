FROM python:3.10-slim

RUN pip install pipenv

WORKDIR /app

# Copy necessary files into the Docker image
COPY ["Pipfile", "Pipfile.lock", "./"]
COPY src/predict.py ./  
COPY models/model.bin models/dv.bin ./models/

# Install dependencies using Pipenv
RUN pipenv install --system --deploy

# Expose the Flask app port
EXPOSE 9696

# Run the Flask app with Waitress
ENTRYPOINT [ "waitress-serve", "--port=9696", "predict:app" ]
