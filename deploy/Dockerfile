# Use the official Python 3.9 base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Expose the API port
EXPOSE 8000

COPY . /code

WORKDIR /code/app
CMD ["python", "app.py"]

