FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3-pip

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY docs/requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -e liver-volumetry

# Copy your handler code
COPY docs/handler.py .

# Command to run when the container starts
CMD [ "python3", "-u", "handler.py" ]