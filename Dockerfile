FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3-pip

# 1. Copiez d'abord le dossier du projet dans le conteneur
COPY liver-volumetry /app/liver-volumetry

# 2. Définissez le répertoire de travail
WORKDIR /app

# 3. Installez en utilisant le chemin relatif (le point "." désigne le dossier courant)
RUN pip install --no-cache-dir -e ./liver-volumetry

# Copy your handler code
COPY docs/handler.py .

# Command to run when the container starts
CMD [ "python3", "-u", "handler.py" ]