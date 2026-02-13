FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip python3-dev

# Définissez le répertoire de travail
WORKDIR /app

# Copiez les dossiers
COPY liver-volumetry /app/liver-volumetry
COPY models /app/models/
COPY images /app/images/

# Vérification LFS
RUN find /app/models -name "*.h5" -size -10M -exec echo "ERREUR: Fichier LFS non récupéré : {}" \; -exec false {} +

# Installation des dépendances
RUN pip install --no-cache-dir -e ./liver-volumetry  && rm -rf /root/.cache/pip

# Copie du handler
COPY docs/handler.py .

CMD [ "python3", "-u", "handler.py" ]
