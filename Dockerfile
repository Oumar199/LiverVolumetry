# Utilisation de l'image 'devel' au lieu de 'runtime'
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Install Python (Note: Python 3.10 est le standard sur Ubuntu 22.04, 
# si vous voulez spécifiquement 3.11, gardez votre ligne mais assurez-vous des liens symboliques)
RUN apt-get update && apt-get install -y python3.11 python3-pip python3-dev

# Définissez le répertoire de travail
WORKDIR /app

# Copiez les dossiers
COPY liver-volumetry /app/liver-volumetry
COPY models /app/models/
COPY images /app/images/

# Vérification LFS
RUN find /app/models -name "*.h5" -size -10M -exec echo "ERREUR: Fichier LFS non récupéré : {}" \; -exec false {} +

# Installation des dépendances
RUN pip install --no-cache-dir -e ./liver-volumetry

# Copie du handler
COPY docs/handler.py .

# Variable d'environnement pour aider TensorFlow à trouver les outils CUDA si besoin
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

CMD [ "python3", "-u", "handler.py" ]
