from setuptools import setup, find_packages

setup(
    name="liver_volumetry_analysis",
    version="0.0.1",
    author="Methou Sanghe, Mamadou Bousso, Aby Diallo, Oumar Kane, Cheikh Yakhoub Maas",
    packages=find_packages(),
    author_email="metou.sanghe@univ-thies.sn",
    description="Contains modules to segment liver to detect fibrosis, estimate fibrosis volume, and analysis results using state of art medical LLM for diagnosis.",
    install_requires=[
        "runpod==1.3.0",
        "torch==2.5.1",          # Version stable pour CUDA 12
        "torchvision==0.20.1",   # Aligné sur torch 2.5.1
        "tensorflow>=2.16.1",
        "keras>=3.4.0",  # Remplace tf-keras
        "numpy<2.0.0",
        "transformers>=4.40.0",  # Version plus standard
        "accelerate>=0.30.0",
        "safetensors>=0.4.0",
        "pillow>=10.0.0",
        "matplotlib>=3.8.0",
        "bitsandbytes>=0.42.0",
        "sentencepiece",
        "protobuf<=5.28.3",      # Évite les conflits de génération de code avec TF
    ],
)
