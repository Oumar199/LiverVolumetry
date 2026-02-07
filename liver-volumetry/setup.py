from setuptools import setup, find_packages

setup(
    name="liver_volumetry_analysis",
    version="0.0.1",
    author="Methou Sanghe, Mamadou Bousso, Aby Diallo, Oumar Kane, Cheikh Yakhoub Maas",
    packages=find_packages(),
    author_email="metou.sanghe@univ-thies.sn",
    description="Contains modules to segment liver to detect fibrosis, estimate fibrosis volume, and analysis results using state of art medical LLM for diagnosis.",
    install_requires=[
        "numpy==2.0.2",
        "torch==2.8.0+cu126",
        "torchvision==0.23.0+cu126",
        "tensorflow==2.19.0",
        "keras==3.10.0",
        "transformers==4.57.1",
        "accelerate==1.11.0",
        "safetensors==0.6.2",
        "pillow==11.3.0",
        "matplotlib==3.10.0",
        "bitsandbytes>=0.46.1",
    ],
)
