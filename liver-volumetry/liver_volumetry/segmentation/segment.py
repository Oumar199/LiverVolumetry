"""Liver volume and segmentation analysis
-----

This repository wants to 
- get models and image from configured paths
- do segmentation on image using liver model
- do classification on image using tumor model
- use state of art google health MedGemma based model to analysis results and do diagnosis
"""

from ...liver_volumetry.utils import liver_tumor_pipeline_py as ltp
from transformers import AutoModelForImageTextToText, AutoProcessor
from typing import *
import torch
import os

def get_models(models_path: str = "models\\ModelSegmentation"):
    """Function to models

    Args:
        models_path (str, optional): path to models. Defaults to "models\ModelSegmentation".

    Raises:
        OSError: When liver model is not defined
        OSError: When tumor model is not defined

    Returns:
        tuple: liver model, tumor model
    """
    assert os.path.exists(models_path)
    
    models_paths = [os.path.join(models_path, "final_model_unet_foie.h5"), 
                   os.path.join(models_path, "final_model_tumor_resunet.h5")]
    
    if not os.path.exists(models_paths[0]):
        
        raise OSError("The liver model was not loaded in the defined directory.")
    
    if not os.path.exists(models_paths[1]):
        
        raise OSError("The tumor model was not loaded in the defined directory.")
    
    # Charger les modèles
    model_liver, model_tumor = ltp.load_segmentation_models(
        *(models_paths)
    )
    
    return model_liver, model_tumor


def load_image(image_path: str):
    """Load image of liver

    Args:
        image_path (str): Path to image

    Returns:
        Any: image
    """
    assert os.path.exists(image_path)
    
    # Charger l'image
    img = ltp.load_and_preprocess_image(
        image_path
    )
    
    return img

def segment_image(img: Any, models_path: str = "models\\ModelSegmentation"):
    """Segment image to get regions

    Args:
        img (Any): image
        models_path (str, optional): Path to models. Defaults to "models\ModelSegmentation".

    Returns:
        tuple: liver mask, tumor mask
    """
    liver_model, tumor_model = get_models(models_path)
    
    # Segmentation
    liver_mask, tumor_mask = ltp.run_segmentation(
        img, liver_model, tumor_model
    )
    
    return liver_mask, tumor_mask

def identify_volumes(img: Any, masks: tuple):
    """Trace overlay and identify volums

    Args:
        img (Any): Image
        masks (tuple): Masks

    Returns:
        tuple: overlay, volumes
    """
    # Overlay
    overlay = ltp.build_overlay(img, *masks)

    # Volumes (EXEMPLE valeurs)
    volumes = ltp.compute_volumes(
        masks[0],
        masks[1],
        pixel_area_mm2=0.61,
        slice_thickness_mm=1.6
    )
    
    return overlay, volumes

def get_llm_and_processor(model_repo: str = "Metou/MedGemma-1.5-4B", subfolder: str = "bismedgemma-4bit"):
    """Get llm and processor

    Args:
        model_repo (str, optional): huggingface repository. Defaults to "Metou/MedGemma-1.5-4B".
        subfolder (str, optional): the subfolder to model. Defaults to "bismedgemma-4bit".

    Returns:
        tuple: model, processor
    """
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_repo,
        subfolder=subfolder,
        device_map="auto",
        dtype=torch.float16
    )

    # 🔹 Processor OFFICIEL MedGemma (OBLIGATOIRE)
    processor = AutoProcessor.from_pretrained(
        "google/medgemma-1.5-4b-it",
        use_fast=False  # IMPORTANT
    )
    
    return model, processor

def analysis_image(img, models_path: str = "models\\ModelSegmentation", model_repo: str = "Metou/MedGemma-1.5-4B", subfolder: str = "bismedgemma-4bit")
    """Plot segmentation and volumes and obtain analysis from llm

    Returns:
        str: analysis
    """
    
    model_llm, processor = get_llm_and_processor(model_repo, subfolder)
    
    masks = segment_image(img, models_path)
    
    overlay, volumes = identify_volumes(img, masks)
    
    # MedGemma
    model_llm, processor = ltp.load_medgemma_4bit()
    analysis = ltp.run_medgemma_analysis(
        model_llm,
        processor,
        overlay,
        volumes
    )
    
    ltp.plot_results(img, *masks)
    
    return analysis


"""## Example of output :

Abdominal CT slice with liver and tumor segmentation.

Liver volume: 8.26 cm3
Tumor volume: 0.53 cm3
Tumor-to-liver ratio: 6.36%

Provide a concise clinical interpretation.
model
This is a CT scan of the abdomen showing a small liver lesion. The liver volume is 8.26 cm3, and the tumor volume is 0.53 cm3. The tumor-to-liver ratio is 6.36%. This suggests a small liver lesion, which may be a benign or malignant tumor. Further evaluation is needed to determine the nature of the lesion.

**Disclaimer:** This is an AI-generated interpretation and should not be used for medical diagnosis. A qualified healthcare professional should be consulted for any medical concerns.
"""