"""Liver volume and segmentation analysis
-----

This repository wants to 
- get models and image from configured paths
- do segmentation and calculate tumor volume on image using liver and tumor models
- use state of art google health MedGemma based model to analysis results and provide diagnosis
"""

from liver_volumetry.utils import liver_tumor_pipeline_py as ltp
from transformers import AutoModelForImageTextToText, AutoProcessor
from huggingface_hub import snapshot_download
from typing import *
import torch
import os

def download_segmentation_models(hf_id: str = "Metou/ModelSegmentation", local_directory: str = "models"):
    """Download segmentation models from our huggingface account
    """
    path = snapshot_download(hf_id, local_dir=local_directory)
    
    return path

def get_models(models_path: str = "models/ModelSegmentation"):
    """Function to load liver and tumor models

    Args:
        models_path (str, optional): path to models. Defaults to "models/ModelSegmentation".

    Raises:
        OSError: When liver model is not defined
        OSError: When tumor model is not defined

    Returns:
        tuple: liver model, tumor model
    """
    assert os.path.exists(models_path)

    models_paths = [
        os.path.join(models_path, "final_model_unet_foie.h5"),
        os.path.join(models_path, "final_model_tumor_resunet.h5"),
    ]

    if not os.path.exists(models_paths[0]):

        raise OSError("The liver model was not loaded in the defined directory.")

    if not os.path.exists(models_paths[1]):

        raise OSError("The tumor model was not loaded in the defined directory.")

    # Charger les modèles
    model_liver, model_tumor = ltp.load_segmentation_models(*(models_paths))

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
    img = ltp.load_and_preprocess_image(image_path)

    return img


def segment_image(img: Any, models: tuple):
    """Segment image to get regions

    Args:
        img (Any): image
        models (tuple, optional): liver and tumor models.

    Returns:
        tuple: liver mask, tumor mask
    """

    liver_model, tumor_model = models

    # Segmentation
    liver_mask, tumor_mask = ltp.run_segmentation(img, liver_model, tumor_model)

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
        masks[0], masks[1], pixel_area_mm2=0.61, slice_thickness_mm=1.6
    )

    return overlay, volumes


def get_llm_and_processor(
    model_repo: str = "Metou/MedGemma-1.5-4B",
    subfolder: str = "bismedgemma-4bit",
    from_local_files: bool = True,
    quantized: bool = True
):
    """Get llm and processor

    Args:
        model_repo (str, optional): huggingface repository (or local path). Defaults to "Metou/MedGemma-1.5-4B".
        subfolder (str, optional): the subfolder to quantized model for drastic differentiation and abstraction. Defaults to "bismedgemma-4bit".
        from_local_files (bool, optional): Indicates if the model and processor are got from local files. Defaults to True.
        quantized (bool, optional): Indicates if we want to use the quantized version of the model. Defaults to True.
        
    Returns:
        tuple: model, processor
    """

    if quantized:
        
        model = AutoModelForImageTextToText.from_pretrained(
            model_repo, 
            subfolder=subfolder, 
            device_map="auto", 
            torch_dtype=torch.float16,
            local_files_only=from_local_files
        )

        processor = AutoProcessor.from_pretrained(
            model_repo, 
            subfolder=subfolder, 
            use_fast=False,
            local_files_only=from_local_files
        )
    
    else:
        
        model = AutoModelForImageTextToText.from_pretrained(
            model_repo, 
            device_map="auto", 
            torch_dtype=torch.float16,
            local_files_only=from_local_files
        )

        processor = AutoProcessor.from_pretrained(
            model_repo, 
            token=True,
            local_files_only=from_local_files
        )

    return model, processor


def analysis_image(
    img: Any, models: tuple, llm_model: Any, processor: Any, get_image=False, max_new_tokens: int = 2000, do_sample=False
):
    """Plot segmentation and volumes and obtain analysis from llm

    Args:
        img (Any): the image.
        models (tuple): the liver and tumor models.
        llm_model (Any): the llm model for analysis.
        processor (Any): the processor to process image.
        get_image (bool, optional): indicate if image should be returned or rendered. Defaults to False.
        max_new_tokens (int, optional): the maximum number of new tokens. Defaults to 2000.
        do_sample (bool, optional): indicate if you include sampling in the generation process. Defaults to False.

    Returns:
        str: analysis, volumes, image (with segmentation) string
    """
    masks = segment_image(img, models)
  
    overlay, volumes = identify_volumes(img, masks)

    analysis = ltp.run_medgemma_analysis(llm_model, processor, overlay, volumes, max_new_tokens, do_sample)

    img_string = ltp.plot_results(img, get_image, *masks)

    return analysis, volumes, img_string


"""## Example of output :
This is a CT scan of the abdomen showing a small liver lesion. The liver volume is 8.26 cm3, and the tumor volume is 0.53 cm3. The tumor-to-liver ratio is 6.36%. This suggests a small liver lesion, which may be a benign or malignant tumor. Further evaluation is needed to determine the nature of the lesion.

**Disclaimer:** This is an AI-generated interpretation and should not be used for medical diagnosis. A qualified healthcare professional should be consulted for any medical concerns.
"""
