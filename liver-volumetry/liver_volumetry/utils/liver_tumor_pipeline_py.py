# %% [code]
# liver_tumor_pipeline.py
"""Utils
---
Some functions to help us segment, plot, analysis, ...
"""
from liver_volumetry import *
from typing import Tuple
import os


# ======================================================
# MODELS LOADING
# ======================================================

def load_segmentation_models(liver_path: str, tumor_path: str):
    class CpuSafeDropout(Dropout):
        def __init__(self, rate=0.0, noise_shape=None, seed=None, **kwargs):
            super().__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)
        
        def call(self, inputs, training=None):
            return inputs  # SIMPLEMENT retourne input
    
    print("📥 CPU Loading...")
    custom_objects = {'Dropout': CpuSafeDropout}
    
    model_liver = load_model(liver_path, compile=False, custom_objects=custom_objects)
    model_tumor = load_model(tumor_path, compile=False, custom_objects=custom_objects)
    
    print("✅ CPU MODELS READY")
    return model_liver, model_tumor

def load_medgemma_4bit():
    """
    Load 4-bit quantized MedGemma 1.5 4B 4-bit + base model's processor.
    """
    model_repo = "Metou/MedGemma-1.5-4B"
    sub_dir = "bismedgemma-4bit"

    path = snapshot_download(repo_id=model_repo, allow_patterns=f"{sub_dir}/*")

    processor = AutoProcessor.from_pretrained(path, subfolder=sub_dir, use_fast=False)

    model = AutoModelForImageTextToText.from_pretrained(
        path, 
        subfolder=sub_dir, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    return model, processor

def load_medgemma():
    """
    Load MedGemma 1.5 4B + processor.
    """
    model_repo = "google/medgemma-1.5-4b-it"

    path = snapshot_download(repo_id=model_repo)

    processor = AutoProcessor.from_pretrained(path, token=True)

    model = AutoModelForImageTextToText.from_pretrained(
        path, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    return model, processor

# ======================================================
# IMAGE PREPROCESSING
# ======================================================

def load_and_preprocess_image(image_path: str, target_size=(256, 256)):
    """
    Load and preprocess grayscale image.
    """
    img = load_img(image_path, target_size=target_size, color_mode="grayscale")
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


# ======================================================
# SEGMENTATION
# ======================================================

def run_segmentation(img_array, model_liver, model_tumor, threshold=0.5):
    """
    Segment liver + tumor.
    """
    liver_pred = model_liver.predict(img_array, verbose=0)
    tumor_pred = model_tumor.predict(img_array, verbose=0)

    liver_mask = (liver_pred > threshold).astype(np.uint8)
    tumor_mask = (tumor_pred > threshold).astype(np.uint8)

    return liver_mask, tumor_mask


# ======================================================
# VISUALIZATION / OVERLAY
# ======================================================


def build_overlay(img_array, liver_mask, tumor_mask):
    """
    Create an RGB image with overlay added :
    - fond : CT
    - foie : blanc
    - tumeur : gris
    """
    base = img_array[0, :, :, 0].copy()

    overlay = base.copy()
    overlay[liver_mask[0, :, :, 0] == 1] = 1.0
    overlay[tumor_mask[0, :, :, 0] == 1] = 0.5
    overlay = np.clip(overlay, 0, 1)

    overlay_rgb = np.stack([overlay] * 3, axis=-1)
    overlay_uint8 = (overlay_rgb * 255).astype(np.uint8)

    return Image.fromarray(overlay_uint8)


def plot_results(img_array, get_image, liver_mask, tumor_mask):
    """
    Fast visualization with matplotlib and return string format.
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Image CT")
    plt.imshow(img_array[0, :, :, 0], cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Liver mask")
    plt.imshow(liver_mask[0, :, :, 0], cmap="jet")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Tumor mask")
    plt.imshow(tumor_mask[0, :, :, 0], cmap="hot")
    plt.axis("off")

    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout()

    if get_image:

        # Save the figure to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")

        # Close the plot to free memory and prevent display in interactive environments
        plt.close()

        # Get the image data as a string (bytes)
        buf.seek(0)
        img_bytes = buf.read()

        # Encode the bytes to a base64 string
        img_string = base64.b64encode(img_bytes).decode("utf-8")

        # Close the buffer
        buf.close()

        return img_string

    else:

        plt.show()


# ======================================================
# VOLUME COMPUTATION
# ======================================================


def compute_volumes(
    liver_mask, tumor_mask, pixel_area_mm2=0.37, slice_thickness_mm=1.5
):
    """
    Calculate liver / tumor volume and tumoral ratio.
    """
    liver_pixels = np.count_nonzero(liver_mask)
    tumor_pixels = np.count_nonzero(tumor_mask)

    liver_volume_cm3 = (liver_pixels * pixel_area_mm2 * slice_thickness_mm) / 1000
    tumor_volume_cm3 = (tumor_pixels * pixel_area_mm2 * slice_thickness_mm) / 1000

    ratio = 100 * tumor_volume_cm3 / liver_volume_cm3 if liver_volume_cm3 > 0 else 0.0

    return {
        "liver_volume_cm3": liver_volume_cm3,
        "tumor_volume_cm3": tumor_volume_cm3,
        "tumor_ratio_percent": ratio,
    }


# ======================================================
# MEDGEMMA ANALYSIS
# ======================================================


def build_medgemma_prompt(volumes: dict):
    """
    Prompt multimodal MedGemma (image + clinical text).
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Abdominal CT slice with liver and tumor segmentation.\n\n"
                        f"Liver volume: {volumes['liver_volume_cm3']:.2f} cm3\n"
                        f"Tumor volume: {volumes['tumor_volume_cm3']:.2f} cm3\n"
                        f"Tumor-to-liver ratio: {volumes['tumor_ratio_percent']:.2f}%\n\n"
                        "Provide a concise clinical interpretation."
                    ),
                },
            ],
        }
    ]


def run_medgemma_analysis(
    model,
    processor,
    overlay_image: Image.Image,
    volumes: dict,
    max_new_tokens: int = 2000,
    do_sample: bool = False
):
    """
    Generating clinical reports with MedGemma.
    """
    messages = build_medgemma_prompt(volumes)

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(text=prompt, images=overlay_image, return_tensors="pt").to(
        model.device
    )

    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample
        )

    return processor.decode(output[0], skip_special_tokens=True)
