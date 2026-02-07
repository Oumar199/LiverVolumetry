from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  
from huggingface_hub import hf_hub_download
from liver_volumetry.interpretation.analysis import *
import fasttext
import runpod  
import base64
import os  


######################
# import models : models must be placed at models\\ModelSegmentation
# liver model filename: final_model_unet_foie.h5
# tumor model filename: final_model_tumor_resunet.h5
######################

models = get_models()

llm_model, processor = get_llm_and_processor()

TARGET_DIR = "images"

######################


def handler(job):
    """
    RunPod serverless handler.

    """
    job_input = job.get("input", {}) or {}
    
    image_base64 = job_input.get("image")
    
    if not image_base64:
        return {"error": "Aucune image reçue"}

    file_name = "input_image.png"
    save_path = os.path.join(TARGET_DIR, file_name)
    
    if os.path.exists(save_path):
        
        os.remove(save_path)

    # Décodage et écriture binaire
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(image_base64))

    try:
        
        img = load_image(save_path)
        
        analysis = analysis_image(img)

        return {
            "status": "success",
            "analysis": analysis,
        }

    except Exception as e:
        print(f"[Handler] ❌ Error during generation: {e}")
        return {
            "status": "error",
            "error": str(e),
        }

runpod.serverless.start({"handler": handler})


