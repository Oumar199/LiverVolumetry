from liver_volumetry.interpretation.analysis import (
    get_models,
    get_llm_and_processor,
    load_image,
    analysis_image,
)
import runpod
import base64
import os


######################
# import models : models must be placed at models/ModelSegmentation
# liver model filename: final_model_unet_foie.h5
# tumor model filename: final_model_tumor_resunet.h5
######################

models = get_models("/app/models/ModelSegmentation")

llm_model, processor = get_llm_and_processor(
    model_repo="/runpod-volume/medgemma_analysis_model",
    base_repo="/runpod-volume/medgemma_base_model",
)

TARGET_DIR = "/app/images"

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

        analysis, volumes, img_string = analysis_image(
            img, models, llm_model, processor, get_image=True
        ).split("\nmodel\n", 1)[1]

        return {
            "status": "success",
            "liver_volume_cm3": volumes["liver_volume_cm3"],
            "tumor_volume_cm3": volumes["tumor_volume_cm3"],
            "tumor_ratio_percent": volumes["tumor_ratio_percent"],
            "analysis": analysis,
            "img_string": img_string,
        }

    except Exception as e:
        print(f"[Handler] ❌ Error during generation: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


runpod.serverless.start({"handler": handler})
