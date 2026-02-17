import logging
import os
from typing import Tuple, Optional
import runpod
import base64

# Logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables with lazy initialization
TARGET_DIR = "/app/images"

from liver_volumetry.interpretation.analysis import get_models, download_segmentation_models
from liver_volumetry.interpretation.analysis import get_llm_and_processor
from liver_volumetry.interpretation.analysis import load_image, analysis_image
        

path = download_segmentation_models(local_directory="/app/models")
models = get_models(path)
llm_model, processor = get_llm_and_processor(
    model_repo="/runpod-volume/medgemma_analysis_model"
)

def handler(job):
    
    job_input = job.get("input", {}) or {}
    image_base64 = job_input.get("image")
    max_new_tokens = job_input.get("max_new_tokens", 2000)
    do_sample = job_input.get("do_sample", True)
    
    if not image_base64:
        logger.warning("⚠️ No image received")
        return {"error": "Aucune image reçue"}
    
    file_name = "input_image.png"
    save_path = os.path.join(TARGET_DIR, file_name)
    
    # Clean previous image
    if os.path.exists(save_path):
        os.remove(save_path)
    
    try:
        with open(save_path, "wb") as f:
            f.write(base64.b64decode(image_base64))
    except Exception as e:
        logger.error(f"❌ Image decode failed: {e}")
        return {"status": "error", "error": "Invalid image data"}
    
    try:
        logger.info("🔬 Starting image analysis...")
        
        img = load_image(save_path)
        analysis, volumes, img_string = analysis_image(img, models, llm_model, processor, get_image=True)
        analysis = analysis.split("\nmodel\n", 1)[1]
        
        logger.info("✅ Analysis completed successfully")
        return {
            "status": "success",
            "liver_volume_cm3": volumes["liver_volume_cm3"],
            "tumor_volume_cm3": volumes["tumor_volume_cm3"],
            "tumor_ratio_percent": volumes["tumor_ratio_percent"],
            "analysis": analysis,
            "img_string": img_string,
        }
        
    except Exception as e:
        logger.error(f"❌ Analysis FAILED: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "status": "error",
            "error": str(e),
        }

# Start serverless
runpod.serverless.start({"handler": handler})
