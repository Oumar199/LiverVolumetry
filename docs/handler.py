import logging
import os
from typing import Tuple, Optional
import runpod
import base64

# Logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # SILENCE TF
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Global variables with lazy initialization
_models = None
_llm_model = None
_processor = None
TARGET_DIR = "/app/images"

def safe_load_models() -> Tuple[Optional[object], Optional[object]]:
    """Safe model loading with GPU isolation and full logging."""
    global _models
    
    if _models is not None:
        logger.info("✅ Models already loaded (using cache)")
        return _models
    
    logger.info("🔄 Starting SAFE model initialization...")
    logger.info(f"CUDA before: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    # CRITICAL: Complete GPU isolation
    original_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # NO GPU during loading
    
    try:
        from liver_volumetry.interpretation.analysis import get_models
        logger.info("📥 Loading models from /app/models/ModelSegmentation")
        _models = get_models("/app/models/ModelSegmentation")
        logger.info("✅ Models loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Model loading FAILED: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        _models = None
        
    finally:
        os.environ['CUDA_VISIBLE_DEVICES'] = original_gpu
        logger.info(f"GPU restored: {original_gpu}")
    
    return _models

def safe_load_llm() -> Tuple[Optional[object], Optional[object]]:
    """Safe LLM loading with logging."""
    global _llm_model, _processor
    
    if _llm_model is not None and _processor is not None:
        logger.info("✅ LLM already loaded (using cache)")
        return _llm_model, _processor
    
    try:
        logger.info("📥 Loading LLM models...")
        from liver_volumetry.interpretation.analysis import get_llm_and_processor
        _llm_model, _processor = get_llm_and_processor(
            model_repo="/runpod-volume/medgemma_analysis_model",
            base_repo="/runpod-volume/medgemma_base_model",
        )
        logger.info("✅ LLM loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ LLM loading FAILED: {str(e)}")
        _llm_model = _processor = None
        
    return _llm_model, _processor

def handler(job):
    """Main RunPod serverless handler with lazy loading."""
    logger.info("🎯 New job received")
    
    # LAZY LOAD: Initialize only on first request
    models = safe_load_models()
    if models is None:
        logger.error("❌ Cannot process job: models failed to load")
        return {"status": "error", "error": "Segmentation models unavailable"}
    
    llm_model, processor = safe_load_llm()
    if llm_model is None:
        logger.error("❌ Cannot process job: LLM failed to load")
        return {"status": "error", "error": "LLM models unavailable"}
    
    job_input = job.get("input", {}) or {}
    image_base64 = job_input.get("image")
    
    if not image_base64:
        logger.warning("⚠️ No image received")
        return {"error": "Aucune image reçue"}
    
    file_name = "input_image.png"
    save_path = os.path.join(TARGET_DIR, file_name)
    
    # Clean previous image
    if os.path.exists(save_path):
        os.remove(save_path)
    
    # Decode and save image
    logger.info("🖼️ Saving input image...")
    try:
        with open(save_path, "wb") as f:
            f.write(base64.b64decode(image_base64))
        logger.info("✅ Image saved")
    except Exception as e:
        logger.error(f"❌ Image decode failed: {e}")
        return {"status": "error", "error": "Invalid image data"}
    
    try:
        logger.info("🔬 Starting image analysis...")
        from liver_volumetry.interpretation.analysis import load_image, analysis_image
        
        img = load_image(save_path)
        result = analysis_image(img, models, llm_model, processor, get_image=True)
        analysis, volumes, img_string = result.split("\nmodel\n", 1)[1]
        
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
