# LiverVolumetry

**Deep Learning for Liver and Tumor Volumetry in Preoperative Planning**  
_Estimation précise des volumes tumoraux et hépatiques pour le CHC, avec segmentation automatique et analyse avancée._

<img width="1432" height="953" alt="image" src="https://github.com/user-attachments/assets/7a5feef7-e7da-43c6-a4fc-12b267dba18b" />




## Description

LiverVolumetry is an open-source Python package for automatic segmentation and volumetric analysis of the liver and tumors, based on deep learning. Developed within the context of Senegalese medical workflows, the repository implements and extends state-of-the-art algorithms for preoperative planning in hepatocellular carcinoma (HCC) and related pathologies. It facilitates reproducible, scalable, and clinically relevant volumetry from DICOM images, with support for data normalization, robust experimentation, and advanced model integration.

---

## Problem Statement: Liver Volumetry Challenges

Accurate tumor and liver volume estimation is critical for optimizing surgical planning in HCC, a common and often fatal primary liver cancer. Manual volumetry remains time-consuming, error-prone, and operator-dependent—especially with complex cases or noisy images—limiting reproducibility and standard clinical integration.

---

## Proposed Solution: Deep Learning-Based Volumetry

This project leverages deep learning to automate liver and tumor segmentation from DICOM images, followed by precise volumetric calculations. The workflow minimizes human error, enhances quantitative reliability, and provides an innovative decision-support tool for clinicians.

**Recent update:**  
We added automatic outcome analysis of the volumetry model using a quantized 4-bit Medgemma 1.5-4b model. This large language model offers efficient, on-device clinical report generation and result summarization, supporting deployment in resource-limited settings.

---

## Methodology

### Overall Workflow

- **Data Preprocessing:** DICOM volumes are converted to 2D JPEG series for deep learning compatibility, with intensity normalization.
- **Segmentation:** Multiple CNN architectures (U-Net, RESU-Net, FCN, Attention U-Net, U-Net/ResNet-50+CBAM) are implemented. U-Net variants proved most robust.
- **Volumetric Calculation:** Segmentation masks and DICOM metadata are combined to calculate liver and tumor volumes using voxel dimensions (voxel_size_x, voxel_size_y, voxel_size_z).  
- **Experimentation:** Models are trained and validated using the public 3D-IRCAdb-01 dataset (20 patients, 75% with hepatic tumors).  
- **Augmentation & Optimization:** Intensity normalization, random splits (70/15/15 for training/validation/test), and data augmentation (rotation, zoom, flip, contrast) are employed. Hyperparameters are optimized via Optuna.

### Model Comparison

- **Baseline Models:** U-Net, Attention U-Net, U-Net+ResNet50+CBAM, ResuNet, FCN
- **Evaluation Metrics:** Dice coefficient, accuracy, binary cross-entropy loss, volumetric RMSE

### Medgemma 1.5-4b Integration

- **Contribution:** The repository now supports outcome analysis via the Medgemma 1.5-4b model quantized to 4-bit, enabling fast and memory-efficient clinical summarization.  
- **Workflow:** Results from the volumetry pipeline are passed to Medgemma for automated interpretation and report generation, facilitating clinical adoption.
- **MedGemma Liver Tumor Volume Interpretation Template Example** For multimodal input with segmentation volumes:

```json
[
  {
    "role": "user",
    "content": [
      {
        "type": "image"
      },
      {
        "type": "text",
        "text": "Abdominal CT slice with liver and tumor segmentation.\n\nLiver volume: 8.26 cm³\nTumor volume: 0.53 cm³\nTumor-to-liver ratio: 6.36%\n\nProvide a concise clinical interpretation."
      }
    ]
  }
]
```

## MedGemma's Contribution to Liver Volumetry

MedGemma has greatly enhanced the accuracy and efficiency of liver volumetry through its advanced integration into the existing framework. The following outlines its significant contributions:

1. **Functional Contribution**: MedGemma provides real-time analysis and insights, allowing for quicker decision-making during liver assessment. Its algorithms improve measurement precision, directly impacting patient outcomes.

2. **Methodological Value**: Integrating MedGemma into our pipeline enables the use of state-of-the-art techniques for volumetric measurements. This ensures consistency and reliability in the methodologies applied across different patient datasets.

3. **Advantages Over Developing a Custom LLM**: Building a custom Large Language Model (LLM) from scratch often requires substantial resources in terms of time, data, and expertise. MedGemma offers a ready-to-use solution that leverages existing research and algorithms, thus reducing development overhead and expediting implementation.

4. **Impact on the Overall Pipeline**: The integration of MedGemma has streamlined our workflow, allowing for seamless data input and output. This not only enhances productivity but also leads to more robust and reproducible results, improving the overall quality of liver volumetry assessments in clinical practice.

In summary, the MedGemma integration represents a significant advancement in liver volumetry, combining functional, methodological, and practical advantages that elevate our analytical capabilities and patient care standards.

---

## Key Results

- **U-Net Outperformed Other Architectures:**  
  - Dice coefficient: 0.98 (liver), 0.90 (tumors)
  - Volumetric RMSE: 6.41 cm³ (tumors), 20.95 cm³ (liver)
- **Reliability & Efficiency:** U-Net offers consistent, reproducible results—reducing practitioner workload and improving clinical decision-making.
- **Model Complexity:** Complex models (e.g., Attention U-Net or U-Net+ResNet+CBAM) did not significantly improve generalization, especially with limited data.

---

## Clinical Implications

- Offers a rapid, precise alternative to manual volumetry.
- Supports liver surgery, radiotherapy, and transplantation with improved residual liver volume estimation.
- Automated segmentation can optimize treatment margins and radiologist workflow.

---

## Limitations & Future Work

- Small dataset size and segmentation errors from image variability or anatomy.
- Upcoming research will integrate new regional medical data to boost robustness.
- Architectural improvements (attention, transformers, hybrid 2D/3D models) are planned.

---

## 🚀 Quick Start & Usage

You can use the **Liver Volumetry** project in two ways, depending on your needs:

### 1. Direct Analysis (Recommended)

To analyze a liver image and retrieve all results instantly, we recommend using a **RunPod Serverless API**. This method utilizes a quantized **Medgemma-1.5-4b** model for high-performance inference without local setup. We provide in this repository all necessary elements to deploy a Runpod Serverless API from Github.

👉 **Go to**: [Deploy workers from GitHub](https://docs.runpod.io/serverless/workers/github-integration) and follow the deployment process.

### 2. Local Package Integration

If you prefer to integrate the core logic into your own environment, you can install and use the standard package.

👉 **Go to**: [Usage Guide](https://github.com/Oumar199/LiverVolumetry/edit/main/README.md#-usage-guide)

---

## 🧪 Quick Test (Google Colab)
For a zero-setup experience, you can run the full analysis pipeline in one click (consider moving to t4 Tesla GPU for faster execution and restarting the session after executing the first cell):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Oumar199/LiverVolumetry/blob/main/liver_tumor_volumetry_analysis.ipynb)

---

## 🛠 Usage Guide

Follow these steps to install the `liver-volumetry` package and run a complete segmentation and analysis pipeline locally.

### 1. Installation

We recommend using a virtual environment to avoid dependency conflicts.

```bash
# Clone the repository
git clone https://github.com/Oumar/LiverVolumetry.git
cd LiverVolumetry

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\activate # Windows

# Install the package in editable mode
pip install -e liver-volumetry
```

### 2. Loading Segmentation Models

The pipeline uses pre-trained models that we are going do download from huggingface and place in the `models/` directory by default.
```python
from liver_volumetry.interpretation.analysis import get_models,  download_segmentation_models

# downloading models our huggingface account and getting the path
path = download_segmentation_models()

# Loads both liver and tumor segmentation models
liver_model, tumor_model = get_models(path)

```

### 3. Image Preprocessing

Place your CT scans or medical images in the `images/` directory. Use the `load_image` helper to prepare them for analysis.

```python
from liver_volumetry.interpretation.analysis import load_image

img = load_image('images/output_liver_segmentation.jpg') # modify the path as needed
```

> **Note:** An example image is provided in the repository for testing.

<img width="265" height="258" alt="image" src="https://github.com/user-attachments/assets/0f65c42d-ed57-428f-9ff1-ec107adc4e08" />



### 4. Segmentation Pipeline

Process the image through the loaded models to generate binary masks for both the liver and potential tumors.
```python
from liver_volumetry.interpretation.analysis import segment_image

# Returns a tuple of (liver_mask, tumor_mask)
liver_mask, tumor_mask = segment_image(img, (liver_model, tumor_model))

```

### 5. Volumetry & Overlay Generation

Calculate the precise volumes and generate an overlay visualization.
```python
from liver_volumetry.interpretation.analysis import identify_volumes

# 'volumes' contains the calculated cubic measurements
overlay, volumes = identify_volumes(img, (liver_mask, tumor_mask))

print(f"Calculated Volumes: {volumes}")

```

### 6. Visualization

To render the results directly in your notebook or UI, use the `plot_results` function.
```python
from liver_volumetry.utils import liver_tumor_pipeline_py as ltp

# Set get_image=False for direct rendering
ltp.plot_results(
    img_array=img, 
    liver_mask=liver_mask, 
    tumor_mask=tumor_mask, 
    get_image=False
)

```

**Expected Output** (Masque Foie: Liver Mask | Masque Tumeur: Tumor Mask):

<img width="943" height="307" alt="image" src="https://github.com/user-attachments/assets/f3164299-6f08-41c8-8a32-5710d4163e5d" />



### 🩺 AI-Powered Clinical Analysis

The final clinical interpretation is powered by the quantized **Medgemma-1.5-4b**  model. While the segmentation models are included locally, the analysis weights must be downloaded separately or accessed via API.

#### 1. Local Analysis (Advanced Users)
To run the analysis locally, download the quantized weights from our **Hugging Face** profile.

```python
from liver_volumetry.interpretation.analysis import analysis_image
from liver_volumetry.utils import liver_tumor_pipeline_py as ltp

# 1. Load the quantized Medgemma model and processor
llm_model, processor = ltp.load_medgemma_4bit()

# 1. if you are planning to use the base Medgemma model and processor
# uncomment the three following lines
# from huggingface_hub import login
# login() # log into your huggingface account 
# llm_model, processor = ltp.load_medgemma()

# 2. Run full pipeline (Segmentation + Volumetry + AI Interpretation)
models = (liver_model, tumor_model)
analysis, volumes, img_string = analysis_image(img, models, llm_model, processor, get_image=True)

# 3. Extract and display the medical interpretation
analysis = analysis.split("\nmodel\n", 1)[1]
print(f"--- Clinical Insight ---\n{analysis}")
```

#### 📝 Example Output

> **Note:** The following is an example of the AI-generated interpretation:
>
> "This is a CT scan of the abdomen showing a small liver lesion. The liver volume is **8.26 cm³**, and the tumor volume is **0.53 cm³**. The tumor-to-liver ratio is **6.36%**. This suggests a small liver lesion, which may be a benign or malignant tumor. Further evaluation is needed."

> [!CAUTION]
> **Medical Disclaimer:** This analysis is **AI-generated** and intended for research purposes only. It must not be used for medical diagnosis. Always consult a qualified healthcare professional for medical concerns.

#### 📄 Medical Report Generation

Once the analysis is complete, you can generate a comprehensive PDF medical report. This process involves two steps: first, saving the segmented image locally, and then compiling the final document.

```python
from liver_volumetry.utils.save_image import save_base64_image
from liver_volumetry.interpretation.prepare_report import generate_medical_report

# 1. Save the segmented image locally (converts base64 string from analysis_image)
image_path = save_base64_image(img_string)

# 2. Generate the final medical report as a PDF
generate_medical_report(
    output_pdf="liver_report.pdf", 
    patient_name="Jean Dupont", 
    seg_path=image_path, 
    clinical_analysis=analysis
)
```

**Expected Report** 

<img width="488" height="678" alt="image" src="https://github.com/user-attachments/assets/859b6c32-c675-43ae-99f3-9064c8e150e5" />



---

## Data & Model Information

- **Dataset:** 3D-IRCAdb-01 public CT scans (20 patients)
- **Models:** Pre-trained neural networks (`Metou/ModelSegmentation`) and quantized **Medgemma-1.5-4b** (`Metou/MedGemma-1.5-4B`) available in our huggingface repositories.
- **No real patient data** is included in the repository; all examples use synthetic or public data.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Maintainers

- **Oumar199** (GitHub: [@Oumar199](https://github.com/Oumar199))
- **ms-dl** (Github: [@ms-sl](https://github.com/ms-dl))
- **CheikhYakhoubMAAS** (Github: [@CheikhYakhoubMAAS](https://github.com/CheikhYakhoubMAAS))
- **MamadouBousso** (Github: [@MamadouBousso](https://github.com/MamadouBousso))
- **Aby1diallo** (Github: [@Aby1diallo](https://github.com/Aby1diallo))

---

## References

- *Apprentissage en profondeur pour la volumétrie du foie dans la planification préopératoire* 
- [3D-IRCAdb-01 dataset](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/)

---

## Disclaimer

This is not a substitute for professional medical advice. Outputs should be validated with clinical experts before deployment. Do not use real patient data in this repository.

## Citation
```bibtex
@misc{snaimasters2026livervolumetry,
  title={Liver Volumetry},
  author={Metou SANGHE, Aby DIALLO, Mamadou BOUSSO, Oumar KANE, Cheikh Yakhoub MAAS},
  howpublished={https://github.com/Oumar199/LiverVolumetry},
  year={2026}
}
```
