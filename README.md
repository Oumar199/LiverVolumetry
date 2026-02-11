# LiverVolumetry

**Deep Learning for Liver and Tumor Volumetry in Preoperative Planning**  
_Estimation précise des volumes tumoraux et hépatiques pour le CHC, avec segmentation automatique et analyse avancée._

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

## Notes for Usage

If you want to use our `liver-volumetry` package follows the Usage Example section. 

However to directly analyze (recommended) a liver image using the quantized Medgemma-1.5-4b model and get all results at once, you can directly follows the steps in the RunPod Serverless API Examples section which provides access to our runpod serverless endpoint. 

---

## Usage Example 

It is possible to install our repository package and use the different functionalities. We provide below a step by step guide.

### 1. Installation (terminal only)

First clone the repository from github using `git` and enter into the directory. 
```bash
git clone https://github.com/Oumar199/LiverVolumetry.git
cd LiverVolumetry
```
To overcome version conflicts you should create an environment and install the `liver-volumetry` package in the environment.
```bash
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
pip install -e liver-volumetry
```

### 2. Loading Models

Next you can load segmentation models using our `get_models` function. This function takes as arguments path to segmentation models. They are initially put into the models directory.
```python
from liver_volumetry.interpretation.analysis import get_models

liver_model, tumor_model = get_models()
```

### 3. Loading an Image

Images are put into the images directory and corresponds to the liver image that should be segmented and analyzed. If you have a new image placed it in the images directory and then use the `load_image` function to recuperate the image. Only the image path should be passed to the `load_image` function. You can use the example image available in the images directory.

<img width="402" height="386" alt="image" src="https://github.com/user-attachments/assets/55f38b01-8b6c-4e1b-8bcb-c78ca55fa239" />


```python
from liver_volumetry.interpretation.analysis import load_image

img = load_image('images/output_liver_segmentation.jpg') # modify the path as needed
```

### 4. Segment Image

The image, `img`, is going through a segmentation process using loaded segmentation models. image and models (as a tuple) are passed to the function `segment_image`.
```python
from liver_volumetry.interpretation.analysis import segment_image

liver_mask, tumor_mask = segment_image(img, (liver_model, tumor_model))
```

### 5. Identify Volumes

Next we should calculate the liver and tumor volumes and get overlay of segmentations using the `identify_volumes` function. Image and masks (as a tuple) should be passed as arguments to the function.
```python
from liver_volumetry.interpretation.analysis import identify_volumes

overlay, volumes = identify_volumes(img, (liver_mask, tumor_mask))

print(volumes)
```

### 6. Plot Segmentations

You can plot the results of the segmentation process using the `plot_results` function. The `get_image` argument, when equal to `True`, return the plot as a string. It should be equal to `False` to enable direct rendering of the plot. 
```python
from liver_volumetry.utils import liver_tumor_pipeline_py as ltp

plot_results(img_array = img, get_image = False, liver_mask = liver_mask, tumor_mask = tumor_mask)
```

Example of plot from the initial image (Masque Foie = Liver Mask, Masque Tumeur = Tumor Mask):

<img width="946" height="308" alt="image" src="https://github.com/user-attachments/assets/ca859775-e229-453c-a350-6ce8578f3bbd" />


### 7. Analysze Outcomes

Since we cannot provide the analysis model in the repository due to the large scale of the base model of Medgemma-1.5-4b available in huggingface and that the fine-tuned model is private. You should follow the RunPod Serverless API Examples guideline to access to the analysis process of our project.

---

## RunPod Serverless API Examples (Recommended)

Here are detailed examples demonstrating how to use the RunPod serverless API with binary image encoding in base64.

### Step 1: Install the runpod library (from the terminal) 

```bash
pip install runpod==1.3.0
```

### Step 2: Import all libraries

```python
import runpod
import base64
import os
```

### Step 3: Encode Image in Base64

```python
with open('path_to_image.jpg', 'rb') as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
```

### Step 4: Send Request to RunPod API

```python
endpoint = runpod.Endpoint("jmyvktagmulnfm", api_key="rpa_5C7ARZ9TEAT21XNBGA9Q16P1H151ODBOVDDU80C92xocxf")

try:
    run_request = endpoint.run_sync(
        {
            "image": encoded_string
        },
        timeout=120
    )

    print(run_request)
except TimeoutError:
    print("Job timed out.")
```

---

## Data & Model Information

- **Dataset:** 3D-IRCAdb-01 public CT scans (20 patients)
- **Models:** Pre-trained neural networks and quantized Medgemma available under `models/`
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
- [3D-IRCAdb-01 dataset](https://www.ircad.fr/research/3dircadb/)

---

## Disclaimer

This is not a substitute for professional medical advice. Outputs should be validated with clinical experts before deployment. Do not use real patient data in this repository.
