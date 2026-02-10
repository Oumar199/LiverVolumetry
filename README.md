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

## Installation

```bash
git clone https://github.com/Oumar199/LiverVolumetry.git
cd LiverVolumetry
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
pip install -r requirements.txt
```

---

## Usage Example

```python
from livervolumetry.segmentation import UNetSegmenter
from livervolumetry.volumetry import compute_volume
from livervolumetry.medgemma import MedgemmaAnalyzer

# Segment liver and tumor
segmenter = UNetSegmenter(model_path="models/unet.pt")
mask = segmenter.segment(dicom_series_path="data/patient01.dcm")

# Compute volumes
volume = compute_volume(mask, dicom_metadata="data/patient01.dcm")

# Analyze outcome with quantized Medgemma
analyzer = MedgemmaAnalyzer(model_path="models/medgemma-1.5-4b-quantized.pt")
report = analyzer.analyze(volume)
print(report)
```

---

## Data & Model Information

- **Dataset:** 3D-IRCAdb-01 public CT scans (20 patients)
- **Models:** Pre-trained neural networks and quantized Medgemma available under `models/`
- **No real patient data** is included in the repository; all examples use synthetic or public data.

---

## Testing & Development

```bash
pytest
flake8
black .
```

---

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Maintainers

- **Oumar199** (GitHub: [@Oumar199](https://github.com/Oumar199))

---

## References

- *Apprentissage en profondeur pour la volumétrie du foie dans la planification préopératoire* (scientific paper description)
- [3D-IRCAdb-01 dataset](https://www.ircad.fr/research/3dircadb/)

---

## Disclaimer

This is not a substitute for professional medical advice. Outputs should be validated with clinical experts before deployment. Do not use real patient data in this repository.