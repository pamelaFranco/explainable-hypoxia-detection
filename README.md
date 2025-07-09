# 🧠 Explainable Hypoxia Detection using Synthetic Vascular Models and Machine Learning

This project integrates a modified version of the **Vascusynth** simulator, **oxygen reaction-diffusion modeling**, and an **explainable machine learning pipeline** to study hypoxia in synthetic vascular networks.

It is part of a research effort to develop interpretable AI tools for biomedical image analysis and personalized healthcare, particularly in the context of **tumor hypoxia**, **vascular diseases**, and **tissue oxygenation** modeling.

![Figure 1](https://raw.githubusercontent.com/pamelaFranco/explainable-hypoxia-detection/main/Figure_1.png)
---

## 📁 Project Structure

/Binary VascuSynth/ ← Modified Vascusynth executable for vascular image simulation
/Database/ ← Synthetic vascular datasets (.tiff (CT-simualted images), .mat (PO2 map), .xml (tree structure))
/Oxygen Reaction-Diffusion Equation (FDM)/ ← MATLAB code for simulating oxygen transport using finite difference method
/Machine Learning Pipeline/ ← Python scripts for training and interpreting ML models


---

## 🧪 Objectives

- Simulate realistic vascular architectures with **Vascusynth**.
- Model **oxygen diffusion** using the **Finite Difference Method (FDM)**.
- Build explainable **ML models** to classify and interpret oxygenation patterns.
- Provide insights into the **biophysical mechanisms** behind tissue hypoxia.

---

## 📊 Key Methods

- **Synthetic image generation** using Vascusynth.
- **Numerical modeling** of oxygen transport using PDEs and FDM in MATLAB.
- **Data labeling** based on hypoxic status.
- **Machine Learning** includes BorderlineSMOTE for class balancing, feature selection, and logistic regression, reporting the F1 score via stratified cross-validation
- **Interpretability** via LIME values to identify key image-based biomarkers.

---

## 🛠 Requirements

### C++
- `CMake` (for building the modified VascuSynth code)
- C++11-compatible compiler (e.g., `g++`, `clang`)
- `OpenCV` (optional, for image output or visualization support)
  

### MATLAB
- Required for executing the oxygen diffusion model.

  ### Python (3.10)
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `lime`

---

## 🚀 Getting Started

1. **Simulate vascular images** with the Vascusynth binary.
2. **Run oxygen diffusion model** in MATLAB using the FDM scripts.
3. **Extract features and labels** from output `.mat` and `.xml` files.
4. **Train ML models** using the scripts in the `Machine Learning Pipeline`.
5. **Visualize and interpret** model outputs with LIME.

---

## 📌 Publications & Presentations

This repository is part of ongoing work presented at:
 
- 15th International Conference on Pattern Recognition Systems, ICPRS-2025, 1-4 Dec, 2025 (Vina del Mar, Chile)

---

## 👩‍💻 Author

**Pamela Franco Leiva, Ph.D.**  
Energy Transformation Center, Universidad Andrés Bello  
📧 pamela.franco@unab.cl  
🌐 [ORCID](https://orcid.org/0000-0001-7629-3653)

**Cristian Montalba, B.Sc.**  
Researcher, iHEALTH – Millennium Institute for Intelligent Healthcare Engineering  
Biomedical Imaging Center & Radiology Department  
Pontificia Universidad Católica de Chile  
🌐 [ORCID](https://orcid.org/0000-0003-3370-0233)

**Raúl Caulier-Cisterna, Ph.D.**  
Department of Informatics and Computing  
Universidad Tecnológica Metropolitana  
🌐 [ORCID](https://orcid.org/0000-0002-0125-485X)

**Jorge Vergara, Ph.D.**  
Department of Informatics and Computing  
Universidad Tecnológica Metropolitana  
🌐 [ORCID](https://orcid.org/0000-0001-6699-4181)

**Ignacio Espinoza, Ph.D.** *(Corresponding author)*  
Physics Institute  
Pontificia Universidad Católica de Chile  
📧 igespino@uc.cl  
🌐 [ORCID](https://orcid.org/0000-0003-2400-4498)

---

## 📜 License

MIT License.  
Please acknowledge the original **Vascusynth** repository if using its components:  
https://github.com/sfu-mial/VascuSynth

---

## 🤝 Acknowledgements

This work is supported by the Agencia Nacional de Investigación y Desarrollo de Chile (ANID), through the FONDECYT Iniciación 2025 N° 11250867 grant and the Competition for Research Regular Projects, year 2023, code LPR23-17, Universidad Tecnológica Metropolitana. PF was funded by ANID FONDECYT de Postdoctorado 2024 N°3240078.



