# MESAHA-Net: Multi-Encoder Self-Adaptive Hard Attention Network

> **Paper:** Multi-Encoder Self-Adaptive Hard Attention Network with Maximum Intensity Projections for Lung Nodule Segmentation  
> **Journal:** Computers in Biology and Medicine, 2025  
> **Authors:** Muhammad Usman, Azka Rehman, Abd Ur Rehman, Abdullah Shahid, Tariq Mahmood Khan, Imran Razzak, Minyoung Chung, Yeong-Gil Shin  

---

## Overview

Accurate lung nodule segmentation is critical for early-stage lung cancer diagnosis.  
MESAHA-Net introduces a **multi-encoder, self-adaptive hard attention mechanism** combined with **bidirectional Maximum Intensity Projections (MIPs)** to achieve state-of-the-art performance with **lightweight computation**.

- **Triple Encoder:** Raw slice + forward MIP + backward MIP  
- **Adaptive Hard Attention:** ROI-guided, eliminating rescaling-induced errors  
- **Lightweight Design:** ~0.44M parameters with ~48 ms per slice inference  
- **High Accuracy:** Consistently outperforms Res-UNet, DEHA-Net, CMSF, and other baselines  

---

## Architecture

<p align="center">
  <img src="MESHA_Net.png" alt="MESAHA-Net Architecture" width="700"/>
</p>

MESAHA-Net integrates **contextual 2D and 3D information** via multi-input encoders.  
A **self-adaptive hard attention block** selectively amplifies nodule regions while suppressing irrelevant background features.  
This leads to more precise 3D segmentation across heterogeneous lung nodules.

---

## Results

### **LIDC-IDRI Dataset**
- **DSC:** 88.27 ± 7.42  
- **Sensitivity:** 92.88 ± 9.54  
- **PPV:** 86.95 ± 11.29  

### **LNDb Dataset**
- **DSC:** 82.17  
- **Sensitivity:** 85.96  
- **PPV:** 86.57  

### **Efficiency**
- Parameters: ~0.44M  
- Inference time: ~48 ms per slice  

MESAHA-Net **outperforms existing methods** while being faster and lighter.

<p align="center">
  <img src="plots.png" alt="Performance Comparison" width="750"/>
</p>

---

## Key Advantages

✔ Eliminates **rescaling-induced errors** with adaptive hard attention  
✔ Handles **small and heterogeneous nodules** better than prior models  
✔ Provides **robust generalization** across datasets (LIDC-IDRI, LNDb)  
✔ **Clinically feasible**: real-time inference on standard GPUs  

---

## Citation

If you use this work, please cite:

```bibtex
@article{MESAHA-Net-2025,
  title   = {Multi-Encoder Self-Adaptive Hard Attention Network with Maximum Intensity Projections for Lung Nodule Segmentation},
  author  = {Muhammad Usman and Azka Rehman and Abd Ur Rehman and Abdullah Shahid and Tariq Mahmood Khan and Imran Razzak and Minyoung Chung and Yeong-Gil Shin},
  journal = {Computers in Biology and Medicine},
  year    = {2025},
  note    = {Preprint}
}
