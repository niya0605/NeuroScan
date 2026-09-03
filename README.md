# NeuroScan

An AI-based web application that analyzes brain MRI scans and provides tumor classification results with confidence scores and visual insights.

## Overview

NeuroScan is a web-based MRI analysis dashboard built to demonstrate how a machine learning model can be integrated into an interactive application.

The application allows users to upload brain MRI scans, view model predictions, explore confidence scores, visualize Grad-CAM attention maps, process multiple scans, and download a PDF report of the results.

> **Note:** NeuroScan is a research and educational project. The model results are not intended for clinical diagnosis or medical decision-making.

## Features

- **MRI Scan Analysis** — Upload brain MRI images for model-based classification
- **Confidence Scores** — View the model's confidence for each prediction
- **Probability Distribution** — See the predicted probability across all four classes
- **Grad-CAM Visualization** — View an attention map showing areas that influenced the model
- **Batch Processing** — Upload and analyze multiple MRI scans in one session
- **PDF Reports** — Export analyzed scans and their results as a PDF
- **Dark & Light Mode** — Switch between themes
- **Error Handling** — Invalid or unreadable images are handled with clear feedback
- **Responsive Interface** — Designed to work across desktop, tablet, and mobile screens

## Tech Stack

### Frontend

- React
- TypeScript
- Vite
- Tailwind CSS
- shadcn/ui
- Wouter
- tRPC Client

### Backend

- Node.js
- Express.js
- tRPC

### Machine Learning

- TensorFlow / Keras
- EfficientNet
- 224 × 224 RGB image input
- 4-class softmax classification

### Other Tools

- Drizzle ORM
- MySQL
- jsPDF

## Model Classes

The model is configured for four classes:

1. **Glioma**
2. **Meningioma**
3. **No Tumor**
4. **Pituitary**

The model receives RGB images resized to **224 × 224 pixels** and produces a four-class probability distribution.

## Project Structure

```text
NEUROSCAN/
├── client/
│   ├── src/
│   │   ├── components/
│   │   ├── contexts/
│   │   ├── hooks/
│   │   ├── lib/
│   │   ├── pages/
│   │   └── index.css
│   └── index.html
│
├── server/
│   ├── inference.ts
│   ├── routers.ts
│   ├── index.ts
│   └── _core/
│
├── shared/
│   └── model.ts
│
├── model/
│   └── brain_tumor_model.keras
│
├── public/
│   └── brainimg.png
│
├── package.json
├── pnpm-lock.yaml
├── vite.config.ts
├── tsconfig.json
└── README.md

## Using the Application

1. Open the NeuroScan dashboard.
2. Upload one or more brain MRI images.
3. Wait for the model to process the scans.
4. Review the predicted class and confidence score.
5. Check the probability distribution for all four classes.
6. Switch to **Grad-CAM** to view the model's attention map.
7. Download a PDF report for completed analyses.

## Model Limitations

The included model is provided as a research artifact and has known classification limitations.

It was retrained on the public [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (glioma/meningioma/notumor/pituitary, ~7k images) using transfer learning on an EfficientNetB0 backbone at 300x300 resolution with the full backbone fine-tuned, evaluated on a held-out test set of 1,600 images never seen during training:

| Class | Precision | Recall | F1 |
| --- | --- | --- | --- |
| glioma | 1.00 | 0.72 | 0.84 |
| meningioma | 0.87 | 0.92 | 0.90 |
| notumor | 0.92 | 1.00 | 0.96 |
| pituitary | 0.88 | 1.00 | 0.93 |

Overall accuracy: **91%**. See `model/training_report.json` for the full classification report and confusion matrix.

Glioma remains the weakest class (72% recall) — when the model says glioma it's always right (100% precision), but it still misses about 1 in 4 real glioma cases, mostly calling them meningioma instead. That distinction is genuinely difficult from a single 2D slice without clinical context, and one radiologists themselves don't always agree on from imaging alone. On some real-world glioma images, the model is confidently wrong (>99% on the wrong class) rather than uncertain — retraining reduced how often it's wrong overall, but made it more confident, not less, on the cases it still misses.

These results indicate that the current model should **not** be treated as clinically accurate.

Training code is in `scripts/train_model.py`, `scripts/continue_training.py`, and `scripts/train_highres.py`. Further improvement on glioma specifically would likely require more/higher-quality labeled data or a fundamentally different approach (e.g. an ensemble, or additional input beyond a single slice) rather than more training on this dataset.

The application intentionally displays the model's actual output rather than modifying or fabricating predictions.

## Explainability

NeuroScan includes Grad-CAM visualization to provide an additional view of model behavior.

The Grad-CAM output highlights regions of the MRI image that contributed most strongly to the model's prediction.

This is intended to make the model's output easier to inspect and understand, but the visualization should not be interpreted as a medical finding.

## Safety Disclaimer

NeuroScan is intended for **research, demonstration, and educational purposes only**.

The predictions, confidence scores, probability distributions, and Grad-CAM visualizations:

- are generated by a machine learning model
- may be incorrect
- are not medical diagnoses
- should not be used to make medical decisions

Always consult a qualified medical professional for medical interpretation of MRI scans.

