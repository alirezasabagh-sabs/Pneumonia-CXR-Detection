# Data, pretrained models, and licensing

## Repository code

The original code in this repository is released under the MIT License. See `LICENSE`.

## Third-party datasets

This repository does **not** redistribute medical datasets. Dataset files must be obtained from their respective sources and stored locally.

The RSNA Pneumonia Detection Challenge terms state that the de-identified imaging datasets and annotations may be used for academic research/education and other commercial or non-commercial purposes subject to the stated provisions and attribution requirements. Those terms also restrict redistribution and impose additional competition-specific obligations. Review the current official terms before commercial use. 

Official references:

- RSNA Pneumonia Detection Challenge: https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018
- RSNA/Kaggle challenge rules and terms: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules

## Pretrained models

The Patch-MIL encoder uses TorchXRayVision's DenseNet121 model family. TorchXRayVision states that licenses vary by subpackage/model, so the exact pretrained weights used by an experiment must be checked before redistribution or commercial deployment.

Official reference:

- TorchXRayVision repository/license: https://github.com/mlmed/torchxrayvision

The lung segmentation dependency used by this project is `lungs-segmentation`. Verify the exact installed release and its model-weight terms before redistribution or embedding its weights in a commercial product.

## Practical commercial checklist

Before turning this research repository into a commercial product, verify separately:

1. rights to every dataset used for training and validation;
2. rights to every pretrained model/weight checkpoint;
3. licenses of all runtime dependencies;
4. whether model redistribution is allowed;
5. whether commercial SaaS/API use is allowed;
6. medical-device/regulatory requirements in the target market;
7. privacy, security, retention, and de-identification requirements for clinical images;
8. external validation and prospective clinical evaluation requirements.

The GitHub repository itself should contain code and documentation, not patient data or private clinical annotations.
