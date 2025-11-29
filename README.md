# Skin Lesion Classification Using SSL, LoRA-Adaptive CLIP and Hybrid Multi-Modal Learning

This repository presents a hybrid deep-learning pipeline for state-of-the-art skin lesion classification, combining:

Self-Supervised Learning (SSL) for domain-specific feature extraction

LoRA-based domain adaptation of CLIP

Multi-modal contrastive learning

End-to-end hybrid training on the SIIM-ISIC dataset

The project synergizes modern parameter-efficient tuning methods with SSL to achieve strong performance (>93% accuracy target).

## 📌 Project Objectives
🔹 Objective 1 — Self-Supervised Learning (SSL) for EfficientNet-B2

Pre-train EfficientNet-B2 on the full 33,000-image unlabeled dermoscopy dataset using frameworks like DINO or MoCo.

Purpose: Learn domain-specific “visual grammar” of skin lesions without labels.

Impact: Stronger initialization → improved accuracy + faster downstream convergence.

🔹 Objective 2 — Domain Adaptation of CLIP via LoRA

Fine-tune OpenAI CLIP ViT-B/32 using Low-Rank Adaptation (LoRA) on the SIIM-ISIC data.

Purpose: Efficiently specialize CLIP’s attention layers for dermatology.

Impact: Major accuracy improvement at low compute cost, without full retraining.

🔹 Objective 3 — SSL-EfficientNet-B2 in Contrastive Learning Pipeline

Replace the ImageNet-pretrained EfficientNet with the SSL-pretrained EfficientNet-B2 inside the multimodal contrastive learning setup.

Purpose: Improve alignment of image, metadata, and textual embeddings.

Impact: Stronger multimodal representation learning using SSL + CLIP specialization.

🔹 Objective 4 — End-to-End Hybrid Model Training

Combine the LoRA-adapted CLIP and SSL-enhanced EfficientNet-B2 into one unified classifier.

Purpose: Fuse the best multimodal and visual domain representations.

Impact: Achieve a new benchmark (goal: >93% accuracy) on skin lesion classification.
