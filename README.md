# Atlantis: Enabling Underwater Depth Estimation with Stable Diffusion 

> [**Atlantis: Enabling Underwater Depth Estimation with Stable Diffusion**]()  
> Fan Zhang, Shaodi You, Yu Li, Ying Fu  
> CVPR 2024

![Teaser](assets/teaser.jpg)

This repository contains the official implementation and dataset of the CVPR2024 paper "Learning Rain Location Prior for Nighttime Deraining", by Fan Zhang, Shaodi You, Yu Li, Ying Fu.

[Paper](https://arxiv.org/pdf/2312.12471.pdf) | [Supp](https://drive.google.com/file/d/1mNkiC5XyrcdPeLUt32-E94EYnH4IUrrW/view?usp=sharing) | [Data]()

> Monocular depth estimation has experienced significant progress on terrestrial images in recent years, largely due to deep learning advancements. However, it remains inadequate for underwater scenes, primarily because of data scarcity. Given the inherent challenges of light attenuation and backscattering in water, acquiring clear underwater images or precise depth information is notably difficult and costly. Consequently, learning-based approaches often rely on synthetic data or turn to unsupervised or self-supervised methods to mitigate this lack of data. Nonetheless, the performance of these methods is often constrained by the domain gap and looser constraints. In this paper, we propose a novel pipeline for generating photorealistic underwater images using accurate terrestrial depth data. This approach facilitates the training of supervised models for underwater depth estimation, effectively reducing the performance disparity between terrestrial and underwater environments. Contrary to prior synthetic datasets that merely apply style transfer to terrestrial images without altering the scene content, our approach uniquely creates vibrant, non-existent underwater scenes by leveraging terrestrial depth data through the innovative Stable Diffusion model. Specifically, we introduce a unique Depth2Underwater ControlNet, trained on specially prepared \{Underwater, Depth, Text\} data triplets, for this generation task. Our newly developed dataset enables terrestrial depth estimation models to achieve considerable improvements, both quantitatively and qualitatively, on unseen underwater images, surpassing their terrestrial pre-trained counterparts. Moreover, the enhanced depth accuracy for underwater scenes also aids underwater image restoration techniques that rely on depth maps, further demonstrating our dataset's utility.

## Method
![framework](assets/pipeline.png)

## Update
- [ ] Data Release.
- [ ] Code Release.
- **2024.02.27:** Accepted by CVPR 2024!
- **2023.12.18:** Repo created.

## Experimental Results
#### Results on UIEB
![depth](assets/depth.png)

#### Results on Sea-Thru
![real](assets/real.jpg)

#### Results on Underwater Image Enhancement
![downstream](assets/downstream.jpg)