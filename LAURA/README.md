<div align="center">

# Transformer-based long-term predictor of<br> subthalamic beta activity in Parkinson’s disease

<p align="center">
<img src="LAURAframework.png" width="95%">
</p>


[📇 Preprint](https://cebra.ai/docs/demos.html) |
[📚 Documentation]() |
[⌨️ Code]() 

[Salvatore Falciglia](https://scholar.google.com/citations?user=E-nObHcAAAAJ&hl=it&oi=ao)<sup>1,4#</sup>, 
[Laura Caffi](https://scholar.google.com/citations?user=xoOsKu8AAAAJ&hl=it&oi=ao)<sup>1,2,3,4#</sup>,
[Claudio Baiata]()<sup>2</sup>, 
[Chiara Palmisano](https://scholar.google.com/citations?user=XxgMD7gAAAAJ&hl=it&oi=ao)<sup>3</sup>,
[Ioannis Ugo Isaias](https://scholar.google.com/citations?user=c_2TmpUAAAAJ&hl=it&oi=ao)<sup>2,3\*</sup>
&
[Alberto Mazzoni](https://scholar.google.com/citations?user=b4tE6ScAAAAJ&hl=it&oi=ao)<sup>1,4\*</sup>


<sup>1</sup>The BioRobotics Institute, Sant’Anna School of Advanced Studies, 56025 Pisa, Italy,   
<sup>2</sup>Parkinson Institute of Milan, ASST G.Pini-CTO, 20126 Milano, Italy, 
<sup>3</sup>University Hospital of Würzburg and Julius Maximilian University of Würzburg, 97080 Würzburg, Germany, 
<sup>4</sup>Department of Excellence in Robotics and AI, Sant’Anna School of Advanced Studies, 56127 Pisa, Italy

<sup>#</sup> Corresponding authors, <sup>\*</sup> These authors contributed equally to this work

</div>

# Welcome! 👋

**CEBRA** is a library for estimating **C**onsistent **E**m**B**eddings of high-dimensional **R**ecordings using **A**uxiliary variables. It contains self-supervised learning algorithms implemented in PyTorch, and has support for a variety of different datasets common in biology and neuroscience.

To receive updates on code releases, please 👀 watch or ⭐️ star this repository!

``cebra`` is a self-supervised method for non-linear clustering that allows for label-informed time series analysis.
It can jointly use behavioral and neural data in a hypothesis- or discovery-driven manner to produce consistent, high-performance latent spaces. While it is not specific to neural and behavioral data, this is the first domain we used the tool in. This application case is to obtain a consistent representation of latent variables driving activity and behavior, improving decoding accuracy of behavioral variables over standard supervised learning, and obtaining embeddings which are robust to domain shifts.


# Reference

- 📄 **Publication May 2023**:
  [Learnable latent embeddings for joint behavioural and neural analysis.](https://doi.org/10.1038/s41586-023-06031-6)
  Steffen Schneider*, Jin Hwa Lee* and Mackenzie Weygandt Mathis. Nature 2023.

- 📄 **Preprint April 2022**:
  [Learnable latent embeddings for joint behavioral and neural analysis.](https://arxiv.org/abs/2204.00673)
  Steffen Schneider*, Jin Hwa Lee* and Mackenzie Weygandt Mathis

# License

- Since version 0.4.0, CEBRA is open source software under an Apache 2.0 license.
- Prior versions 0.1.0 to 0.3.1 were released for academic use only (please read the license file).
