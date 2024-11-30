<div align="center">

# **Transformer-based long-term predictor of<br> subthalamic beta activity in Parkinson’s disease**

<p align="center">
<img src="LAURAframework.png" width="95%">
</p>


[📇 Preprint](https://doi.org/10.1101/2024.11.25.24317759) |
[📉 Linear predictions](linear_prediction/linear_visualization/readme.md) |
[📈 LAURA predictions](src/EDA/patient_visualization/readme.md) 

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

# Welcome👋 to the LAURA beta-telling🔮 programme!🧠

**LAURA** (**L**earning bet**A**-power distrib**U**tions through **R**ecurrent **A**nalysis) is a personalized Transformer-based framework for forecasting subthalamic beta power distribution. Our algorithm analyses home recordings with 1-min resolution in patients who have undergone DBS surgery and are experiencing chronic stimulation. LAURA proved efficacy in both the one-day-ahead and multi-day-ahead prediction tasks, with forecasts extending up to six days. The performance vastly outperformed linear algorithms. The approach was validated in four parkinsonian patients exhibiting heterogeneous subthalamic beta power dynamics and independently of stimulation parameters.

LAURA resulted from the translation of recent achievements in natural language processing (NLP) and time-series forecasting into the domain of brain disorders through the use of deep learning (DL) techniques. Crucially, the need of DL architectures for large sets of diverse data for training poses a challenge in the clinical setting. Indeed, our results rely on the availability of heterogeneous recordings collected with a resolution of minutes over one year. For each patient, several consecutive days over months of recordings within different sets of DBS parameters have been analysed.

LAURA framework represents a significant advancement within the aDBS workflow, potentially supporting the **widespread adoption of the aDBS therapy as a long-term treatment strategy** for Parkinson's Disease (PD). 

Our study paves the way for remote monitoring strategies and the implementation of **new algorithm for personalized auto-tuning aDBS devices**. (Stay tuned!👀)




# Reference

- 📄 **Preprint December 2024**:
  [Transformer-based long-term predictor of subthalamic beta activity in Parkinson’s disease.](https://doi.org/10.1101/2024.11.25.24317759)
  S.Falciglia, L.Caffi, C.Baiata, C.Palmisano, I.U.Isaias* and A.Mazzoni*
