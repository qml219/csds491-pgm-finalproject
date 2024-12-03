# Mixture Discriminant Analysis (MDA) Classifier

This repository implements a Mixture Discriminant Analysis (MDA) classifier in Python, inspired by Trevor Hastie’s *“Discriminant Analysis by Gaussian Mixtures”*. MDA models each cluster as a mixture of Gaussian distributions, effectively extending Quadratic Discriminant Analysis (QDA) with a more nuanced assumption about data distribution.

## Key Features
- Derives probability terms using **Bayes’ Rule** and **Lagrangian Multipliers**.
- Demonstrates that MDA is equivalent to QDA under specific assumptions.
- Highlights overfitting risks of nuanced models through a comprehensive comparison of MDA and QDA.

## Performance Comparison
The model is evaluated on classifying **benign vs. malignant breast cancers**, showcasing:
- MDA's flexibility with complex distributions.
- Overfitting tendencies compared to QDA.

## Dependencies
- Python in Jupyter Notebook
- Scikit-learn
- Numpy

## References

Hastie, T., & Tibshirani, R. (1996). Discriminant Analysis by Gaussian Mixtures. *Journal of the Royal Statistical Society. Series B (Methodological), 58*(1), 155–176. 

Ma, J., Gao, W. The supervised learning Gaussian mixture model. *J. of Comput. Sci. & Technol. 13*, 471–474 (1998). https://doi.org/10.1007/BF02948506

Dr. Michael Lewicki’s lecture slides

John Ramey: A Brief Look at Mixture Discriminant Analysis | R-bloggers 

### This repo includes a written report in markdown and PDF (Results and Evaluation Report), a source code implementation Jupyter Notebook, the presentation slides in PDF (which is a summarizing powerpoint version of the written report), together with the images used in rendering the markdown.
