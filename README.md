# Interpretable-ML-shear-squat-wall

Interpretability of machine learning (ML) methods is one of the largest criticisms of applications of ML methods in structural engineering is that they are generally black-box models that provide no insights unlike the mechanics-based or even empirical regression-based models. Understanding the input-output relationships/sensitivities helps to have informed decision strategies and proper deployment of the model, and, perhaps more importantly, to identify situations where the ML model is not reliable. To this end, this work adopts a unified approach dubbed as SHapley Additive exPlanations (SHAP) to explain why the ML model makes the prediction for shear strength of RC squat walls. It will not only gives the feature importance in an average sense, but also quantitatively explains how each feature affects the prediction of the model for the whole data set, and how each feature affects the prediction for a given individual sample. That is to say, SHAP unifies global interpretation (for the whole set) and local interpretation (for an individual sample) for an ML model.

Reference:

De-Cheng Feng, Wen-Jie Wang, Sujith Mangalathu, and Ertugrul Taciroglu. "Implementing ensemble learning methods to predict the shear strength of RC deep beams with/without web reinforcements". Journal of Structural Engineering, Volume 147, Issue 11.
(https://doi.org/10.1061/(ASCE)ST.1943-541X.0003115)
