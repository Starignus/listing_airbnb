# Code structure

* **EDA_cleaning_models**: Contains the python notebooks and html version for:
  
    * **01_EDA_cleaning_processing_new_metric.ipynb**: Data cleaning, processing, EDA, and computation of new metric 
      for model 2.
    * **02_Model_1_with_GA_output.ipynb**: Model 1 first listing price.
    * **03_Model_2_with_GA_output.ipynb**: Model 2 Overall Review Value
      

* **GA_runs**: Feature selection notebooks run in Google Colab environment.
    * **01_Genetic_algorithm_Airbnb_Model_1.ipynb**: [Link to Colab notebook](https://colab.research.google.com/drive/1TczLRnKJtOsEZWAMM-Dx4lflkqyfn6pV?usp=sharing)
    * **02_Genetic_algorithm_Airbnb_Model_2_not_cut.ipynb**: [Link to Colab notebook](https://colab.research.google.com/drive/1Cxw_NtZ33Z8YQg-qu6VXy6-Wrt-qORxk?usp=sharing)

* **Text_processing**: Notebook to process features with text.
    * **01_Texthero_Text_processing.ipynb**: [Link to Colab notebook](https://colab.research.google.com/drive/15gaKR8-b7NN4emIJ3dIYsB1COR6BD-Ek?usp=sharing)
    
* **SHAP**:SHAP Values(SHapley Additive exPlanations) break down a prediction of 1st model to 
  show the impact of each feature in a particular prediction.
  * The notebooks are in this code but they can be found as well in [Colab notebook](https://colab.research.google.com/drive/1_zEL9cPk9Y5zLMGWa5NBYwIeqVFuYxFD?usp=sharing)

* **tools**: This is a module that contains code for data exploration, GA feature selection, and model selection.

* **python_env.yml**: Conda environment requirements.