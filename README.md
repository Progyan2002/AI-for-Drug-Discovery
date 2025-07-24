# AI-for-Drug-Discovery
PPARγ QSAR Drug Discovery Model
This Quantitative Structure-Activity Relationship (QSAR) model is designed to predict the potency (pEC50) of small molecules targeting the Peroxisome Proliferator-Activated Receptor gamma (PPARγ), a key therapeutic target for metabolic disorders such as type 2 diabetes. By leveraging machine learning and cheminformatics, the model accelerates the identification of promising drug candidates with high potency and potential therapeutic efficacy.

Key Features 

Data Preprocessing: The model processes bioactivity data from a dataset (PPARg_bioactivity_data_nM.csv) containing 1,247 compounds with canonical SMILES and standard value (IC50) measurements. Invalid SMILES are filtered, duplicates are removed, and IC50 values are converted to pEC50 (-log10[IC50 in M]) for regression analysis.
Feature Generation: Molecular descriptors are generated using Morgan fingerprints (radius=2, 2048 bits) via RDKit, capturing structural and physicochemical properties of compounds to enable robust predictive modeling.
Machine Learning Model: An XGBoost regressor is employed with optimized hyperparameters (n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8) to predict pEC50 values, balancing model complexity and generalization.
Model Performance: Evaluated on a 20% test set, the model achieves a training R² of 0.9029 (MSE: 0.1375) and a test R² of 0.5663 (MSE: 0.5705, MAE: 0.5532), indicating strong predictive capability with moderate generalization to unseen data.
Visualization and Diagnostics: The model includes scatter plots of predicted vs. actual pEC50 values and residual plots to assess prediction accuracy and model fit, saved as pec50_prediction_plot_xgboost.png and residuals_plot_xgboost.png.
Prediction Pipeline: The model supports predictions for new compounds by generating Morgan fingerprints from SMILES strings and outputting pEC50 and corresponding EC50 (nM) values. Example predictions for two compounds yielded EC50 values of 89.8576 nM and 25.2321 nM, demonstrating practical applicability.
Model and Data Storage: The trained model is saved as pparg_drug_discovery_model.pkl, and processed data is stored as processed_pparg_data.csv for reproducibility and future use.

Workflow
The model follows a streamlined pipeline:

Data Loading and Preprocessing: Reads and cleans bioactivity data, ensuring valid SMILES and converting IC50 to pEC50.
Feature Generation: Computes Morgan fingerprints for each compound.
Model Training and Evaluation: Trains the XGBoost model and evaluates performance using MSE, R², and MAE metrics.
Model Persistence: Saves the trained model and processed data for deployment.
Prediction: Generates pEC50 predictions for new compounds, facilitating virtual screening.

Applications
This model is tailored for pharmaceutical researchers and drug discovery teams focusing on PPARγ-targeted therapies. It enables rapid virtual screening of compound libraries, prioritization of high-potency candidates, and reduction of experimental costs in early-stage drug development.
Future Improvements

Enhance generalization by incorporating additional molecular descriptors or larger, more diverse datasets.
Optimize hyperparameters further using grid search or Bayesian optimization.
Integrate ADMET (absorption, distribution, metabolism, excretion, toxicity) predictions to improve candidate selection.

This PPARγ QSAR model offers a robust, data-driven tool to accelerate the discovery of novel therapeutics, with potential applications in metabolic disease research and beyond.
