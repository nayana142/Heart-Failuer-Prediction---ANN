# Heart Failure Prediction using ANN with SMOTE and Early Stopping

## **Project Overview**
This project focuses on predicting heart failure using a deep learning model (Artificial Neural Network - ANN). The dataset consists of various patient health parameters, and the model is trained to classify whether a patient is at risk of heart failure (DEATH_EVENT). To handle class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** is used. Additionally, **Early Stopping** is implemented to prevent overfitting and optimize model performance.

## **Dataset Description**
The dataset consists of the following features:

| Feature                  | Description |
|--------------------------|-------------|
| **age**                 | Age of the patient |
| **anaemia**             | Whether the patient had haemoglobin below the normal range (0 = No, 1 = Yes) |
| **creatinine_phosphokinase** | Level of creatine phosphokinase in the blood (mcg/L) |
| **diabetes**            | Whether the patient has diabetes (0 = No, 1 = Yes) |
| **ejection_fraction**   | Percentage of blood pumped by the left ventricle per heartbeat |
| **high_blood_pressure** | Whether the patient has hypertension (0 = No, 1 = Yes) |
| **platelets**           | Platelet count in the blood (kiloplatelets/mL) |
| **serum_creatinine**    | Level of serum creatinine in the blood (mg/dL) |
| **serum_sodium**        | Level of serum sodium in the blood (mEq/L) |
| **sex**                 | Gender of the patient (0 = Female, 1 = Male) |
| **smoking**             | Whether the patient smokes or has ever smoked (0 = No, 1 = Yes) |
| **time**                | Follow-up period (months) |
| **DEATH_EVENT**         | Target variable (0 = No death, 1 = Death) |

## **Project Workflow**
1. **Data Preprocessing**  
   - Handling missing values (if any)  
   - Feature scaling using MinMaxScaler  
   - Splitting the dataset into training and testing sets  
   
2. **Handling Class Imbalance**  
   - Using **SMOTE** to balance the dataset by generating synthetic samples for the minority class
   
3. **Building the ANN Model**  
   - Input Layer: Number of neurons equal to the number of features
   - Hidden Layers: Fully connected layers with **ReLU activation**
   - Output Layer: Single neuron with **Sigmoid activation**
   - Optimizer: Adam
   - Loss Function: Binary Cross-Entropy
   
4. **Implementing Early Stopping**  
   - Monitoring validation loss
   - Stopping training if validation loss does not improve for a certain number of epochs
   
5. **Model Evaluation**  
   - Performance metrics: Accuracy, Precision, Recall, F1-score
 
   
## **Installation & Dependencies**
To run this project, install the required libraries:

```bash
pip install numpy pandas scikit-learn tensorflow imbalanced-learn matplotlib seaborn
```

## **Results**
- The model achieves **high accuracy** after applying SMOTE and Early Stopping.



