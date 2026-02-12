# Assignment-2-ML
Machine Learning Classification Project with Streamlit Deployment

a.  Problem Statement - This project aims to classify breast cancer tumors as malignant or benign using multiple machine learning classification models and deploy them using Streamlit.

b.  Dataset Description - 

    b.1 Dataset: Breast Cancer Wisconsin (Diagnostic)
    
    b.2 Source: UCI Machine Learning Repository
    
    b.3 Number of Instances: 569
    
    b.4 Number of Features: 30
    
    b.5 Target Variable: Diagnosis (0 = Benign, 1 = Malignant)
    
c.  Models Used

    c.1 Logistic Regression
    
    c.2 Decision Tree
    
    c.3 K-Nearest Neighbors
    
    c.4 Naive Bayes (Gaussian)
    
    c.5 Random Forest
    
    c.6 XGBoost

    Comparison Table

    | =================================================================================================== |
    | Model                     |  Accuracy  |    AUC    | Precision |  Recall   |  F1 Score |    MCC     |
    | =================================================================================================== |
    | Logistic Regression       | 0.9825     | 0.9812    | 0.9861    | 0.9861    | 0.9861    | 0.9623     |
    | Decision Tree             | 0.9123     | 0.9157    | 0.9559    | 0.9028    | 0.9286    | 0.8174     |
    | KNN                       | 0.9561     | 0.9788    | 0.9589    | 0.9722    | 0.9655    | 0.9054     |
    | Naive Bayes               | 0.9386     | 0.9878    | 0.9452    | 0.9583    | 0.9517    | 0.8676     |
    | Random Forest             | 0.9561     | 0.9931    | 0.9589    | 0.9722    | 0.9655    | 0.9054     |
    | XGBoost                   | 0.9474     | 0.9931    | 0.9459    | 0.9722    | 0.9589    | 0.8864     |
    | =================================================================================================== |

    Observation

    | =================================================================================================== |     
    | **ML Model Name**            | **Observation about model performance**                              |
    | =================================================================================================== |     
    |                                                                                                     |     
    | **Logistic Regression**      | Achieved the highest accuracy (98.25%) and MCC (0.9623), indicating  |
    |                              | excellent overall classification performance. The high precision and |
    |                              | recall (0.9861) suggest balanced detection of both malignant and     |
    |                              | benign cases. The dataset appears highly linearly separable, which   |
    |                              | explains its superior performance.                                   |
    | **Decision Tree**            | Recorded the lowest accuracy (91.23%) and MCC (0.8174) among all     |
    |                              | models. Although precision was high, recall was comparatively lower, |
    |                              | indicating some misclassification of malignant cases. This may be    |
    |                              | due to overfitting and limited generalization ability.               |
    | **kNN**                      | Achieved strong performance with 95.61% accuracy and 0.9054 MCC.     |
    |                              | High recall (0.9722) shows effective identification of malignant     |
    |                              | cases. Performance benefited from feature scaling since KNN is       |
    |                              | distance-based.                                                      |
    | **Naive Bayes**              | Obtained 93.86% accuracy with a strong AUC (0.9878). Despite the     |
    |                              | independence assumption between features, the model performed well.  |
    |                              | Slightly lower accuracy suggests minor impact of correlated features.|
    | **Random Forest (Ensemble)** | Achieved 95.61% accuracy and the highest AUC (0.9931). The ensemble  |
    |                              | approach improved generalization and reduced overfitting. Balanced   |
    |                              | precision and recall indicate stable performance.                    |
    | **XGBoost (Ensemble)**       | Achieved 94.74% accuracy and the highest AUC (0.9931), demonstrating |
    |                              | excellent class separation. Although slightly lower in accuracy      |
    |                              | compared to Logistic Regression, it provided robust                  |
    |                              | probability-based predictions.                                       |
    | =================================================================================================== |     
