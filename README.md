# US Arrests Clustering Analysis

This repository contains the code and analysis for clustering US states based on crime rates using the `USArrests` dataset. The project explores the application of hierarchical clustering and K-means clustering to identify patterns and group states with similar crime characteristics.

## Project Overview

This project aims to apply clustering techniques to understand the relationships between different types of crimes (Murder, Assault, Rape) and urban population percentages in US states. We perform exploratory data analysis, feature scaling, dimensionality reduction using PCA, and implement both hierarchical and K-means clustering. The project evaluates the performance of these models and provides insights into crime patterns across states.

## Dataset

The dataset used is `usArrests.csv`, which contains the following features:

-   `City`: Name of the US state.
-   `Murder`: Murder rate per 100,000 population.
-   `Assault`: Assault rate per 100,000 population.
-   `UrbanPop`: Percentage of urban population.
-   `Rape`: Rape rate per 100,000 population.

## Libraries Used

-   `numpy`
-   `pandas`
-   `seaborn`
-   `matplotlib`
-   `scikit-learn` (`sklearn`)
-   `scipy`

## Analysis and Model

1.  **Exploratory Data Analysis (EDA):**
    -   Loaded the dataset and examined its structure.
    -   Checked for missing values and data types.
    -   Calculated and visualized the correlation matrix using a heatmap.
    -   Observed high correlations between Murder, Assault, and Rape.

2.  **Data Preprocessing:**
    -   Excluded the `City` column for numerical analysis.
    -   Scaled the numerical features using `StandardScaler`.
    -   Performed dimensionality reduction using Principal Component Analysis (PCA).

3.  **Clustering:**
    -   **Hierarchical Clustering:**
        -   Used `AgglomerativeClustering` to perform hierarchical clustering.
        -   Visualized the dendrogram to determine the optimal number of clusters.
        -   Evaluated the clustering performance using the silhouette score.
    -   **K-means Clustering:**
        -   Used the elbow method to determine the optimal number of clusters (K=4).
        -   Encoded the `City` column into numerical values for K-means.
        -   Trained and evaluated the K-means model.
        -   Visualized the clusters using scatter plots and calculated the silhouette score.
        -   Generated a confusion matrix and classification report.

4.  **Model Evaluation:**
    -   Evaluated both clustering methods using the silhouette score, confusion matrix, and classification report.
    -   Observed low silhouette scores for both hierarchical (0.33) and K-means (0.34) clustering.
    -   Observed poor performance in the K-means classification report, indicating that clustering is not the ideal model for this dataset.

## Key Findings

-   The correlation heatmap revealed high correlations between murder, assault, and rape rates.
-   The elbow method and dendrogram suggested four clusters as optimal.
-   Both hierarchical and K-means clustering yielded low silhouette scores, indicating poor clustering performance.
-   The K-means classification report showed low precision, recall, and f1-scores, suggesting that these clustering methods are not well-suited for this dataset.
-   Given the continuous nature of the data, a multiple linear regression model might be more appropriate.

## How to Run

1.  Clone the repository: `git clone [repository URL]`
2.  Install the required libraries: `pip install numpy pandas seaborn matplotlib scikit-learn scipy`
3.  Run the Jupyter notebook `US_Arrests_Clustering.ipynb` to reproduce the analysis and model training.

## Conclusion

This project demonstrates the application of hierarchical and K-means clustering to the US Arrests dataset. The evaluation results indicate that clustering methods may not be the most effective approach for this dataset. Future work could explore regression models or alternative clustering techniques.

## References

-   [Seaborn Correlation Heatmap Tutorial](https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e)
-   [Kaggle USArrests Starter Notebook](https://www.kaggle.com/code/kerneler/starter-usarrets-73d905cc-b)

## Contact

For any questions or feedback, please feel free to contact me.
