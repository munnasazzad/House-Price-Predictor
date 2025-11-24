
This is only made for  Vityarthi college project under the assistance of Dr Santosh Kumar Sahoo, VIT BHOPAL, 

🏠 README: Simple House Price Predictor (Python)

This project uses a basic machine learning model to estimate house prices in Indian Rupees (₹) based on a small sample dataset. The model uses core Python functions, making it a great, self-contained example of how machine learning algorithms work without relying on external libraries like Scikit-learn or Pandas.

-----

🧠 Model Overview: Linear Regression

The core of this script is **Linear Regression**, an algorithm used to predict a continuous value (the price) based on input features (size, location, and age).

| Feature | Type | How it's used |
| :--- | :--- | :--- |
| **Size** (sqft) | Numerical | Larger size generally means a higher price. |
| **Location** | Categorical | Encoded as numbers (Rural: 0, Suburban: 1, Urban: 2) to reflect its impact on price. |
| **Age** (years) | Numerical | Older age generally means a lower price. |
| **Target** | **Price** (₹) | The value the model learns to predict. |

-----

📉 Training Method: Gradient Descent

The model is trained using **Gradient Descent**. This is an optimization process where the model starts with random guesses for the relationship parameters (**weights** and **bias**).

1.  **Calculate Error:** It makes a prediction and calculates the difference (error) between the prediction and the actual price.
2.  **Adjust Parameters:** It then calculates the **gradient** (the slope of the error function) and slightly adjusts the weights and bias in the direction that reduces this error.

[Image of gradient descent visualization]

3.  **Iterate:** This process is repeated thousands of times (`iterations`) until the model's parameters settle at values that give the best possible price predictions for the given data.

🧹 Data Handling

To help the training process run smoothly:

  * **Feature Normalization:** All numerical features (Size, Age, and the encoded Location) are **standardized**. This involves adjusting their scale so they all contribute equally to the gradient calculation, leading to faster and more stable training.
  * **Location Encoding:** Categorical data like 'urban,' 'suburban,' and 'rural' is converted into numerical codes (0, 1, 2) because machine learning models only understand numbers.

-----

How to Run

1.  Save the code as a Python file (e.g., `predictor.py`).
2.  Run the script from your terminal:
    ```bash
    python predictor.py
    ```
3.  The script will first display the training results and the final weights.
4.  It will then start an **interactive loop** where you can enter the size, location, and age of a house to get an immediate price prediction.

submitted by:-
Munna Babu Ansari
25BEY10007
