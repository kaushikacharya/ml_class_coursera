# CS229 Lecture Notes

## Chapter 1: Linear Regression

- Magnitude of the update is proportional to the error term.
- Closed form:
  - $\theta = (X^TX)^{-1}X^Ty$
  - Assumption: $X^TX$ is an invertible matrix.
  - Non-invertible matrix cases:
    - Number of linearly independent examples < Number of features
    - Features are not linearly independent.
  - $y^{(i)} = \theta^Tx^{(i)} + \epsilon^{(i)}$
    - $\epsilon^{(i)}$ captures
      - Unmodeled effects:
        - Some features very pertinent to prediction left out of the regression.
      - Random noise
- Probabilistic interpretation:
  - Maximizing log likelihood:
    - Instead of maximizing $L(\theta)$, we can also maximize any strictly increasing function of $L(\theta)$.
    - $\theta$ that maximizes $l(\theta)$ minimizes least-squares cost function.
- Geometric interpretation:
  - Source: Section 7.3.2 of Kevin Murphy's Machine Learning - A Probabilistic Perspective
  - Derivation:
    - To minimize the norm of the residual, we want the residual vector to be orthogonal to every column of $X$.

## Chapter 2: Classification and Logistic Regression

- Derivation of cost function $J(\theta)$
  - Coursera slide skips the steps of the derivation.
  - Whereas Page 22 of CS229 notes show the steps.

## Chapter 8: Generalization

- Domain Shift:
  - In classical statistical learning settings, the training examples are also drawn from the same distribution as the test distribution $D$.
  - But nowadays, researchers are increasingly more interested in the setting where training and test distributions are different.
- This chapter studies how the test error is influenced by the learning procedure, especially the choice of model parameterizations.
- Test error is decomposed into:
  - bias
  - variance
- Bias-variance tradeoff
  - Bias explanation:
    - Underfitting example:
      - Ground truth data distribution: Non-linear say quadratic.
      - Model fit: Linear
      - High bias:
        - Large training error.
        - Even with infinite training data, the best fitted linear model is still inaccurate and fails to capture the structure of the data.
          - This point is highlighted by showing error of fitting linear model on a noiseless dataset.
    - Overfitting example:
      - Ground truth data distribution: Quadratic
      - Model fit: 5-th degree polynomial
      - Low bias:
        - Low training error but high test error.
        - Model learnt from the training set **does not generalize** well to test set.
        - Reasoning for being termed low bias:
          - Fitting a 5-th degree polynomial to an extremely large dataset, the model would be close to a quadratic function and be accurate.
            - Setting $\theta_5=0$, $\theta_4=0$, $\theta_3=0$ would result in quadratic function.

    - Variance explanation:
      - Spurious patterns in the training set
        - Observation noise
        - Data that happens to be present in our small, finite training set.
          - But that do not reflect the wider pattern of the relationship between x and y.
        - Example (Fig 8.7) shows dataset generated from the same distribution, leads to highly varying 5-th degree model.

    - Tradeoff:
      - Simple model with very few parameters:
        - May have large bias (but small variance).
        - Typically may suffer from underfitting.
      - Complex model with very many parameters:
        - May suffer from large variance (but have smaller bias) and thus overfitting.
