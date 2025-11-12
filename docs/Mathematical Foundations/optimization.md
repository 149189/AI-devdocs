# Optimization

Optimization finds model parameters that minimize or maximize a given objective.

## Types
- **Convex**: One global minimum (e.g., linear regression).
- **Non-convex**: Many local minima (e.g., deep networks).

## Gradient Descent
Iteratively update weights using gradients:
$$
w_{t+1} = w_t - \\eta \\nabla L(w_t)
$$
where \\( \\eta \\) is the learning rate.

## ML Applications
- Training neural networks.
- Hyperparameter tuning.
