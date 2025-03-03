import numpy as np
import matplotlib.pyplot as plt

class MultipleLinearRegression:
    def __init__(self):
        self.coefficients = None
        self.r_squared = None
    
    def fit(self, X, y):
        """
        Fit the multiple linear regression model.
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        self : returns an instance of self
        """
        # Add intercept term (column of ones)
        X_with_intercept = np.column_stack((np.ones(len(X)), X))
        
        # Reshape y as needed
        y_reshaped = y.reshape(-1, 1) if len(y.shape) == 1 else y
        
        # Calculate coefficients using least squares
        self.coefficients = np.linalg.lstsq(X_with_intercept, y_reshaped, rcond=None)[0]
        
        # Calculate R-squared
        y_pred = self.predict(X)
        y_pred_flat = y_pred.flatten()
        y_flat = y.flatten()
        rss = np.sum((y_flat - y_pred_flat)**2)
        tss = np.sum((y_flat - np.mean(y_flat))**2)
        self.r_squared = 1 - (rss / tss)
        
        return self
    
    def predict(self, X):
        """
        Predict using the linear model.
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            Samples
        
        Returns:
        y_pred : array of shape (n_samples,)
            Predicted values
        """
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        
        # Add intercept term
        X_with_intercept = np.column_stack((np.ones(len(X)), X))
        
        # Make predictions
        return X_with_intercept @ self.coefficients
    
    def get_coefficients(self):
        """Return the model coefficients"""
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        return self.coefficients
    
    def get_r_squared(self):
        """Return the R-squared value"""
        if self.r_squared is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        return self.r_squared
    
    def get_equation_string(self):
        """Return a string representation of the regression equation"""
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        
        eq = f"y = {self.coefficients[0][0]:.4f}"
        for i in range(1, len(self.coefficients)):
            eq += f" + {self.coefficients[i][0]:.4f} * X{i}"
        
        return eq
    
    def plot_3d(self, X, y):
        """
        Plot the regression plane and data points in 3D (only for 2 features)
        
        Parameters:
        X : array-like of shape (n_samples, 2)
            The features (must be exactly 2 for 3D plotting)
        y : array-like of shape (n_samples,)
            The target values
        """
        if X.shape[1] != 2:
            raise ValueError("3D plotting requires exactly 2 feature dimensions")
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter plot of the data points
        ax.scatter(X[:, 0], X[:, 1], y, color='red', label='Data points')
        
        # Create the mesh grid for the regression plane
        x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
        y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
        X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
        
        # Predict Z values for the mesh grid
        XY = np.column_stack((X_mesh.flatten(), Y_mesh.flatten()))
        Z_mesh = self.predict(XY).reshape(X_mesh.shape)
        
        # Plot the regression plane
        surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh, color='blue', alpha=0.3, label='Regression plane')
        
        # Set labels
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')
        ax.set_title(f'Multiple Linear Regression (R² = {self.r_squared:.3f})')
        
        return fig, ax


# Example usage:
if __name__ == "__main__":
    # Sample data from the original file
    Points = [(3.5, 2.9, 3.1), (1.3, 2.1, 2.8),
              (5.9, 4.1, 6.1), (4.8, 3.2, 3.8)]
    
    # Extract X and y
    X = np.array([[p[0], p[1]] for p in Points])
    y = np.array([p[2] for p in Points])
    
    # Create and fit the model
    model = MultipleLinearRegression()
    model.fit(X, y)
    
    # Print coefficients
    coeffs = model.get_coefficients()
    print(f"Intercept (β0): {coeffs[0][0]:.4f}")
    print(f"Coefficient X1 (β1): {coeffs[1][0]:.4f}")
    print(f"Coefficient X2 (β2): {coeffs[2][0]:.4f}")
    print(f"Equation: {model.get_equation_string()}")
    print(f"R-squared: {model.get_r_squared():.4f}")
    
    # Make predictions for the original data points
    predictions = model.predict(X)
    print("Predictions:", predictions.flatten())
    
    # Plot the model
    fig, ax = model.plot_3d(X, y)
    plt.show() 