import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

def draw_beta_pdf(alpha):
    # Define the alpha parameter for the beta distribution
    beta_param = alpha

    # Generate a range of x values
    lam = np.linspace(0, 1, 1000)

    # Generate the corresponding y values from the beta distribution
    y = beta.pdf(lam, alpha, beta_param)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(lam, y)

    # Set the title and labels
    plt.title('Beta Distribution with alpha = {}'.format(alpha))
    plt.xlabel('lambda')
    plt.ylabel('Probability Density')

    # Show the plot
    
    plt.show()
    
    
if __name__ == "__main__":
    draw_beta_pdf(0.2)

