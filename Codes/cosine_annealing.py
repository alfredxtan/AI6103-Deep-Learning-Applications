import numpy as np
import matplotlib.pyplot as plt

def cosine_annealing(t):
    return 0.025 * (1 + np.cos(np.pi * t / 300))

if __name__ == "__main__":
# Generate t values
    t = np.linspace(0, 300, 1000)

# Generate y values
    y = cosine_annealing(t)

# Create the plot
    plt.figure(figsize=(10,6))
    plt.plot(t, y)
    plt.title('Learning Rate Following cosine Annealing')
    plt.xlabel('Epoch, t')
    plt.ylabel('Learning Rate')
    #plt.grid(True)
    plt.show()


