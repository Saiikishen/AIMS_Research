
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


data = [10.46, 9.74, 9.79, 10.53, 10.6, 9.05, 8.97, 10.19, 9.33, 9.93, 11.76, 9.62]
#excluded subj:2,6,11,12,13,


if not data or not all(isinstance(x, (int, float)) for x in data):
    raise ValueError("Data must be a non-empty list of numbers.")


mean = np.mean(data)
std_dev = np.std(data)


x_values = np.linspace(min(data) - 3*std_dev, max(data) + 3*std_dev, 1000)

pdf_values = norm.pdf(x_values, mean, std_dev)


plt.hist(data, bins=10, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Data Histogram')

plt.plot(x_values, pdf_values, 'r', linewidth=2, label=f'Normal Dist.\nμ={mean:.2f}, σ={std_dev:.2f}')


plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Normal Distribution Fit')
plt.legend()
plt.grid(True)

plt.show()


