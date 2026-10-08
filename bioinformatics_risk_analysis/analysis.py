import numpy as np
from data import expression, labels

healthy = expression[:, labels == 0]
disease = expression[:, labels == 1]

# Mean expression per gene

# print(healthy)

healthy_mean = np.mean(healthy, axis=1)
disease_mean = np.mean(disease, axis=1)

# difference in mean expression
difference = disease_mean - healthy_mean

# Identify genes with high expression in disease
mask = difference > 1.0

gene_indices = np.where(mask)[0]

print("Genes with higher expression:")
print(gene_indices)

print("Expression differences:")
print(difference[mask])