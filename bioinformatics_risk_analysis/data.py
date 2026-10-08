import numpy as np


rng = np.random.default_rng(42)

n_genes = 100
n_healthy = 50
n_disease = 50


healthy = rng.normal(
    loc=5.0,
    scale=1.0,
    size=(n_genes, n_healthy)
)

disease = rng.normal(
    loc=5.0,
    scale=1.0,
    size=(n_genes, n_disease)
)

# Calculate 10 genes with higher expression of disease

disease[:10]+=2.0

expression = np.concatenate(
    [healthy, disease],
    axis=1
)

# print(expression[:15])
# print(disease[:15])

labels = np.array(
    [0] * n_healthy + [1] * n_disease
)


print("Expression shape:", expression.shape)
print("Labels shape:", labels.shape)