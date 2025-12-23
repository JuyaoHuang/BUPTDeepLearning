import kagglehub

# Download latest version
path = kagglehub.dataset_download("vermaavi/food11")

print("Path to dataset files:", path)