import kagglehub

# Download latest version
path = kagglehub.dataset_download("steubk/wikiart")

print("Path to dataset files:", path)