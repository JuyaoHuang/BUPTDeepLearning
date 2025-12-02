import kagglehub

# Download latest version
path = kagglehub.dataset_download("jeffaudi/coco-2014-dataset-for-yolov3")

print("Path to dataset files:", path)