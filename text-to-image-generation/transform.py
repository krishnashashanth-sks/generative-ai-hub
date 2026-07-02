import torchvision.transforms as transforms
from PIL import Image

# Define image transformations
image_transform = transforms.Compose([
    transforms.Resize((256, 256)), # Resize to 256x256
    transforms.ToTensor(), # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
])

def load_and_transform_image(image_path):
  """
  Loads an image from the given path and applies the defined transformations.
  """
  try:
    image = Image.open(image_path).convert('RGB')
    transformed_image = image_transform(image)
    return transformed_image
  except Exception as e:
    print(f"Error loading or transforming image {image_path}: {e}")
    return None