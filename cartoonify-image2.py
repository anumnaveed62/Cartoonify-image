
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read and display the image
def read_file(filename):
    img = cv2.imread(filename)
    if img is None:
        raise FileNotFoundError(f"Image not found: {filename}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

# Step 2: Create edge mask
def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, line_size, blur_value
    )
    return edges

# Step 3: Reduce color palette using K-Means
def color_quantization(img, k):
    data = np.float32(img).reshape((-1, 3))
    
    # Define criteria and apply k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, center = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    
    # Convert back to uint8 and reconstruct image
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

# Step 4: Apply bilateral filter for smooth coloring (cartoon look)
def apply_bilateral_filter(img, d, sigma_color, sigma_space):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

# Step 5: Combine quantized color + edges to create cartoon
def cartoonify(img, k=8, line_size=7, blur_value=7, bilateral_d=9, sigma_color=75, sigma_space=75):
    # Reduce colors
    quantized = color_quantization(img, k)
    
    # Apply bilateral filter to smooth colors while keeping edges
    blurred = apply_bilateral_filter(quantized, bilateral_d, sigma_color, sigma_space)
    
    # Create edge mask
    edges = edge_mask(img, line_size, blur_value)
    
    # Invert edges to make them black
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Combine: use blurred color where no edge, black where edge
    cartoon = np.where(edges == 0, blurred, 0)  # Black lines on colored background
    
    return cartoon

# ============================
# Main Execution
# ============================

# Update this path to your image
filename = r"C:\Users\Anam\Downloads\Screenshot 2025-10-21 183553.jpg"  # <-- CHANGE THIS TO YOUR IMAGE PATH

# Read image
img = read_file(filename)

# Display original
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

# Generate cartoon
cartoon_img = cartoonify(
    img,
    k=8,              # Number of colors (lower = more cartoonish)
    line_size=7,      # Edge thickness
    blur_value=7,     # Blur for edge detection
    bilateral_d=9,    # Diameter of pixel neighborhood
    sigma_color=75,   # Color sigma
    sigma_space=75    # Space sigma
)

# Display cartoon
plt.subplot(2, 2, 2)
plt.title("Cartoonified Image")
plt.imshow(cartoon_img)
plt.axis("off")

# Optional: Show intermediate steps
quantized = color_quantization(img, k=8)
blurred = apply_bilateral_filter(quantized, 9, 75, 75)
edges = edge_mask(img, 7, 7)
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

plt.subplot(2, 2, 3)
plt.title("Color Quantized")
plt.imshow(quantized)
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Edges")
plt.imshow(edges, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

# Optional: Save the cartoon image
cv2.imwrite("cartoon_output.jpg", cv2.cvtColor(cartoon_img, cv2.COLOR_RGB2BGR))
print("Cartoon image saved as 'cartoon_output.jpg'")











