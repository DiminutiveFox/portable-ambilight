import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from numba import jit
import dxcam

# Assuming dxcam, cam are defined elsewhere
# Assuming OpenCV (cv2) and other necessary libraries are imported

cam = dxcam.create(output_idx=0, output_color="BGR")

# color_palette_256(image_resized)

def create_hsv_histogram(hsv_image):

    # Separate the pixels with V < 64
    low_v_mask = hsv_image[:, :, 2] < 64
    low_v_pixels = hsv_image[low_v_mask]
    high_v_pixels = hsv_image[~low_v_mask]
    
    h_divider = 16
    s_divider = 4
    v_divider = 4

    # Define bin edges for H, S, and V channels for the remaining pixels
    h_bins = np.linspace(0, 179, h_divider+1)  # 16 bins for Hue (0-178)
    s_bins = np.linspace(0, 256, s_divider+1)   # 4 bins for Saturation (0-255)
    v_bins = np.linspace(64, 256, v_divider+1)  # 4 bins for Value (64-255)
    
    # Compute the 3D histogram for pixels with V >= 64
    high_v_hist, _ = np.histogramdd(
        high_v_pixels.reshape(-1, 3),
        bins=(h_bins, s_bins, v_bins)
    )
    
    # Combine the histograms
    hist = np.zeros((h_divider+1, s_divider, v_divider))  # +1 bins to include the separate bin for low V
    hist[0, 0, 0] = low_v_pixels.shape[0]  # All low V pixels go into the first bin
    hist[1:, :, :] = high_v_hist
    
    return hist, (h_bins, s_bins, v_bins), hsv_image.shape[0] * hsv_image.shape[1]

def get_top_n_bins(hist, bin_edges, total_pixels, top_n=5):
    # Flatten the histogram
    hist_flat = hist.flatten()
    
    # Get the indices of the top n bins
    top_indices = np.argsort(hist_flat)[-top_n:][::-1]
    
    # Get the corresponding frequencies
    top_frequencies = hist_flat[top_indices]
    
    # Convert flat indices to multi-dimensional indices
    top_bins = np.unravel_index(top_indices, hist.shape)
    
    # Map bin indices to bin centers
    h_bin_centers = np.concatenate(([0], (bin_edges[0][:-1] + bin_edges[0][1:]) / 2))
    s_bin_centers = (bin_edges[1][:-1] + bin_edges[1][1:]) / 2
    v_bin_centers = np.concatenate(([0], (bin_edges[2][:-1] + bin_edges[2][1:]) / 2))
    
    top_hsv_values = [(h_bin_centers[h], s_bin_centers[s], v_bin_centers[v]) for h, s, v in zip(*top_bins)]
    
    # Calculate the percentage of total pixels
    top_percentages = (top_frequencies / total_pixels) * 100
    
    return top_frequencies, top_hsv_values, top_percentages


def save_bin_centers_to_file(hist, bin_edges, file_path):
    # Map bin indices to bin centers
    h_bin_centers = np.concatenate(([0], (bin_edges[0][:-1] + bin_edges[0][1:]) / 2)).astype(int)
    s_bin_centers = ((bin_edges[1][:-1] + bin_edges[1][1:]) / 2).astype(int)
    v_bin_centers = np.concatenate(([0], (bin_edges[2][:-1] + bin_edges[2][1:]) / 2)).astype(int)

    bin_centers_list = []

    for h in h_bin_centers:
        for s in s_bin_centers:
            for v in v_bin_centers:
                rgb = hsv_to_rgb_file(h, s, v)
                bin_centers_list.append(rgb)
    
    # Save to file
    with open(file_path, 'w') as file:
        for center in bin_centers_list:
            file.write(f"{center[0]},{center[1]},{center[2]}\n")

def hsv_to_rgb(h, s, v):
    # Special case for the bin with V < 64
    if v == 0:
        return np.array([0, 0, 0])
    
    # Convert a single HSV value to RGB
    hsv = np.uint8([[[h, s, v]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
    return rgb / 255.0  # Normalize to [0, 1] for matplotlib


def hsv_to_rgb_file(h, s, v):
    # Convert a single HSV value to RGB
    hsv = np.uint8([[[h, s, v]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
    return rgb


def plot_top_n_bins(top_frequencies, top_hsv_values, top_percentages):
    plt.figure(figsize=(10, 6))
    bar_colors = [hsv_to_rgb(*hsv) for hsv in top_hsv_values]
    bars = plt.bar(range(len(top_frequencies)), top_frequencies, color=bar_colors)
    plt.xlabel('Top Bin Index')
    plt.ylabel('Frequency')
    plt.title('Top 10 HSV Histogram Bins')
    
    # Annotate bars with HSV values and percentages
    for i, (freq, hsv, pct) in enumerate(zip(top_frequencies, top_hsv_values, top_percentages)):
        if hsv == (0, 0, 0):
            label = f"Black\n{pct:.2f}%"
        else:
            label = f'H:{hsv[0]:.1f}\nS:{hsv[1]:.1f}\nV:{hsv[2]:.1f}\n{pct:.2f}%'
        plt.text(i, freq, label, ha='center', va='bottom')
    
    plt.show()

def hsv_to_rgb_dominant(h, s, v):
    # Convert a single HSV value to RGB
    hsv = np.uint8([[[h, s, v]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
    return rgb

def find_dominant_bin(hist, total_pixels, bin_edges):
    # Flatten the histogram
    hist_flat = hist.flatten()
    
    # Get the indices of the bins sorted by frequency (highest to lowest)
    sorted_indices = np.argsort(hist_flat)[::-1]
    
    # Get the positions of the top two bins
    most_frequent_bin_idx = sorted_indices[0]
    second_most_frequent_bin_idx = sorted_indices[1]
    
    # Get the frequency of the most frequent and second most frequent bins
    most_frequent_count = hist_flat[most_frequent_bin_idx]
    second_most_frequent_count = hist_flat[second_most_frequent_bin_idx]
    
    # Get the indices of the bins corresponding to the black pixels (V < 64)
    black_bin_idx = 0  # Black pixels are assigned to the first bin
    
    # Get the frequency of black pixels (V < 64)
    black_count = hist_flat[black_bin_idx]
    
    # Calculate the percentage of black pixels in the image
    black_percentage = (black_count / total_pixels) * 100
    
    # Map bin indices to bin centers
    h_bin_centers = np.concatenate(([0], (bin_edges[0][:-1] + bin_edges[0][1:]) / 2)).astype(int)
    s_bin_centers = ((bin_edges[1][:-1] + bin_edges[1][1:]) / 2).astype(int)
    v_bin_centers = np.concatenate(([0], (bin_edges[2][:-1] + bin_edges[2][1:]) / 2)).astype(int)
    
    # Get the HSV values of the most frequent and second most frequent bins
    def get_hsv_from_idx(bin_idx):
        h, s, v = np.unravel_index(bin_idx, hist.shape)
        return h_bin_centers[h], s_bin_centers[s], v_bin_centers[v]
    
    # Get the RGB value of the most frequent bin
    most_frequent_hsv = get_hsv_from_idx(most_frequent_bin_idx)
    most_frequent_rgb = hsv_to_rgb_dominant(*most_frequent_hsv)
    
    # Get the RGB value of the second most frequent bin
    second_most_frequent_hsv = get_hsv_from_idx(second_most_frequent_bin_idx)
    second_most_frequent_rgb = hsv_to_rgb_dominant(*second_most_frequent_hsv)
    
    # Check the conditions
    if most_frequent_count == black_count and black_percentage < 95:
        # If the most frequent is black and covers less than 95%, return second most frequent bin
        return second_most_frequent_bin_idx, second_most_frequent_rgb
    else:
        # Otherwise, return the position of the black pixel bin
        return black_bin_idx, hsv_to_rgb_dominant(*get_hsv_from_idx(black_bin_idx))




# Example usage
image_path = 'test_pictures/test3.png'
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2HSV)
image_resized = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
s_time = time.time()
hist, bin_edges, total_pixels = create_hsv_histogram(image)
top_frequencies, top_hsv_values, top_percentages = get_top_n_bins(hist, bin_edges, total_pixels, top_n=5)

# plot_top_n_bins(top_frequencies, top_hsv_values, top_percentages)

# Example usage
# file_path = 'bin_centers.txt'
# save_bin_centers_to_file(hist, bin_edges, file_path)
dominant_bin_position, dominant_bin_rgb = find_dominant_bin(hist, total_pixels, bin_edges)
print(time.time()- s_time)
print("The position of the dominant bin is:", dominant_bin_position)
print("The RGB value of the dominant bin is:", dominant_bin_rgb)
