import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns

# Load the .mat file
file_path = 'D:\pr1\K.mat'  # Update path if needed
data = scipy.io.loadmat(file_path)

# Access the modes
modes = data['modes']
num_modes = len(modes[0])

print("Loaded modes:")
print(f"Total modes: {num_modes}")

# Function to compute the adjacency matrix
def compute_adjacency_matrix(signal, window_size=50, threshold=0.3):
    num_windows = len(signal) - window_size + 1
    adjacency_matrix = np.zeros((num_windows, num_windows))

    for i in range(num_windows):
        for j in range(num_windows):
            if i != j:
                similarity = np.corrcoef(signal[i:i+window_size], signal[j:j+window_size])[0, 1]
                adjacency_matrix[i, j] = similarity

    # Apply a threshold to convert similarity to adjacency (binary matrix)
    adjacency_matrix[adjacency_matrix < threshold] = 0
    adjacency_matrix[adjacency_matrix >= threshold] = 1

    return adjacency_matrix

# Output folder path
output_folder = 'D:\pr1\MODES'
os.makedirs(output_folder, exist_ok=True)

# Loop through each mode, compute the adjacency matrix for concatenated signals
for mode_index in range(num_modes):
    mode = modes[0, mode_index]  # Access the mode matrix
    print(f"Processing mode {mode_index+1}")

    # Check dimensions
    if mode.ndim == 2 and mode.shape[1] == 600:
        # Concatenate all rows to form a single long signal
        concatenated_signal = mode.flatten()  # Concatenate rows
        print(f"Mode {mode_index+1}: concatenated signal length = {len(concatenated_signal)}")
    else:
        print(f"Skipping mode {mode_index+1} due to incorrect dimensions.")
        continue

    # Compute adjacency matrix for the concatenated signal
    adjacency_matrix = compute_adjacency_matrix(concatenated_signal)
    print(f"Adjacency matrix for mode {mode_index+1} computed.")

    # Save the adjacency matrix
    output_file = os.path.join(output_folder, f'adjacency_matrix_{mode_index+1}.npy')
    np.save(output_file, adjacency_matrix)
    print(f"Saved adjacency matrix to {output_file}")

    # Plot and save the adjacency matrix heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(adjacency_matrix, cmap='viridis', cbar=True, square=True, annot=False)
    plt.title(f'Adjacency Matrix for Mode {mode_index+1}')
    plt.xlabel('Window Index')
    plt.ylabel('Window Index')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'adjacency_matrix_{mode_index+1}.png'))
    plt.close()
    print(f"Saved heatmap to {os.path.join(output_folder, f'adjacency_matrix_{mode_index+1}.png')}")