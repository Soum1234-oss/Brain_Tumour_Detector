import cv2
import numpy as np
import pandas as pd
import os
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error

# ----------------------------------
# Step 1: Functions
# ----------------------------------

def add_noise(image, prob=0.05):
    noisy = image.copy()
    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            r = np.random.random()

            if r < prob/2:
                noisy[i,j] = 0
            elif r < prob:
                noisy[i,j] = 255

    return noisy


def adaptive_median_filter(image, max_window=7):

    padded = np.pad(image, max_window//2, mode='edge')
    filtered = image.copy()

    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):

            window_size = 3

            while window_size <= max_window:

                half = window_size//2
                window = padded[i:i+window_size, j:j+window_size]

                zmin = np.min(window)
                zmax = np.max(window)
                zmed = np.median(window)
                zxy = padded[i+half, j+half]

                if zmin < zmed < zmax:

                    if zmin < zxy < zmax:
                        filtered[i,j] = zxy
                    else:
                        filtered[i,j] = zmed
                    break

                else:
                    window_size += 2

                    if window_size > max_window:
                        filtered[i,j] = zmed
                        break

    return filtered


def weighted_mean_filter(image):

    kernel = np.array([
        [1,2,1],
        [2,4,2],
        [1,2,1]
    ])

    kernel = kernel / kernel.sum()

    filtered = cv2.filter2D(image,-1,kernel)

    return filtered


# ----------------------------------
# Step 2: Dataset Path
# ----------------------------------

dataset_path = r"C:\Users\souma\Downloads\archive (2)\brain_tumor_dataset"

results = []

# ----------------------------------
# Step 3: Process Entire Dataset
# ----------------------------------

for label in ["yes","no"]:

    folder = os.path.join(dataset_path, label)

    if not os.path.exists(folder):
        print("Folder not found:", folder)
        continue

    for file in os.listdir(folder):

        img_path = os.path.join(folder,file)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img,(224,224))

        # Add noise
        noisy_img = add_noise(img)

        # Adaptive median filter
        amf_img = adaptive_median_filter(noisy_img)

        # Weighted mean filter
        wmf_img = weighted_mean_filter(amf_img)

        # Evaluation
        mse = mean_squared_error(img, wmf_img)
        psnr = peak_signal_noise_ratio(img, wmf_img)

        results.append([file,label,mse,psnr])

        # Show sample images
        cv2.imshow("Original",img)
        cv2.imshow("Noisy",noisy_img)
        cv2.imshow("Filtered",wmf_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break


cv2.destroyAllWindows()

# ----------------------------------
# Step 4: Save Results
# ----------------------------------

df = pd.DataFrame(results,columns=["Image","Tumor","MSE","PSNR"])

print(df.head())

print("\nAverage MSE:",df["MSE"].mean())
print("Average PSNR:",df["PSNR"].mean())

df.to_csv("brain_tumor_results.csv",index=False)

print("\nResults saved to brain_tumor_results.csv")