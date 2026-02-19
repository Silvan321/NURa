import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread(r"C:\Users\toetseb\Documents\NURa\tutorial_2\dog_image.jpg")
grey_img = np.dot(img[..., :3], [0.640, 0.595, 0.155])  # Rec601 hack to turn greyscale
# Do NOT fall into the rabbit hole of that is the
# luminous efficiency function for human eyes.
# In case you need to see the image
# plt.imshow(grey_img, cmap="gray")

# assignment 1a
# Use full_matrices = False to make sure the number of columns of U matches the number of rows of VT, so we can do matrix multiplication: U x W x VT
# The @ operator can do matrix multiplication!
U, W, VT = np.linalg.svd(grey_img, full_matrices=False)
print(f"{grey_img.shape=}")
print(f"{U.shape=}")
print(f"{W.shape=}")
print(f"{VT.shape=}")
# reconstruction = (U * W[..., None, :]) @ VT
# print(f"{reconstruction}")
reconstruction = U @ np.diag(W) @ VT
print(f"{reconstruction.shape=}")

truncation_size = [5, 10, 50]
for size in truncation_size:
    truncated_W = np.zeros_like(W)  # Make sure truncated_W still has a shape of 534 x 800, but all the elements after the truncation size are 0.
    truncated_W[:size] = W[:size]
    truncated_reconstruction = U @ np.diag(truncated_W) @ VT
    plt.figure()
    plt.imshow(truncated_reconstruction, cmap="gray")
    plt.title(f"Truncated SVD with {size} singular values")
    plt.show()
