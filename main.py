import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Read the image file
input_image = Image.open('input.png')

# Convert the image to a numpy array
input_data = np.array(input_image)

# Split image into color channels
r = input_data[:, :, 0]
g = input_data[:, :, 1]
b = input_data[:, :, 2]


# Perform histogram equalization on a single channel
def hist_eq(channel):
    # Compute the histogram of the channel
    hist, bins = np.histogram(channel, bins=256, range=(0, 255))

    # Calculate the cumulative sum of the histogram
    cum = hist.cumsum()

    # Normalize the histogram
    norm = 255 * cum / cum[-1]

    # Interpolate the original channel with the cumulative sum of the histogram
    eq = np.interp(channel, bins[:-1], norm)

    # Convert the float numbers into unsigned 8-bit integers
    return eq.astype(np.uint8)


# Apply histogram equalization on each channel
r_eq = hist_eq(r)
g_eq = hist_eq(g)
b_eq = hist_eq(b)

# Combine the color channels into a rgb image
output_data = np.dstack((r_eq, g_eq, b_eq))

# Convert the numpy array into an image
output_image = Image.fromarray(output_data)

# Save the output image file
output_image.save('output.png')

# Create 2 sub-plots
_, (input_plt, output_plt) = plt.subplots(nrows=1, ncols=2, num='Histogram Equalization')

# Show the input image
input_plt.set_title('Input')
input_plt.imshow(input_image)

# Show the output image
output_plt.set_title('Output')
output_plt.imshow(output_image)

plt.show()
