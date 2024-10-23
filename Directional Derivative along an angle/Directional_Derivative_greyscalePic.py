import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_directional_derivative(image_path, theta):
    """
    Computes and displays the directional derivative of an image along a specific angle.

    Parameters:
    - image_path: Path to the input image file.
    - theta: The angle (in degrees) along which to compute the directional derivative.
    """
    # Convert theta from degrees to radians
    theta_rad = np.deg2rad(theta)

    # Compute the direction vector components based on the angle
    v_x = np.cos(theta_rad)
    v_y = np.sin(theta_rad)

    # Load the image
    image = cv2.imread(image_path)

    # Step 1: Convert to grayscale if the image is not already grayscale
    if len(image.shape) == 3:  # Check if the image has color channels
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = image

    # Step 2: Compute the gradients in x and y directions using Sobel filters
    sobel_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
    sobel_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction

    # Step 3: Compute the directional derivative in the given direction
    directional_derivative = sobel_x * v_x + sobel_y * v_y

    # Step 4: Plot the results
    plt.figure(figsize=(10, 8))

    # Original image (if in color)
    if len(image.shape) == 3:
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image (Color)')

    # Grayscale image
    plt.subplot(2, 2, 2)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Grayscale Image')

    # Gradient in x-direction
    plt.subplot(2, 2, 3)
    plt.imshow(sobel_x, cmap='gray')
    plt.title('Gradient in X direction')

    # Gradient in y-direction
    plt.subplot(2, 2, 4)
    plt.imshow(sobel_y, cmap='gray')
    plt.title('Gradient in Y direction')

    # Step 5: Show the directional derivative image
    plt.figure()
    plt.imshow(directional_derivative, cmap='gray')
    plt.title(f'Directional Derivative along {theta} degrees')

    # Display all the plots
    plt.show()

# Example of how to use the function:
compute_directional_derivative('/Users/ae/Documents/Magistrale Ingeneria Informatica/Computer Vision/pics/group-water-buffalo-260nw-82222993.jpg', 45)  # Compute derivative along 45 degrees