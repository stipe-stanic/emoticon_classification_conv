import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from plotting import plot_single_image


# manual segmentation for now - testing
def segment_face(image_path):
    img = Image.open(image_path)

    # 'L' -> Luminance, creates greyscale image with all pixels set to zero
    # img.size = [width, height]
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, img.size[0], img.size[1]), fill=255)

    masked_img = Image.new('RGBA', img.size, 0)
    masked_img.paste(img, (0, 0), mask=mask)
    plot_single_image(masked_img, "Masked image")

    labels = label_pixels_masked_img(masked_img)
    labeled_image = apply_labels(masked_img, labels)
    plot_single_image(labeled_image, "Labeled image")

    return labeled_image


# labels the image based on rgba values of original image
def label_pixels_masked_img(masked_image):
    labels = []
    pixels = masked_image.load()

    for i in range(masked_image.size[0]):
        for j in range(masked_image.size[1]):
            pixel = pixels[i, j]
            if pixel[3] == 0:  # pixel = [RGBA -> red, green, blue, alpha]
                labels.append('background')
            elif pixel[0] > 248 and pixel[1] > 190 and pixel[2] > 30:
                labels.append("facial_features_space")
            elif pixel[0] <= 220 and pixel[1] <= 120 and pixel[2] >= 0:
                labels.append("facial_features")
            else:
                labels.append("smiling_face_background")

    return labels


# applies colors to the new image based on label values
def apply_labels(masked_image, labels):
    label_img = Image.new('RGB', masked_image.size, (255, 255, 255))
    draw = ImageDraw.Draw(label_img)

    for i in range(label_img.size[0]):
        for j in range(label_img.size[1]):
            label = labels[i * label_img.size[1] + j]
            if label == 'background':
                color = (32, 143, 140)
            elif label == 'facial_features_space':
                color = (68, 1, 84)
            elif label == 'facial_features':
                color = (53, 25, 14)
            else:
                color = (253, 231, 36)

            draw.point((i, j), fill=color)

    return label_img
