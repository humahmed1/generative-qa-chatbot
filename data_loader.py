import pandas as pd
from langchain.document_loaders.csv_loader import CSVLoader
import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = (
    r"/usr/local/Cellar/tesseract/5.3.2_1/bin/tesseract"
)


# load csv table for tqa
def load_csv_data(file_path):
    loader = CSVLoader(file_path=file_path)
    return loader.load()


def draw_column_separator(image_path):
    """
    Draw a line to separate two columns of text in an image.
    The line starts before the word "Description" and goes from the top to the bottom of the page.

    Parameters:
        image_path (str): Path to the image.

    Returns:
        None
    """
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Use pytesseract to do OCR on the grayscale image
    ocr_text = pytesseract.image_to_string(image)

    # Find the position of the word "Description" in the OCR output
    description_index = ocr_text.find("Description") - 130

    # Get the height and width of the image
    height, width, _ = image.shape

    # Draw a line starting before the word "Description" going from top to bottom
    line_start_point = (description_index, 0)
    line_end_point = (description_index, height)
    color = (0, 0, 255)  # Green color in BGR
    thickness = 2
    image_with_line = cv2.line(
        image, line_start_point, line_end_point, color, thickness
    )

    # Save the image with the line
    output_path = "output_image_with_line.png"
    cv2.imwrite(output_path, image_with_line)

    return output_path


def extract_table_data(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to extract the red line (assuming red line has a specific color range)
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([100, 100, 255])
    mask = cv2.inRange(image, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and get the largest one (assuming the red line is the largest contour)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    # Extract coordinates of the red line
    red_line_x, _, red_line_w, _ = cv2.boundingRect(contours[0])

    # Crop the image to get left and right columns
    left_column = gray[:, :red_line_x]
    right_column = gray[:, red_line_x + red_line_w :]

    # Perform OCR on the left and right columns and remove empty strings
    left_text = [
        line.strip()
        for line in pytesseract.image_to_string(left_column).split("\n")
        if line.strip()
    ]
    right_text = [
        line.strip()
        for line in pytesseract.image_to_string(right_column).split("\n")
        if line.strip()
    ]

    # Extract the headings and data separately
    left_heading = left_text[0]
    right_heading = right_text[0]
    data = list(zip(left_text[1:], right_text[1:]))

    # Create a DataFrame using the key-value pairs and specify column names
    df = pd.DataFrame(data, columns=[left_heading, right_heading])

    # Create a third column with the context for the model
    df["Concatenated"] = df["Feature"] + " means " + df["Description."]

    return df
