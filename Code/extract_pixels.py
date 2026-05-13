import os
from PIL import Image
import pandas as pd
import numpy as np

def extract_pixels(image_path, output_path):
    print(f"Loading image from {image_path}...")
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
        return

    # Convert to RGB
    img = img.convert('RGB')
    width, height = img.size
    print(f"Image size: {width}x{height} pixels")

    # Get pixel data as numpy array for speed
    print("Extracting pixel data...")
    pixels = np.array(img)
    
    # Reshape to list of pixels
    # pixels is (height, width, 3)
    # We want a DataFrame with x, y, r, g, b
    
    # Create coordinate grids
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    # Flatten arrays
    r = pixels[:,:,0].flatten()
    g = pixels[:,:,1].flatten()
    b = pixels[:,:,2].flatten()
    x = x_coords.flatten()
    y = y_coords.flatten()
    
    print("Creating DataFrame...")
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'r': r,
        'g': g,
        'b': b
    })
    
    print(f"Saving data to {output_path}...")
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    # Save sample to Excel
    excel_path = output_path.replace('.csv', '_sample.xlsx')
    print(f"Saving sample (first 10,000 rows) to {excel_path}...")
    # Excel has a row limit of 1,048,576. The full dataset is ~100M rows.
    # We save a small sample for the user to view the format.
    df.head(10000).to_excel(excel_path, index=False)
    
    print("Done!")

if __name__ == "__main__":
    # Paths relative to this script
    # Script is in Code/, Image is in parent directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    
    image_file = os.path.join(parent_dir, "Sample.png")
    output_file = os.path.join(parent_dir, "pixel_data.csv")
    
    extract_pixels(image_file, output_file)
