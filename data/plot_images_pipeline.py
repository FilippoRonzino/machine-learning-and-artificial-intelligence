import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import os

def draw_line(data, size: int = 80) -> Image:
    '''
    Draws a line graph from the data and normalizes each column to sum to 255.

    :param data: The data to be plotted. It should be a 1D array or list.
    :param size: The side size of the image to be created. Default is 80.
    :return: A PIL Image object representing the line graph.
    '''

    if not hasattr(data, '__iter__'):
        raise ValueError("Input data must be an iterable (e.g., list, numpy array).")
    if len(data) == 0:
        raise ValueError("Input data cannot be empty.")
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer.")
    
    data = np.array(data)

    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")

    x_vals = np.linspace(0, size - 1, len(data))
    y_min, y_max = np.min(data), np.max(data)
    y_vals = (1 - (data - y_min) / (y_max - y_min)) * (size - 1)
    img = Image.new("L", (size, size), "black")
    draw = ImageDraw.Draw(img)

    for i in range(len(x_vals) - 1):
        x1, y1 = x_vals[i], y_vals[i]
        x2, y2 = x_vals[i + 1], y_vals[i + 1]
        draw.line((x1, y1, x2, y2), fill="white")
    
    arr = np.array(img)
    normalized_arr = np.uint8((arr / arr.sum(axis=0))*254) + 1

    normalized_img = Image.fromarray(normalized_arr, 'L')
    return normalized_img

def draw_line_row(row, size: int = 80) -> Image:
    '''
    Draws a line graph from the row data and normalizes each column to sum to 255.

    :param row: The row data to be plotted. It should be a 1D array or list.
    :param size: The side size of the image to be created. Default is 80.
    :return: A PIL Image object representing the line graph.
    '''
    
    if not hasattr(row, '__iter__'):
        raise ValueError("Input row must be an iterable (e.g., list, numpy array).")
    if len(row) == 0:
        raise ValueError("Input row cannot be empty.")
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer.")
    
    return draw_line(row, size)

def apply_draw_line_to_dataframe(df, size: int = 80) -> pd.DataFrame:
    '''
    Applies the draw_line function to each row of the DataFrame.

    :param df: The DataFrame containing the data to be plotted.
    :param size: The side size of the image to be created. Default is 80.
    :return: A DataFrame with a new column 'image' containing the drawn images.
    '''
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")
    
    df['image'] = df.apply(lambda row: draw_line_row(row, size), axis=1)
    return df

def save_images_from_dataframe(df, output_dir: str):
    '''
    Saves the images from the DataFrame to the specified directory.

    :param df: The DataFrame containing the images to be saved.
    :param output_dir: The directory where the images will be saved.
    '''
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")
    if not isinstance(output_dir, str):
        raise ValueError("Output directory must be a string.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, row in df.iterrows():
        image_path = os.path.join(output_dir, f'{idx}.png')
        row['image'].save(image_path)

def save_images_from_df_file(df_file: str, output_dir: str, size: int = 80):
    '''
    Loads a DataFrame from a file and saves the images to the specified directory.

    :param df_file: The file path of the DataFrame.
    :param output_dir: The directory where the images will be saved.
    :param size: The side size of the image to be created. Default is 80.
    '''
    
    if not isinstance(df_file, str):
        raise ValueError("Input df_file must be a string.")
    if not isinstance(output_dir, str):
        raise ValueError("Output directory must be a string.")
    
    df = pd.read_parquet(df_file)
    df = apply_draw_line_to_dataframe(df, size=80)
    save_images_from_dataframe(df, output_dir)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    ecg_train_path = os.path.join(current_dir, 'data_storage', 'ecg_parquets', 'train_ecg.parquet')
    ecg_test_path = os.path.join(current_dir, 'data_storage', 'ecg_parquets', 'test_ecg.parquet')

    harmonic_train_path = os.path.join(current_dir, 'data_storage', 'harmonic_ou_parquets', 'train_harmonic.parquet')
    harmonic_test_path = os.path.join(current_dir, 'data_storage', 'harmonic_ou_parquets', 'test_harmonic.parquet')

    ou_train_path = os.path.join(current_dir, 'data_storage', 'harmonic_ou_parquets', 'train_ou.parquet')
    ou_test_path = os.path.join(current_dir, 'data_storage', 'harmonic_ou_parquets', 'test_ou.parquet')

    sp500_train_path = os.path.join(current_dir, 'data_storage', 'sp500_parquets', 'train_sp500.parquet')
    sp500_test_path = os.path.join(current_dir, 'data_storage', 'sp500_parquets', 'test_sp500.parquet')

    
    save_images_from_df_file(ecg_train_path, os.path.join(current_dir, 'images', 'ecg', 'train'), size=80)
    save_images_from_df_file(ecg_test_path, os.path.join(current_dir, 'images', 'ecg', 'test'), size=80)
    print("ECG images saved successfully.")
    save_images_from_df_file(harmonic_train_path, os.path.join(current_dir, 'images', 'harmonic', 'train'), size=80)
    save_images_from_df_file(harmonic_test_path, os.path.join(current_dir, 'images', 'harmonic', 'test'), size=80)
    print("Harmonic images saved successfully.")
    save_images_from_df_file(ou_train_path, os.path.join(current_dir, 'images', 'ou', 'train'), size=80)
    save_images_from_df_file(ou_test_path, os.path.join(current_dir, 'images', 'ou', 'test'), size=80)
    print("OU images saved successfully.")
    save_images_from_df_file(sp500_train_path, os.path.join(current_dir, 'images', 'sp500', 'train'), size=80)
    save_images_from_df_file(sp500_test_path, os.path.join(current_dir, 'images', 'sp500', 'test'), size=80)
    print("SP500 images saved successfully.")
    print("Done.")
