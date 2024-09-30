import torch
import os
import glob
import subprocess
import flirpy
import flirpy.io.seq
from flirpy.io.fff import Fff
import numpy as np
import imageio.v3 as iio
import cv2
import time
import csv
from datetime import datetime
from tqdm import tqdm

# Define the paths
script_path = r"C:\Users\jland\anaconda3\Scripts\split_seqs.py"
exiftool_path = r"G:\seq_manage"
seq_input_folder = r"G:\seq_manage\Process_seqs"
output_folder = r"F:\Seq_results"


# Functions
def extract_timestamps(exiftool_path, tempfolder):
    exiftool_cmd = os.path.join(exiftool_path, 'exiftool.exe')
    output_file = os.path.join(tempfolder, 'datetime.txt')
    input_pattern = os.path.join(tempfolder, '*.fff')

    command = [
        exiftool_cmd,
        '-DateTimeOriginal',
        '-s',
        '-r',
        '-T',
        '-f',
        '-fast',
        input_pattern
    ]

    with open(output_file, 'w') as outfile:
        subprocess.run(command, stdout=outfile)


def convert_timestamps_to_csv(video_id, timestamps_file, output_folder):
    output_csv = os.path.join(output_folder, "Timestamp_lut.csv")

    with open(timestamps_file, 'r') as infile:
        timestamps = infile.readlines()

    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['video_ID', 'frame_number', 'timestamp'])

        for i, timestamp in enumerate(tqdm(timestamps, desc="Processing timestamps", unit="timestamp")):
            timestamp = timestamp.strip()

            # Remove the timezone marker
            if '-' in timestamp:
                timestamp = timestamp.split('-')[0]
            elif '+' in timestamp:
                timestamp = timestamp.split('+')[0]

            frame_number = f"frame{str(i).zfill(6)}"
            csvwriter.writerow([video_id, frame_number, timestamp])


def calculate_fps(timestamps_file):
    with open(timestamps_file, 'r') as infile:
        timestamps = infile.readlines()

    # Convert timestamps to datetime objects
    datetime_format = "%Y:%m:%d %H:%M:%S.%f"
    datetimes = [
        datetime.strptime(timestamp.strip().split('+')[0].split('-')[0], datetime_format)
        for timestamp in timestamps
    ]

    # Calculate time differences between consecutive frames
    time_deltas = [
        (datetimes[i + 1] - datetimes[i]).total_seconds()
        for i in range(len(datetimes) - 1)
    ]

    # Calculate average time difference
    avg_time_delta = sum(time_deltas) / len(time_deltas)

    # Calculate frames per second (fps)
    fps = 1 / avg_time_delta
    return fps


def image_process(array):
    min_val = np.min(array)
    max_val = np.max(array)
    scaled_array = (array - min_val) / (max_val - min_val)
    # Convert to from 0-255
    scaled_array *= 255
    scaled_array = scaled_array.astype(int)
    return scaled_array


def save_as_rainbow(gray_path, rainbow_path):
    # Convert temp values to scaled values from 0-1
    gray_image = image_process(iio.imread(gray_path))

    # Ensure image is 8-bit and single channel
    if len(gray_image.shape) == 2:  # Already single channel
        gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif len(gray_image.shape) == 3 and gray_image.shape[2] == 1:  # Single channel with an extra dimension
        gray_image = cv2.normalize(gray_image[:, :, 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        print(f"Unexpected image format: {gray_path}")

    # Apply rainbow color to image
    rainbow_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)

    # IF necessary:
    # Reverse color order to match thermal image (red = hot, blue = cold)
    # rainbow_export = cv2.cvtColor(rainbow_image, cv2.COLOR_BGR2RGB)
    # plt.imshow(rainbow_image)

    # Save the image into output folder
    cv2.imwrite(rainbow_path, rainbow_image)


def rainbow_to_avi(rainbow_path, avi_output_path, fps):
    image_files = [f for f in os.listdir(rainbow_path) if f.endswith('.png')]
    image_files.sort()
    frame = cv2.imread(os.path.join(rainbow_path, image_files[0]))  # Read first image to get dimensions
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
    video_writer = cv2.VideoWriter(avi_output_path, fourcc, fps, (width, height))
    # Write each image as a frame into the video file
    for image_file in image_files:
        image_path = os.path.join(rainbow_path, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the VideoWriter
    video_writer.release()


'''
# Wait to convert to temperature until after segment is extracted

def tiff2temp(array_path, temp_output_path):
    array = iio.imread(array_path)
    array = array.astype(np.float64)

    # Convert from raw to kelvin
    array *= .04
    # Convert from kelvin to Celsius
    array += -273.15

    # Round to three digits
    array = np.round(array, decimals=3)

    # Save array as .txt file
    np.savetxt(temp_output_path, array, fmt='%.3f', delimiter='\t')
'''

# Output folder for everything
os.makedirs(output_folder, exist_ok=True)

# Split .seq file into png file and txt file containing temps
for filename in os.listdir(seq_input_folder):
    if filename.endswith(".seq"):
        start_time = time.time()
        input_file = os.path.join(seq_input_folder, filename)
        command = [
            "python", script_path,
            "-i", input_file,
            "-o", output_folder,
            "--no_export_preview"
            #  "--width", width,
            #  "--height", height,
            #  "--preview_format", "jpg",
            #  "--export_tiff",
            #  "--merge_folders",
            #  "--export_raw"
        ]

        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        end_time = time.time()
        print(f"File {filename} split, and took {end_time - start_time:.2f} seconds.")
    else:
        print("No .seq files found.")

##### Extract timestamps

if __name__ == "__main__":
    fps_list = []
    for filename in os.listdir(seq_input_folder):
        if filename.endswith(".seq"):
            start_time = time.time()
            video_ID = os.path.splitext(filename)[0]
            fff_folder = os.path.join(output_folder, video_ID, "raw")
            csv_output_folder = os.path.join(output_folder, video_ID)

            extract_timestamps(exiftool_path, fff_folder)
            print(f"Timestamps have been extracted for {video_ID}.")

            timestamps_file = os.path.join(fff_folder, "datetime.txt")
            convert_timestamps_to_csv(video_ID, timestamps_file, csv_output_folder)
            print(f"Timestamps have been written to csv for {video_ID}.")

            fps = calculate_fps(timestamps_file)
            fps_list.append((video_ID, fps))
            print(f"The framerate of video {video_ID} is {fps}.")
            end_time = time.time()
            print(f"Timestamps & FPS for video {video_ID} are extracted, and took {end_time - start_time:.2f} seconds.")

#### Convert gray jpg images to rainbow for easier analysis
####  AND simultaneously save temp data arrays

# Find the new folder where tiff files are being saved
for filename in os.listdir(seq_input_folder):
    if filename.endswith(".seq"):
        start_time = time.time()
        # Remove ".seq"
        video_ID = os.path.splitext(filename)[0]

        # ID folder with grayscale
        gray_image_folder = os.path.join(output_folder, video_ID, "radiometric")

        # Make new folder to save rainbow
        rainbow_image_folder = os.path.join(output_folder, video_ID, "rainbow")
        os.makedirs(rainbow_image_folder, exist_ok=True)

        # Make new folder to save temps
        #        temp_output_folder = os.path.join(output_folder, video_ID, "temp")
        #        os.makedirs(temp_output_folder, exist_ok=True)

        ###  Convert .tiff files to rainbow AND save temps
        image_files = [f for f in os.listdir(gray_image_folder) if f.endswith(".tiff")]

        # Progress bar for processing images
        for image_filename in tqdm(image_files, desc=f"Video {video_ID}:", unit="image"):
            # Split .tiff from image ID
            imageID = os.path.splitext(image_filename)[0]

            # Create paths for input & output of images and temp
            gray_image_path = os.path.join(gray_image_folder, image_filename)
            rainbow_output_path = os.path.join(rainbow_image_folder, f"{imageID}.png")
            #            temp_output_path = os.path.join(temp_output_folder, f"{imageID}.txt")

            # Save rainbow images & temp files
            save_as_rainbow(gray_image_path, rainbow_output_path)
        #            tiff2temp(gray_image_path, temp_output_path)
        end_time = time.time()
        print(f"Rainbow images for {video_ID} are extracted, and took {end_time - start_time:.2f} seconds.")

        # Specify output folder
        avi_output_path = os.path.join(output_folder, video_ID, f"{video_ID}.avi")
        # Retrieve fps for this video:
        fps = next((fps for vid, fps in fps_list if vid == video_ID), None)
        # Convert rainbow to avi
        start_time = time.time()
        rainbow_to_avi(rainbow_image_folder, avi_output_path, fps)
        end_time = time.time()
        print(f"{video_ID}.avi created, and took {end_time - start_time:.2f} seconds.")
