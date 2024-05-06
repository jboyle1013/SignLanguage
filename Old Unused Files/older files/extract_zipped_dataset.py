import subprocess


def extract_multipart_zip(zip_file_path, destination_folder):
    # Ensure 7-Zip is installed and 7z executable is in your PATH
    try:
        subprocess.run(["7z", "x", zip_file_path, "-o" + destination_folder], check=True)
        print(f"Extraction completed successfully to {destination_folder}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    except FileNotFoundError:
        print("7-Zip not found. Ensure it's installed and available in the system PATH.")



if __name__ == "__main__":
    # Path to the first part of the multipart ZIP file (e.g., 'data.zip.001')


    zip_file_path = f"dataset/data.zip.001"

    # Destination folder where the files will be extracted
    destination_folder = f"data"

    extract_multipart_zip(zip_file_path, destination_folder)
