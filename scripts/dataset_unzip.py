import os
import tarfile
import shutil

source_directory = r"C:\Users\RPitchuka4915\Downloads\OAIData\image03\00m\0.C.2"
destination_directory = r"C:\Users\RPitchuka4915\Downloads\OAIData\extracted\00m\0.C.2"

for dir_path, sub_dirs, files in os.walk(source_directory):
    for filename in files:
        if filename.lower().endswith(".tar.gz"):
            tar_file_path = os.path.join(dir_path, filename)
            print(f"Extracting {tar_file_path}")

            try:
                with tarfile.open(tar_file_path, 'r') as tar:
                    for member in tar.getmembers():
                        if member.isfile():
                            extracted_file = tar.extractfile(member)
                            if extracted_file:
                                # Get last two subfolders
                                parts = os.path.normpath(dir_path).split(os.sep)
                                if len(parts) >= 2:
                                    last = parts[-1]
                                    second_last = parts[-2]
                                    new_filename = f"{last}-{second_last}.dcm"
                                else:
                                    new_filename = "unknown.dcm"

                                dest_path = os.path.join(destination_directory, new_filename)
                                os.makedirs(destination_directory, exist_ok=True)

                                with open(dest_path, "wb") as out_f:
                                    shutil.copyfileobj(extracted_file, out_f)

            except Exception as e:
                print(f"Failed to extract {tar_file_path}: {e}")
