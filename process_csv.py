import os
import json
import random
import argparse

def process_files(input_folder, test_split_size, seed):
    # Set random seed for reproducibility
    random.seed(seed)

    # Define output directories
    train_output_dir = f"{input_folder}_train"
    test_output_dir = f"{input_folder}_test"

    # Ensure output directories exist
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Iterate through all JSON files in the folder
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".json"):
            # Read JSON file
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            # Group by label
            label_0 = [item for item in data if item["label"] == 0]
            label_1 = [item for item in data if item["label"] == 1]

            # Shuffle data
            random.shuffle(label_0)
            random.shuffle(label_1)

            # Extract data for the test set
            test_data = label_0[:test_split_size] + label_1[:test_split_size]

            # Remaining data as the training set
            train_data = label_0[test_split_size:] + label_1[test_split_size:]

            # Get file name without extension
            base_name = os.path.splitext(file_name)[0]

            # Write test set file
            test_file_path = os.path.join(test_output_dir, f"{base_name}.json")
            with open(test_file_path, "w", encoding="utf-8") as test_file:
                json.dump(test_data, test_file, ensure_ascii=False, indent=4)

            # Write training set file
            train_file_path = os.path.join(train_output_dir, f"{base_name}.json")
            with open(train_file_path, "w", encoding="utf-8") as train_file:
                json.dump(train_data, train_file, ensure_ascii=False, indent=4)

            print(f"File {file_path} has been processed:")
            print(f"  Test set saved to: {test_file_path}")
            print(f"  Training set saved to: {train_file_path}")

    print(f"All JSON files in folder {input_folder} have been processed!")
    print(f"Training sets saved to: {train_output_dir}")
    print(f"Test sets saved to: {test_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split JSON dataset into training and testing sets.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing JSON files.")
    parser.add_argument("--test_split_size", type=int, default=50, help="Number of samples per label for the test set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    args = parser.parse_args()

    process_files(args.input_folder, args.test_split_size, args.seed)
