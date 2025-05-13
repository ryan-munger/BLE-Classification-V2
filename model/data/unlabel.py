import pandas as pd
import os

def remove_label_and_copy(input_csv_filepath, output_csv_filepath):
    """
    Removes the 'Label' column from a CSV file and saves the modified data
    to a new CSV file.

    Args:
        input_csv_filepath (str): The full path to the original CSV file.
        output_csv_filepath (str): The full path where the new CSV file
                                     without the 'Label' column will be saved.
    """
    try:
        # Read the input CSV file into a pandas DataFrame
        df = pd.read_csv(input_csv_filepath, low_memory=False)

        # Check if the 'Label' column exists
        if 'Label' in df.columns:
            # Remove the 'Label' column
            df = df.drop(columns=['Label'])

            # Save the modified DataFrame to the new CSV file
            df.to_csv(output_csv_filepath, index=False)
            print(f"Successfully removed the 'Label' column and saved to: {output_csv_filepath}")
        else:
            print(f"The 'Label' column was not found in: {input_csv_filepath}")
            print(f"The original file was not modified, and no new file was created.")

    except FileNotFoundError:
        print(f"Error: Input file not found at: {input_csv_filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file = input("Enter the full path to the input CSV file: ")
    output_file = input("Enter the full path for the output CSV file (without 'Label'): ")
    remove_label_and_copy(input_file, output_file)