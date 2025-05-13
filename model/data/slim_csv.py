#!/usr/bin/env python3
"""
CSV Row Remover - Efficiently removes specified rows from a CSV file.
"""
import os
import sys
import csv
import time

def remove_csv_rows(file_path, start_line, end_line):
    """Remove rows from a CSV file between start_line and end_line (inclusive)."""
    # Create a temporary file path
    temp_file = file_path + ".temp"
    
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as input_file, \
             open(temp_file, 'w', newline='', encoding='utf-8') as output_file:
            
            # Initialize writer with same dialect as reader
            reader = csv.reader(input_file)
            writer = csv.writer(output_file)
            
            # Process the file in chunks
            print(f"Processing file: {file_path}")
            print(f"Removing rows {start_line} through {end_line}...")
            start_time = time.time()
            
            # Process each line
            for i, row in enumerate(reader, 1):  # Start counting from 1 to match line numbers
                if i < start_line or i > end_line:
                    writer.writerow(row)
                
                # Print progress for large files
                if i % 100000 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {i:,} rows... ({elapsed:.2f} seconds)")
    
        # Replace the original file with the modified one
        os.replace(temp_file, file_path)
        
        total_time = time.time() - start_time
        print(f"\nOperation completed successfully in {total_time:.2f} seconds.")
        print(f"Removed {end_line - start_line + 1:,} rows from {file_path}.")
        
    except Exception as e:
        print(f"Error: {e}")
        # Clean up temporary file if it exists
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False
    
    return True

def main():
    """Main function to run the script."""
    print("CSV Row Remover")
    print("-" * 40)
    
    # Get file path from user
    file_path = input("Enter the path to your CSV file: ").strip()
    
    # Check if file exists
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    
    try:
        # Get start and end lines from user
        start_line = int(input("Enter the starting line number to remove: "))
        end_line = int(input("Enter the ending line number to remove: "))
        
        # Validate input
        if start_line < 1:
            print("Error: Starting line must be a positive number.")
            return
        
        if end_line < start_line:
            print("Error: Ending line cannot be less than starting line.")
            return
            
        # Process the file
        remove_csv_rows(file_path, start_line, end_line)
        
    except ValueError:
        print("Error: Line numbers must be integers.")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()