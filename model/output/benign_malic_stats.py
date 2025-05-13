def calculate_malicious_percentage(filepath):
    """
    Calculates the percentage of benign (0) and malicious (1) entries in a file.

    Args:
        filepath (str): The path to the file containing 0 or 1 on each line.

    Returns:
        tuple: A tuple containing the percentage of benign and malicious entries,
               or None if the file is empty or not found.
    """
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return None

    if not lines:
        print("Warning: The file is empty.")
        return (0.0, 0.0)

    total_count = len(lines)
    benign_count = lines.count('0')
    malicious_count = lines.count('1')

    benign_percentage = (benign_count / total_count) * 100
    malicious_percentage = (malicious_count / total_count) * 100

    return (benign_percentage, malicious_percentage)

if __name__ == "__main__":
    file_path = input("Enter the path to the file: ")
    percentages = calculate_malicious_percentage(file_path)

    if percentages:
        benign_percent, malicious_percent = percentages
        print(f"\nPercentage of benign entries (0): {benign_percent:.2f}%")
        print(f"Percentage of malicious entries (1): {malicious_percent:.2f}%")