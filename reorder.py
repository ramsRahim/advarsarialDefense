from collections import OrderedDict

def reorder_classnames(unordered_filepath, reference_filepath, output_filepath):
    """Reorders classnames in the unordered file based on the classnames in the reference file."""

    # Read the reference file and build an ordered mapping
    reference_classnames = []
    with open(reference_filepath, "r") as f:
        classname = ""
        for line in f: 
            parts = line.split("\t")
            _, classname = parts[0],  " ".join(parts[1:])  # Extract the index and classname
            classname = classname.strip() # Remove leading/trailing whitespace and commas  
            reference_classnames.append(classname)
   # Read the unordered file to get folder names
    unordered_entries = []
    with open(unordered_filepath, "r") as unordered_file:
        for line in unordered_file:
            parts = line.strip().split(" ")
            folder = parts[0]
            unordered_entries.append(folder)  # Only folder names are needed
    print(unordered_entries)       
    print(f"Loaded {len(unordered_entries)} folders from unordered file.")

    # Ensure the number of classnames matches the number of folders
    if len(reference_classnames) != len(unordered_entries):
        raise ValueError("Mismatch: The number of correct classnames does not match the number of folders.")

    # Write the reordered classnames to the output file
    with open(output_filepath, "w") as output_file:
        for folder, classname in zip(unordered_entries, reference_classnames):
            output_file.write(f"{folder} {classname}\n")

    print(f"Reordered classnames written to {output_filepath}.")

# Example usage
unordered_file = "dataset/imagenet-adversarial/classnames.txt"
reference_file = "dataset/ImageNetV2/classnames.txt"
output_file = "dataset/imagenet-adversarial/correct_classnames.txt"
reorder_classnames(unordered_file, reference_file, output_file)
