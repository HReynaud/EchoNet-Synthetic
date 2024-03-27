#!/bin/bash

# Check for the correct number of arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <source_image_folder> <source_pt_folder> <destination_folder>"
    exit 1
fi

# Assign arguments to variables
source_image_folder="$1"
source_pt_folder="$2"
destination_folder="$3"

# Create the destination folder if it doesn't exist
mkdir -p "$destination_folder"

# Enable nullglob to ensure wildcard patterns that match no files expand to nothing
shopt -s nullglob

# Initialize counters
copied_count=0
total_files=$(ls -1q "$source_image_folder" | wc -l)
matched_files=()
pt_files_count=$(ls -1q "$source_pt_folder"/*.pt | wc -l)
processed_files=0

# Loop through .pt files in the source_pt_folder
for pt_file in "$source_pt_folder"/*.pt; do
    # Extract the basename without the extension
    base_name=$(basename "$pt_file" .pt)

    # Construct the wildcard pattern for the source files
    pattern="$source_image_folder"/"$base_name".*

    # Attempt to copy the matching file(s) to the destination folder
    if cp $pattern "$destination_folder" 2>/dev/null; then
        matched_files+=("$pattern")
        ((copied_count++))
    fi

    # Update processed files count and calculate progress
    ((processed_files++))
    progress=$((processed_files * 100 / pt_files_count))
    echo -ne "Progress: $progress% \r"
done

# Disable nullglob to revert back to default behavior
shopt -u nullglob

# Calculate ignored files
ignored_count=$((total_files - ${#matched_files[@]}))

# Final newline after progress
echo ""

# Display summary
echo "Operation completed."
echo "Files copied: $copied_count"
echo "Files ignored: $ignored_count"
