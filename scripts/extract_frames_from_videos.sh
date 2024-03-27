#!/bin/bash

# This scripts takes a folder of videos (.avi) and extract every 5th frame from each video
# and saves it as a grayscale JPG image. The output images are saved in a folder specified
# by the user.
# The script uses ffmpeg and xarg to reduce processing time.


usage() {
    echo "Usage: $0 <path_to_videos_folder> <output_folder>"
    echo "  <path_to_videos_folder> : Path to the folder containing AVI videos."
    echo "  <output_folder> : Path to the output folder where images will be saved."
}

# Check if exactly two arguments are given
if [ $# -ne 2 ]; then
    echo "Error: Incorrect number of arguments."
    usage
    exit 1
fi

input_folder="$1"
output_folder="$2"

export output_folder
# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

# Function to extract frames and save as grayscale JPGs
extract_frames() {
    local video_file=$1
    local video_name=$(basename -- "$video_file")
    local base_name="${video_name%.*}"
    
    # Create a specific directory for each video's frames
    local video_output_folder="$output_folder"
    
    # Output path for the frames
    local output_path="$video_output_folder/${base_name}_%05d.jpg"

    # Extract every 5th frame and save as a grayscale JPG
    ffmpeg -loglevel error -i "$video_file" -vf "select=not(mod(n\,5)),format=gray,format=yuv420p" \
           -vsync vfr "$output_path"

    echo -n "." 
}

export -f extract_frames

echo "Extracting frames from AVI videos in $input_folder and saving in $output_folder"
echo ""
echo "One dot = one video processed:"
echo ""

# Process each AVI video file
find "$input_folder" -name "*.avi" -print0 | xargs -0 -I {} -P 32 bash -c 'extract_frames "$@"' _ {}
