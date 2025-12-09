#!/bin/bash

# Path to the directory
TARGET_DIR="$HOME/.cache/huggingface"

# Use current user's username and group
EXPECTED_USER=$(id -un)
EXPECTED_GROUP=$(id -gn)

# Check if the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Directory $TARGET_DIR does not exist. Skipping."
    return
fi

# Get current owner and group of the directory
OWNER=$(stat -c "%U" "$TARGET_DIR")
GROUP=$(stat -c "%G" "$TARGET_DIR")

# Change ownership if needed
if [[ "$OWNER" != "$EXPECTED_USER" || "$GROUP" != "$EXPECTED_GROUP" ]]; then
    echo "Changing ownership of $TARGET_DIR from $OWNER:$GROUP to $EXPECTED_USER:$EXPECTED_GROUP"
    sudo chown -R "$EXPECTED_USER:$EXPECTED_GROUP" "$TARGET_DIR"
fi
