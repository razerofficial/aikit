#!/bin/bash

# Path to the directory
TARGET_DIR="$HOME/.cache/huggingface"

# Use current user's username and group
# Get ID only as the name may not exist when using --user
EXPECTED_USER=$(id -u)
EXPECTED_GROUP=$(id -g)

# Check if the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Directory $TARGET_DIR does not exist. Skipping."
    return
fi

# Get current owner and group of the directory
OWNER=$(stat -c "%u" "$TARGET_DIR")
GROUP=$(stat -c "%g" "$TARGET_DIR")

# Change ownership if needed
if [[ "$OWNER" != "$EXPECTED_USER" || "$GROUP" != "$EXPECTED_GROUP" ]]; then
    echo "Changing ownership of $TARGET_DIR from $OWNER:$GROUP to $EXPECTED_USER:$EXPECTED_GROUP"
    sudo chown -R "$EXPECTED_USER:$EXPECTED_GROUP" "$TARGET_DIR"
fi
