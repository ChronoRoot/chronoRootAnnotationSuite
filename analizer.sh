#!/bin/bash

# 1. Initialize Bind Paths
# Always bind necessary system paths
BINDS="--bind /tmp/.X11-unix --bind /run/user/$(id -u):/run/user/$(id -u)"

# 2. Check and Add Optional Paths (Autodetect /net, /media, etc)
if [ -d "/net" ]; then BINDS="$BINDS --bind /net:/net"; fi
if [ -d "/media" ]; then BINDS="$BINDS --bind /media:/media"; fi
if [ -d "/mnt" ]; then BINDS="$BINDS --bind /mnt:/mnt"; fi

# Ensure Home is bound explicitly to save configs correctly
if [ -d "$HOME" ]; then BINDS="$BINDS --bind $HOME:$HOME"; fi

# 3. Allow GUI connections
xhost +local: > /dev/null 2>&1

# 4. Run Apptainer
# We pass $BINDS which now contains the valid system paths
apptainer exec  $BINDS --env DISPLAY=$DISPLAY "image.sif" bash -c "python analyzer.py"

# 5. Cleanup
xhost -local: > /dev/null 2>&1
