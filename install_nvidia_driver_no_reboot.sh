#!/bin/bash

# Check if the script is running as root
if [ "$(id -u)" != "0" ]; then
    echo "This script must be run as root" >&2
    exit 1
fi

# Update the system
apt-get update
apt-get upgrade -y

# Install required packages
apt-get install -y software-properties-common

# Add the graphics drivers PPA
add-apt-repository ppa:graphics-drivers/ppa -y
apt-get update

# Install the latest NVIDIA driver
apt-get install -y nvidia-driver-470

# Switch to TTY
chvt 1

# Stop the display manager
systemctl stop gdm || systemctl stop lightdm || systemctl stop sddm

# Unload the old NVIDIA driver
nvidia-uninstall || true

# Load the new NVIDIA driver
modprobe nvidia

# Start the display manager
systemctl start gdm || systemctl start lightdm || systemctl start sddm

# Switch back to the graphical session
chvt 7

echo "The NVIDIA driver has been installed without rebooting. Please verify if everything works as expected."
