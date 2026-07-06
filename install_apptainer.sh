#!/bin/bash

# ============================================================================
#  ChronoRoot Annotation Suite Apptainer Installer
# ============================================================================

set -e

# --- Visual Styling ---
BOLD='\033[1m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# --- Helper Functions ---
section_title() {
    echo -e "\n${BOLD}--- $1 ---${NC}"
}

print_status() { echo -e "${BLUE}[STATUS]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

command_exists() { command -v "$1" >/dev/null 2>&1; }

# Function to check available disk space in GB
check_disk_space() {
    local required_gb=$1
    local target_dir=$2
    # Get available space in KB and convert to GB
    local available_kb=$(df -Pk "$target_dir" | awk 'NR==2 {print $4}')
    local available_gb=$((available_kb / 1024 / 1024))

    if [ "$available_gb" -lt "$required_gb" ]; then
        print_error "Insufficient disk space."
        echo "Required: ${required_gb} GB"
        echo "Available: ${available_gb} GB"
        exit 1
    else
        print_success "Disk space check passed (${available_gb} GB available)."
    fi
}

# --- Configuration ---
REPO_URL="https://github.com/ChronoRoot/chronoRootAnnotationSuite.git"
DEFAULT_INSTALL_DIR="$HOME/.local/chronoroot-annotation-suite"
DESKTOP_ENTRY_DIR="$HOME/.local/share/applications"
CONFIG_FILE="$HOME/.config/chronoRootAnnotationSuite/config.json"

main() {
    clear
    echo -e "${BOLD}ChronoRoot Annotation Suite Apptainer Installation Wizard${NC}"
    echo "This script will configure the environment and create desktop shortcuts."

    # 1. System Requirements Check
    section_title "1. Checking System Requirements"
    
    # Apptainer/Singularity Check
    if command_exists apptainer; then
        CONTAINER_CMD="apptainer"
        print_success "Container runtime: Apptainer detected."
    elif command_exists singularity; then
        CONTAINER_CMD="singularity"
        print_success "Container runtime: Singularity detected."
    else
        print_warning "Apptainer is required."
        read -p "Attempt to install Apptainer via APT? (y/n): " install_app
        if [[ $install_app =~ ^[Yy]$ ]]; then
            sudo apt update && sudo apt install -y software-properties-common
            sudo add-apt-repository -y ppa:apptainer/ppa
            sudo apt update && sudo apt install -y apptainer
            CONTAINER_CMD="apptainer"
        else
            print_error "Installation cannot proceed without a container runtime."
            exit 1
        fi
    fi

    # GPU Detection (Optional but checked)
    GPU_FLAG=""
    if command_exists nvidia-smi; then
        print_success "NVIDIA GPU detected. Hardware acceleration enabled."
        GPU_FLAG="--nv"
    fi

    # 2. Path Configuration
    section_title "2. Installation Directory"
    echo -e "Choose the directory where ChronoRoot Annotation Suite will be stored."
    echo -e "Default path: ${BLUE}$DEFAULT_INSTALL_DIR${NC}"
    echo -e "${BOLD}Action:${NC} Press ${BOLD}ENTER${NC} to use the default path, or type a new path below:"
    
    read -p "> " user_install_dir
    INSTALL_DIR="${user_install_dir:-$DEFAULT_INSTALL_DIR}"
    INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"
    
    mkdir -p "$INSTALL_DIR"
    print_success "Target directory: $INSTALL_DIR"

    # 3. Repository Setup
    section_title "3. Downloading Repository"
    REPO_DIR="$INSTALL_DIR/chronoRootAnnotationSuite"
    if [ ! -d "$REPO_DIR" ]; then
        print_status "Cloning ChronoRoot Annotation Suite source files..."
        git clone "$REPO_URL" "$REPO_DIR"
    else
        print_status "Repository exists. Updating files..."
        (cd "$REPO_DIR" && git pull)
    fi

    # 4. Container Image Configuration
    section_title "4. Container Image Setup"
    IMAGE_PATH="$INSTALL_DIR/chronoRootAnnotationSuite.sif"
    
    if [ ! -f "$IMAGE_PATH" ]; then
        echo "Please select how to set up the container image:"
        echo "1) Build from local definition file (image.def) (~1 GB)"
        echo "2) Provide path to an existing .sif file"
        
        read -p "Selection [1-2]: " img_choice
        
        case $img_choice in
            1)
                check_disk_space 2 "$INSTALL_DIR"
                print_status "Building Apptainer image from definition file..."
                # Try with --fakeroot first, if fails try sudo
                if $CONTAINER_CMD build --fakeroot "$IMAGE_PATH" "$REPO_DIR/image.def" 2>/dev/null; then
                    print_success "Image built successfully using --fakeroot."
                else
                    print_warning "Rootless build failed or --fakeroot not supported. Attempting build with sudo..."
                    sudo $CONTAINER_CMD build "$IMAGE_PATH" "$REPO_DIR/image.def"
                    # Fix permissions if built with sudo
                    sudo chmod 644 "$IMAGE_PATH"
                    sudo chown $(id -u):$(id -g) "$IMAGE_PATH"
                    print_success "Image built successfully with sudo."
                fi
                ;;
            2)
                read -p "Type or drag-and-drop the .sif file path: " existing_sif
                existing_sif=$(echo "$existing_sif" | tr -d "'\"" | xargs)
                existing_sif="${existing_sif/#\~/$HOME}"
                if [ -f "$existing_sif" ]; then
                    cp "$existing_sif" "$IMAGE_PATH"
                    print_success "Image imported."
                else
                    print_error "File not found."
                    exit 1
                fi
                ;;
            *)
                print_error "Invalid choice."
                exit 1
                ;;
        esac
    else
        print_success "Container image is already present."
    fi

    # 5. Create Application Shortcuts
    section_title "5. Creating Application Shortcuts"
    mkdir -p "$DESKTOP_ENTRY_DIR"

    wrapper_script="$INSTALL_DIR/ChronoRootAnnotationSuite.sh"
    desktop_file="$DESKTOP_ENTRY_DIR/ChronoRootAnnotationSuite.desktop"

    # Create Wrapper
    cat > "$wrapper_script" << EOF
#!/bin/bash

umask 002

# 1. Initialize Bind Paths
# Always bind necessary system paths
BINDS="--bind /tmp/.X11-unix --bind /run/user/\$(id -u):/run/user/\$(id -u)"

# 2. Check and Add Optional Paths (Autodetect /net, /media, etc)
if [ -d "/net" ]; then BINDS="\$BINDS --bind /net:/net"; fi
if [ -d "/media" ]; then BINDS="\$BINDS --bind /media:/media"; fi
if [ -d "/mnt" ]; then BINDS="\$BINDS --bind /mnt:/mnt"; fi

# Ensure Home is bound explicitly to save configs correctly
if [ -d "\$HOME" ]; then BINDS="\$BINDS --bind \$HOME:\$HOME"; fi

# 3. Allow GUI connections
xhost +local: > /dev/null 2>&1

# 4. Run Apptainer
\$CONTAINER_CMD exec $GPU_FLAG \$BINDS --env DISPLAY=\$DISPLAY "$IMAGE_PATH" \\
  bash -c "cd $REPO_DIR && python main.py"

# 5. Cleanup
xhost -local: > /dev/null 2>&1
EOF
    chmod +x "$wrapper_script"

    # Create Desktop Entry
    cat > "$desktop_file" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=ChronoRoot Annotation Suite
Exec=$wrapper_script
Icon=$REPO_DIR/logo.ico
Terminal=true
Categories=Science;Education;
EOF
    chmod +x "$desktop_file"
    print_success "Created: ChronoRoot Annotation Suite shortcut"

    # Save Config
    mkdir -p "$(dirname "$CONFIG_FILE")"
    cat > "$CONFIG_FILE" << EOF
{
  "install_dir": "$INSTALL_DIR",
  "repo_dir": "$REPO_DIR",
  "image_path": "$IMAGE_PATH",
  "container_cmd": "$CONTAINER_CMD",
  "install_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    if command_exists update-desktop-database; then
        update-desktop-database "$DESKTOP_ENTRY_DIR" 2>/dev/null || true
    fi

    section_title "Installation Complete"
    print_success "ChronoRoot Annotation Suite is successfully installed."
    echo "You can now find the application in your system menu by searching 'ChronoRoot Annotation Suite'."
    echo ""
}

main "$@"
