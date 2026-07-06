#!/bin/bash

# ============================================================================
#  ChronoRoot Annotation Suite Conda Installer
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
section_title() { echo -e "\n${BOLD}--- $1 ---${NC}"; }
print_status() { echo -e "${BLUE}[STATUS]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

command_exists() { command -v "$1" >/dev/null 2>&1; }

# --- Configuration ---
REPO_URL="https://github.com/ChronoRoot/chronoRootAnnotationSuite.git"
DEFAULT_INSTALL_DIR="$HOME/.local/chronoroot-annotation-suite"
DESKTOP_ENTRY_DIR="$HOME/.local/share/applications"
CONFIG_FILE="$HOME/.config/chronoRootAnnotationSuite/config.json"

main() {
    clear
    echo -e "${BOLD}ChronoRoot Annotation Suite Conda Installation Wizard${NC}"
    
    # 1. Prerequisite Check
    section_title "1. Checking System Prerequisites"
    
    if ! command_exists conda; then
        print_error "Conda not found. Please install Miniconda or Anaconda first."
        exit 1
    fi
    print_success "Conda detected."

    # 2. Setup Directory
    section_title "2. Directory Setup"
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

    # 4. Conda Environment Setup
    section_title "4. Conda Environment Setup"
    ENV_NAME="chronoRootAnnotation"
    ENV_FILE="$REPO_DIR/environment.yml"
    
    if conda env list | grep -q "$ENV_NAME"; then
        print_status "Updating existing environment ($ENV_NAME)..."
        conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
    else
        print_status "Creating new environment ($ENV_NAME)..."
        conda env create -n "$ENV_NAME" -f "$ENV_FILE"
    fi

    # 5. Launcher Generation
    section_title "5. Creating Desktop Launchers"
    CONDA_BASE=$(conda info --base)
    
    wrapper_script="$INSTALL_DIR/ChronoRootAnnotationSuite.sh"
    desktop_file="$DESKTOP_ENTRY_DIR/ChronoRootAnnotationSuite.desktop"

    # Create Wrapper
    cat > "$wrapper_script" << EOF
#!/bin/bash

umask 002
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_NAME
cd "$REPO_DIR"
python main.py
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
  "env_name": "$ENV_NAME",
  "install_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    # Refresh desktop database
    if command_exists update-desktop-database; then
        update-desktop-database "$DESKTOP_ENTRY_DIR" 2>/dev/null || true
    fi

    section_title "Installation Complete"
    print_success "ChronoRoot Annotation Suite is ready at $INSTALL_DIR"
    echo "You can now find the application in your system menu by searching 'ChronoRoot Annotation Suite'."
}

main "$@"
