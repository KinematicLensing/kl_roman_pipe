#!/usr/bin/env bash
# Script to download data files from CyVerse using WebDAV
# Usage: ./download_cyverse_data.sh [config_file]
#
# Authentication:
# - Public data (dav-anon URLs): No authentication needed
# - Private data: Script will prompt to set up ~/.netrc on first use
# - Can also use CYVERSE_USER and CYVERSE_PASS environment variables
#
# This script will automatically prompt to set up ~/.netrc if needed.

set -e  # Exit on error

# Configuration
CONFIG_FILE="${1:-data/cyverse/cyverse_data.conf}"
DATA_DIR="data"
CURL_OPTS="--retry 5 --retry-delay 3 --connect-timeout 30 --max-time 300 --netrc-optional"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if any private URLs are in config
has_private_urls() {
    grep -v "^[[:space:]]*#" "$CONFIG_FILE" 2>/dev/null | \
        grep -v "^[[:space:]]*$" | \
        grep -q "https://data.cyverse.org/dav/iplant" && \
        ! grep -q "dav-anon"
    return $?
}

# Function to check and setup .netrc
setup_netrc_if_needed() {
    local netrc_file="$HOME/.netrc"
    local needs_setup=false
    
    # Check if config has private URLs that need auth
    if ! grep -v "^[[:space:]]*#" "$CONFIG_FILE" 2>/dev/null | \
         grep -v "^[[:space:]]*$" | \
         grep -v "dav-anon" | \
         grep -q "https://data.cyverse.org/dav"; then
        # No private URLs found
        return 0
    fi
    
    # Check if .netrc exists and has CyVerse entry
    if [ ! -f "$netrc_file" ]; then
        needs_setup=true
        echo -e "${YELLOW}Warning: ~/.netrc file not found${NC}"
    elif ! grep -q "machine data.cyverse.org" "$netrc_file" 2>/dev/null; then
        needs_setup=true
        echo -e "${YELLOW}Warning: No CyVerse credentials in ~/.netrc${NC}"
    else
        # Check permissions
        local perms=$(stat -f "%Lp" "$netrc_file" 2>/dev/null || stat -c "%a" "$netrc_file" 2>/dev/null)
        if [ "$perms" != "600" ]; then
            echo -e "${YELLOW}Warning: ~/.netrc has insecure permissions ($perms), fixing...${NC}"
            chmod 600 "$netrc_file"
            echo -e "${GREEN}Set ~/.netrc permissions to 600${NC}"
        fi
        return 0
    fi
    
    # Check if env vars are set
    if [ -n "$CYVERSE_USER" ] && [ -n "$CYVERSE_PASS" ]; then
        echo -e "${BLUE}Using CYVERSE_USER and CYVERSE_PASS environment variables${NC}"
        return 0
    fi
    
    # Offer to set up .netrc
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}CyVerse Authentication Setup${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Your configuration requires private CyVerse data."
    echo "To download these files, you need CyVerse credentials."
    echo ""
    echo "Options:"
    echo "  1. Set up ~/.netrc (recommended - secure, automatic)"
    echo "  2. Skip authentication (will fail for private files)"
    echo "  3. Cancel and set CYVERSE_USER/CYVERSE_PASS env vars manually"
    echo ""
    read -p "Choose [1/2/3]: " -n 1 -r choice
    echo ""
    
    case $choice in
        1)
            echo ""
            echo -e "${GREEN}Setting up ~/.netrc for CyVerse...${NC}"
            read -p "CyVerse username: " username
            read -s -p "CyVerse password: " password
            echo ""
            
            if [ -z "$username" ] || [ -z "$password" ]; then
                echo -e "${RED}Error: Username or password cannot be empty${NC}"
                return 1
            fi
            
            # Create or append to .netrc
            if [ -f "$netrc_file" ]; then
                echo "" >> "$netrc_file"
            fi
            echo "machine data.cyverse.org login $username password $password" >> "$netrc_file"
            chmod 600 "$netrc_file"
            
            echo -e "${GREEN}Credentials saved to ~/.netrc${NC}"
            echo -e "${GREEN}Permissions set to 600 (secure)${NC}"
            echo ""
            ;;
        2)
            echo -e "${YELLOW}Skipping authentication - private downloads will fail${NC}"
            ;;
        3)
            echo ""
            echo "To set environment variables, run:"
            echo "  export CYVERSE_USER='your_username'"
            echo "  export CYVERSE_PASS='your_password'"
            echo "Then run this script again."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Exiting.${NC}"
            exit 1
            ;;
    esac
}

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file '$CONFIG_FILE' not found${NC}"
    echo "Please create '$CONFIG_FILE' with the data files to download"
    exit 1
fi

# Setup authentication if needed
setup_netrc_if_needed

# Function to build curl command with authentication
build_curl_cmd() {
    local url="$1"
    local output_file="$2"
    
    if [[ "$url" == *"dav-anon"* ]]; then
        # Public anonymous access - no auth needed
        echo "curl $CURL_OPTS -f -o \"$output_file\" \"$url\""
    else
        # Private data - curl will use ~/.netrc automatically with --netrc flag
        # Fall back to env vars if provided
        if [ -n "$CYVERSE_USER" ] && [ -n "$CYVERSE_PASS" ]; then
            echo "curl $CURL_OPTS -f --user \"$CYVERSE_USER:$CYVERSE_PASS\" -o \"$output_file\" \"$url\""
        else
            # Let curl handle it via .netrc
            echo "curl $CURL_OPTS -f -o \"$output_file\" \"$url\""
        fi
    fi
}

# Parse config file and download each file
echo -e "${GREEN}Starting CyVerse data download...${NC}"
downloaded=0
skipped=0
failed=0

while IFS='|' read -r remote_url local_path || [ -n "$remote_url" ]; do
    # Skip empty lines and comments
    [[ -z "$remote_url" || "$remote_url" =~ ^[[:space:]]*# ]] && continue
    
    # Trim whitespace
    remote_url=$(echo "$remote_url" | xargs)
    local_path=$(echo "$local_path" | xargs)
    
    # Construct local file path
    local_file="${DATA_DIR}/${local_path}"
    local_dir=$(dirname "$local_file")
    
    # Create local directory if needed
    mkdir -p "$local_dir"
    
    # Check if file already exists
    if [ -f "$local_file" ]; then
        echo -e "${YELLOW}Skipping (already exists): $local_path${NC}"
        ((skipped++))
        continue
    fi
    
    # Download the file
    echo -e "${GREEN}Downloading: $local_path${NC}"
    echo "   From: $remote_url"
    
    # Build and execute curl command
    curl_cmd=$(build_curl_cmd "$remote_url" "$local_file")
    if eval "$curl_cmd" 2>&1; then
        echo -e "${GREEN}Downloaded successfully${NC}"
        ((downloaded++))
    else
        echo -e "${RED}Failed to download: $local_path${NC}"
        rm -f "$local_file"  # Remove partial download
        ((failed++))
    fi
    echo ""
    
done < "$CONFIG_FILE"

# Summary
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Download Summary:${NC}"
echo -e "  Downloaded: ${GREEN}$downloaded${NC}"
echo -e "  Skipped:    ${YELLOW}$skipped${NC}"
echo -e "  Failed:     ${RED}$failed${NC}"
echo -e "${GREEN}================================${NC}"

if [ $failed -gt 0 ]; then
    echo -e "${RED}Some downloads failed. Please check the URLs in $CONFIG_FILE${NC}"
    exit 1
fi

exit 0
