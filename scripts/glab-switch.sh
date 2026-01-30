#!/usr/bin/env bash
# glab-switch.sh - Simple GitLab identity switching
#
# Usage:
#   glab-switch tinyland    # Switch to tinyland/Jesssullivan identity
#   glab-switch bates       # Switch to bates-ils/jsullivan2 identity
#   glab-switch             # Show current identity
#
# Prerequisites:
#   - Source your .env file with JESSSULLIVAN_GLAB_TOKEN and JSULLIVAN2_BATES_GLAB_TOKEN
#   - glab CLI installed

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

show_current() {
    echo -e "${YELLOW}Current GitLab identity:${NC}"
    if command -v glab &> /dev/null; then
        glab api /user 2>/dev/null | jq -r '"  Username: \(.username)\n  Email: \(.email)\n  Name: \(.name)"' || echo "  Not authenticated or API error"
    else
        echo "  glab CLI not found"
    fi
}

switch_to_tinyland() {
    if [[ -z "${JESSSULLIVAN_GLAB_TOKEN:-}" ]]; then
        echo -e "${RED}Error: JESSSULLIVAN_GLAB_TOKEN not set${NC}"
        echo "Source your .env file first: source /Users/jsullivan2/git/ems/.env"
        exit 1
    fi

    export GITLAB_TOKEN="$JESSSULLIVAN_GLAB_TOKEN"
    glab config set -h gitlab.com token "$GITLAB_TOKEN"

    echo -e "${GREEN}Switched to tinyland identity (Jesssullivan)${NC}"
    show_current
}

switch_to_bates() {
    if [[ -z "${JSULLIVAN2_BATES_GLAB_TOKEN:-}" ]]; then
        echo -e "${RED}Error: JSULLIVAN2_BATES_GLAB_TOKEN not set${NC}"
        echo "Source your .env file first: source /Users/jsullivan2/git/ems/.env"
        exit 1
    fi

    export GITLAB_TOKEN="$JSULLIVAN2_BATES_GLAB_TOKEN"
    glab config set -h gitlab.com token "$GITLAB_TOKEN"

    echo -e "${GREEN}Switched to bates-ils identity (jsullivan2)${NC}"
    show_current
}

case "${1:-}" in
    tinyland|personal|jess)
        switch_to_tinyland
        ;;
    bates|bates-ils|work)
        switch_to_bates
        ;;
    ""|status)
        show_current
        ;;
    *)
        echo "Usage: glab-switch [tinyland|bates]"
        echo ""
        echo "Identities:"
        echo "  tinyland  - Personal (Jesssullivan) - uses JESSSULLIVAN_GLAB_TOKEN"
        echo "  bates     - Work (jsullivan2)       - uses JSULLIVAN2_BATES_GLAB_TOKEN"
        echo ""
        echo "Run without arguments to show current identity."
        exit 1
        ;;
esac
