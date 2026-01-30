# glab-switch shell functions
# Add to your ~/.zshrc or ~/.bashrc:
#   source /path/to/glab-switch-functions.sh
#
# Also requires sourcing .env first:
#   source /Users/jsullivan2/git/ems/.env

glab-switch() {
    case "${1:-}" in
        tinyland|personal|jess)
            if [[ -z "${JESSSULLIVAN_GLAB_TOKEN:-}" ]]; then
                echo "Error: JESSSULLIVAN_GLAB_TOKEN not set"
                echo "Source your .env file first"
                return 1
            fi
            export GITLAB_TOKEN="$JESSSULLIVAN_GLAB_TOKEN"
            glab config set -h gitlab.com token "$GITLAB_TOKEN"
            echo "Switched to tinyland (Jesssullivan)"
            glab api /user 2>/dev/null | jq -r '.username' || echo "(API check failed)"
            ;;
        bates|bates-ils|work)
            if [[ -z "${JSULLIVAN2_BATES_GLAB_TOKEN:-}" ]]; then
                echo "Error: JSULLIVAN2_BATES_GLAB_TOKEN not set"
                echo "Source your .env file first"
                return 1
            fi
            export GITLAB_TOKEN="$JSULLIVAN2_BATES_GLAB_TOKEN"
            glab config set -h gitlab.com token "$GITLAB_TOKEN"
            echo "Switched to bates-ils (jsullivan2)"
            glab api /user 2>/dev/null | jq -r '.username' || echo "(API check failed)"
            ;;
        ""|status)
            echo "Current GitLab identity:"
            glab api /user 2>/dev/null | jq -r '"  \(.username) (\(.email))"' || echo "  Not authenticated"
            ;;
        *)
            echo "Usage: glab-switch [tinyland|bates]"
            echo "  tinyland - Personal (Jesssullivan)"
            echo "  bates    - Work (jsullivan2)"
            ;;
    esac
}
