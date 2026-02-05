{
  description = "DAG Harness - LangGraph-based Ansible orchestration";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ self, nixpkgs, flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      perSystem = { config, pkgs, system, lib, ... }:
        let
          isDarwin = pkgs.stdenv.isDarwin;

          # Python with common development packages
          pythonEnv = pkgs.python311.withPackages (ps: with ps; [
            pip
            setuptools
            wheel
          ]);

          # Native dependencies for Python packages
          nativeDeps = with pkgs; [
            pkg-config
            openssl
            libffi
            zlib
            sqlite
          ] ++ lib.optionals isDarwin [
            pkgs.darwin.apple_sdk.frameworks.Security
            pkgs.darwin.apple_sdk.frameworks.SystemConfiguration
          ];
        in
        {
          # Development shell
          devShells.default = pkgs.mkShell {
            name = "dag-harness-dev";

            packages = with pkgs; [
              # Python
              pythonEnv
              uv

              # Tools
              git
              just
              jq
              gh       # GitHub CLI
              glab     # GitLab CLI

              # Documentation
              python311Packages.mkdocs
              python311Packages.mkdocs-material

              # Shell
              shellcheck
              direnv
            ] ++ nativeDeps;

            env = {
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook = ''
              unset PYTHONPATH
              export UV_PYTHON="${pkgs.python311}/bin/python"

              echo ""
              echo "==========================================="
              echo "DAG Harness Development Shell"
              echo "==========================================="
              echo ""
              echo "Python: $(python --version)"
              echo "UV:     $(uv --version)"
              echo ""
              echo "Quick start:"
              echo "  cd harness && uv sync --all-extras"
              echo "  just test"
              echo "  just run --help"
              echo ""
            '';
          };

          # Minimal shell for CI
          devShells.ci = pkgs.mkShell {
            name = "dag-harness-ci";
            packages = with pkgs; [ python311 uv git ];
          };

          # Formatter
          formatter = pkgs.nixpkgs-fmt;
        };
    };
}
