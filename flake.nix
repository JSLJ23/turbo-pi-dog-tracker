{
  description = "C++/CUDA + OpenCV + ONNX development with Nix and CMake.";

  # Flake inputs
  inputs = { nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; };

  # Flake outputs
  outputs = { self, nixpkgs }:
    let
      # Systems supported
      allSystems = [
        "x86_64-linux" # 64-bit Intel/AMD Linux
        "aarch64-linux" # For rasberry pi 5
      ];

      # Helper to provide system-specific attributes
      forAllSystems = f:
        nixpkgs.lib.genAttrs allSystems (system:
          let enableCuda = system == "x86_64-linux";
          in f {
            pkgs = import nixpkgs {
              inherit system;
              config = {
                allowUnfree = true;
                cudaSupport = enableCuda;
                cudaCapabilities = [ "8.6" ];
              };
            };
            inherit system enableCuda;
          });
    in {
      # Development environment output
      devShells = forAllSystems ({ pkgs, system, enableCuda, ... }:
        let opencvWithGui = pkgs.opencv.override { enableGtk3 = true; };
        in {
          default = pkgs.mkShell {
            # The Nix packages provided in the environment
            packages = with pkgs;
              [
                gcc14 # The GNU Compiler Collection
                cmake
                cxxopts
                opencvWithGui
                libcanberra-gtk3
                v4l-utils
                onnxruntime
                glaze
                openssl
              ] ++ lib.optionals enableCuda [ cudaPackages.cudatoolkit ];

            shellHook = ''
              bash ./nix_env_setup.sh
              echo "You are in a GCC-based nix shell"
              export CC=${pkgs.gcc14}/bin/gcc
              export CXX=${pkgs.gcc14}/bin/g++
            '' + pkgs.lib.optionalString enableCuda ''
              export PATH=${pkgs.cudaPackages.cudatoolkit}/bin:$PATH
              export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}
              export CUDA_LIB=${pkgs.cudaPackages.cudatoolkit}/lib
            '' + ''
              export DOG_TRACKER_CMAKE_FLAGS="-DUSE_CUDA=${
                if enableCuda then "ON" else "OFF"
              }"
              export GTK_PATH=${pkgs.libcanberra-gtk3}/lib/gtk-3.0''${GTK_PATH:+:$GTK_PATH}
            '';
          };
        });

      packages = forAllSystems ({ pkgs, enableCuda, ... }: {
        default = let
          opencvWithGui = pkgs.opencv.override { enableGtk3 = true; };
          buildDependencies = with pkgs; [ gcc14 cmake ];
          cppDependencies = with pkgs;
            [ cxxopts opencvWithGui libcanberra-gtk3 onnxruntime glaze openssl ]
            ++ lib.optionals enableCuda [ cudaPackages.cudatoolkit ];
          projectName = "turbo_pi_dog_tracker";
        in pkgs.stdenv.mkDerivation {
          name = projectName;
          src = self;
          nativeBuildInputs = buildDependencies;
          buildInputs = cppDependencies;
          configurePhase = ''
            export CC=${pkgs.gcc14}/bin/gcc
            export CXX=${pkgs.gcc14}/bin/g++
          '' + pkgs.lib.optionalString enableCuda ''
            export PATH=${pkgs.cudaPackages.cudatoolkit}/bin:$PATH
            export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}
            export CUDA_LIB=${pkgs.cudaPackages.cudatoolkit}/lib
          '' + ''
            mkdir build && cd build
            cmake ../ -DUSE_CUDA=${if enableCuda then "ON" else "OFF"}
          '';
          buildPhase = ''
            make
          '';
          installPhase = ''
            mkdir -p $out
            cp -r ${projectName} $out/
          '';
        };
      });
    };
}
