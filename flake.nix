{
  description = "C++/CUDA + OpenCV + ONNX development with Nix and CMake.";
  inputs = { nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; };
  outputs = { self, nixpkgs }:
    let
      allSystems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f:
        nixpkgs.lib.genAttrs allSystems (system:
          let
            enableCuda = system == "x86_64-linux";
            pkgs = import nixpkgs {
              inherit system;
              config = {
                allowUnfree = true;
                cudaSupport = enableCuda;
                cudaCapabilities = [ "8.6" ];
              };
            };
            opencvWithGui = pkgs.opencv.override { enableGtk3 = true; };
            tensorrtRoot = pkgs.symlinkJoin {
              name = "tensorrt-root";
              paths = with pkgs; [
                cudaPackages.tensorrt.include
                cudaPackages.tensorrt.lib
                cudaPackages.tensorrt.bin
                cudaPackages.tensorrt.static
              ];
            };
            onnxruntimeForSystem = if enableCuda then
              with pkgs;
              onnxruntime.overrideAttrs (oldAttrs: {
                cmakeFlags = (oldAttrs.cmakeFlags or [ ]) ++ [
                  "-Donnxruntime_USE_CUDA=ON"
                  "-Donnxruntime_USE_TENSORRT=ON"
                  "-Donnxruntime_USE_TENSORRT_BUILTIN_PARSER=ON"
                  "-Donnxruntime_CUDA_HOME=${cudaPackages.cudatoolkit}"
                  "-Donnxruntime_CUDNN_HOME=${cudaPackages.cudnn}"
                  "-Donnxruntime_TENSORRT_HOME=${tensorrtRoot}"
                ];
                buildInputs = (oldAttrs.buildInputs or [ ]) ++ [
                  cudaPackages.cudatoolkit
                  cudaPackages.cudnn
                  tensorrtRoot
                ];
              })
            else
              pkgs.onnxruntime;
            projectName = "turbo_pi_dog_tracker";
            buildDependencies = with pkgs; [ gcc14 cmake ];
            cppDependencies = with pkgs;
              [
                cxxopts
                opencvWithGui
                libcanberra-gtk3
                onnxruntimeForSystem
                glaze
                openssl
              ] ++ lib.optionals enableCuda [
                cudaPackages.cudatoolkit
                cudaPackages.cudnn
                tensorrtRoot
              ];
            cmakeCudaFlag = if enableCuda then "ON" else "OFF";
            cmakeTensorRtFlag = if enableCuda then "ON" else "OFF";
            commonEnv = with pkgs;
              ''
                export CC=${gcc14}/bin/gcc
                export CXX=${gcc14}/bin/g++
              '' + lib.optionalString enableCuda ''
                export PATH=${cudaPackages.cudatoolkit}/bin:$PATH
                export CUDA_HOME=${cudaPackages.cudatoolkit}
                export CUDA_LIB=${cudaPackages.cudatoolkit}/lib
                export CUDNN_HOME=${cudaPackages.cudnn}
                export TENSORRT_HOME=${tensorrtRoot}
              '' + ''
                export DOG_TRACKER_CMAKE_FLAGS="-DUSE_CUDA=${cmakeCudaFlag} -DUSE_TENSORRT=${cmakeTensorRtFlag}"
                export GTK_PATH=${libcanberra-gtk3}/lib/gtk-3.0''${GTK_PATH:+:$GTK_PATH}
              '';
          in f {
            inherit pkgs system enableCuda opencvWithGui onnxruntimeForSystem
              projectName buildDependencies cppDependencies cmakeCudaFlag
              cmakeTensorRtFlag commonEnv;
          });
    in {
      devShells = forAllSystems
        ({ pkgs, commonEnv, buildDependencies, cppDependencies, ... }: {
          default = pkgs.mkShell {
            packages = buildDependencies ++ cppDependencies
              ++ [ pkgs.v4l-utils ];
            shellHook = ''
              bash ./nix_env_setup.sh
              echo "You are in a GCC-based nix shell"
            '' + commonEnv;
          };
        });
      packages = forAllSystems ({ pkgs, projectName, buildDependencies
        , cppDependencies, commonEnv, ... }: {
          default = pkgs.stdenv.mkDerivation {
            name = projectName;
            src = self;
            nativeBuildInputs = buildDependencies;
            buildInputs = cppDependencies;
            configurePhase = commonEnv + ''
              mkdir build
              cd build
              cmake .. $DOG_TRACKER_CMAKE_FLAGS
            '';
            buildPhase = ''
              make
            '';
            installPhase = ''
              mkdir -p $out/bin
              cp ${projectName} $out/bin/
            '';
          };
        });
    };
}
