# turbo-pi-dog-tracker
Dog detection and tracking for Turbo Pi robot

## Nix DevShell
```bash
nix develop
```

## Build
```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=ON
make -j 8
```

## Nix CUDA fix on non NixOS
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1
```
Alternatively, symlinking the libcuda.so.1 would also work
```bash
sudo mkdir -p /run/opengl-driver/lib
sudo find /usr/lib -name 'libcuda.so*' -exec ln -s {} /run/opengl-driver/lib/ \;
```

## Server mode


## Demo mode


## Render mode
