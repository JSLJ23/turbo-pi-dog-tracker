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

## Demo mode
```bash
./build/turbo_pi_dog_tracker demo \
  --camera 2 \
  --confidence 0.80 \
  --nms 0.30 \
  --model models/yolo26n_640_bs1.onnx
```

## Server mode
Server mode serves TCP telemetry on `127.0.0.1:8765` by default. Override the bind address with
`TELEMETRY_HOST` and `TELEMETRY_PORT`.
```bash
TELEMETRY_HOST=127.0.0.1 \
TELEMETRY_PORT=8765 \
./build/turbo_pi_dog_tracker server \
  --camera 0 \
  --confidence 0.80 \
  --nms 0.30 \
  --model models/yolo26n_640_bs1.onnx
```

## Render mode
https://github.com/user-attachments/assets/f82b1cb9-7a42-40ee-a319-4e7ce0503a7f
```bash
./build/turbo_pi_dog_tracker render \
  --input-video data/IMG_7243.mov \
  --output-video data/IMG_7243_with_overlay.mov \
  --confidence 0.80 \
  --nms 0.30 \
  --model models/yolo26x_1280_bs32.onnx
```
