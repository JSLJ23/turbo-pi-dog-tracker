env | grep -E '^(CMAKE_INCLUDE_PATH|CMAKE_LIBRARY_PATH|NIXPKGS_CMAKE_PREFIX_PATH)=' \
  | sed 's/^NIXPKGS_CMAKE_PREFIX_PATH=/CMAKE_PREFIX_PATH=/' \
  | sed 's/^/export /' \
  > nix_env_variables.sh

if env | grep -q '^SDKROOT='; then
  env | grep '^SDKROOT=' | sed 's/^/export /' >> nix_env_variables.sh
fi