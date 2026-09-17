#!/usr/bin/env bash
# neurafly installer: builds the release binary and installs it together
# with the FlyWire connectome data file.
#
#   binary -> ~/.local/bin/neurafly
#   data   -> ~/.local/share/neurafly/flywire_net.bin
#
# Usage: ./install.sh        (run from the repo root)
set -euo pipefail

BIN_DIR="${HOME}/.local/bin"
DATA_DIR="${HOME}/.local/share/neurafly"

echo ">> building release binary"
cargo build --release

mkdir -p "$BIN_DIR" "$DATA_DIR"
install -m 755 target/release/neurafly "$BIN_DIR/neurafly"
install -m 644 data/flywire_net.bin "$DATA_DIR/flywire_net.bin"

echo ">> installed:"
echo "   $BIN_DIR/neurafly"
echo "   $DATA_DIR/flywire_net.bin"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) echo ">> note: $BIN_DIR is not in your PATH; add it to your shell rc" ;;
esac

echo ">> done. run: neurafly"
