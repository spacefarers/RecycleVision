#!/bin/bash
# Helper script to run quantize.py with proper .NET 7 environment

export DOTNET_ROOT="/opt/homebrew/opt/dotnet7/libexec"
export PATH="$DOTNET_ROOT/bin:$PATH"

python quantize.py "$@"
