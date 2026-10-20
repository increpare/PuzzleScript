#!/usr/bin/env bash
set -eu

python3 "$(dirname -- "${BASH_SOURCE[0]}")/runserver.py"
