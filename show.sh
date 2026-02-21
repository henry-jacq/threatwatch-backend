#!/bin/bash

find . \
  \( \
    -path "./data" -o \
    -path "./.git" -o \
    -path "./models" -o \
    -path "./venv" -o \
    -path "./old_codes" \
  \) -prune -o \
  -type f \
  -exec grep -Iq . {} \; \
  -print |
while IFS= read -r file; do
  echo "=================================================="
  echo "FILE : $file"
  echo "LINES: $(wc -l < "$file")"
  echo "--------------------------------------------------"
  cat "$file"
  echo
done
