#!/usr/bin/env bash
# run.sh — download repo, ensure data present, setup venv, install deps, run project
set -euo pipefail

echo "=== SMS Spam Detection: Auto-download + Run (no local clone required) ==="

WORKDIR=$(mktemp -d)
echo "Using temp directory: $WORKDIR"
cd "$WORKDIR"

# 1) Download latest repo zip from GitHub and extract
echo "Downloading repository from GitHub..."
REPO_ZIP_URL="https://github.com/kartheekpeddireddy/SMS-Spam-Detection/archive/refs/heads/main.zip"
if command -v wget >/dev/null 2>&1; then
  wget -q "$REPO_ZIP_URL" -O repo.zip
elif command -v curl >/dev/null 2>&1; then
  curl -L -s "$REPO_ZIP_URL" -o repo.zip
else
  echo "ERROR: wget or curl is required to fetch the repo."
  exit 1
fi

echo "Extracting repository..."
unzip -q repo.zip
# The repo extracts to SMS-Spam-Detection-main (expected)
cd SMS-Spam-Detection-main

# 2) Ensure data/sms_spam.tsv exists — if not, download and prepare it
DATA_PATH="data/sms_spam.tsv"
if [ -f "$DATA_PATH" ]; then
  echo "Data file found: $DATA_PATH"
else
  echo "Data file not found. Downloading SMS Spam dataset from UCI..."
  mkdir -p data
  UCI_ZIP_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

  # download
  if command -v wget >/dev/null 2>&1; then
    wget -q "$UCI_ZIP_URL" -O /tmp/smsspamcollection.zip
  else
    curl -L -s "$UCI_ZIP_URL" -o /tmp/smsspamcollection.zip
  fi

  # extract the single file "SMSSpamCollection"
  unzip -p /tmp/smsspamcollection.zip SMSSpamCollection > /tmp/SMSSpamCollection.raw || {
    echo "Unexpected zip structure — trying to extract all and locate the file..."
    unzip -q /tmp/smsspamcollection.zip -d /tmp/sms_data
    if [ -f /tmp/sms_data/SMSSpamCollection ]; then
      cp /tmp/sms_data/SMSSpamCollection /tmp/SMSSpamCollection.raw
    else
      echo "ERROR: Could not find SMSSpamCollection inside the downloaded archive."
      exit 1
    fi
  }

  # Normalize encoding/newlines and move to expected path
  echo "Preparing data file at $DATA_PATH ..."
  if command -v iconv >/dev/null 2>&1; then
    iconv -f utf-8 -t utf-8 /tmp/SMSSpamCollection.raw -o /tmp/SMSSpamCollection.utf8 || cp /tmp/SMSSpamCollection.raw /tmp/SMSSpamCollection.utf8
  else
    cp /tmp/SMSSpamCollection.raw /tmp/SMSSpamCollection.utf8
  fi
  tr -d '\r' < /tmp/SMSSpamCollection.utf8 > "$DATA_PATH"

  echo "Data saved to $DATA_PATH"
fi

# 3) Setup venv
if [ ! -d "venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv venv
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate

# 4) Install requirements if available (skip otherwise but warn)
if [ -f requirements.txt ]; then
  echo "Installing Python dependencies from requirements.txt..."
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "WARNING: requirements.txt not found. Proceeding without pip installs."
fi

# 5) Run main.py (pass through any args)
echo "Running main.py..."
python main.py "$@"

echo "=== DONE ==="

