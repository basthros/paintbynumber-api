#!/bin/bash
set -e

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt-get install -y nodejs

echo "Cloning frontend repository..."
git clone --depth 1 --branch claude/paint-by-number-setup-01RDLkxBUEAGcBRh5awYyHbx https://github.com/basthros/paintbynumber-app.git /tmp/frontend

echo "Building frontend..."
cd /tmp/frontend
npm install
npm run build

echo "Copying frontend build to dist folder..."
mkdir -p /opt/render/project/src/dist
cp -r dist/* /opt/render/project/src/dist/

echo "Build complete!"
