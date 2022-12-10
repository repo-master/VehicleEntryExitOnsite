@echo off
REM Check if python is present
REM TODO

echo Current directory is "%cd%"
echo Make sure this script is run from the program's folder, as 'scripts/windows.bat'

REM Install light-the-torch, used to install torch with cuda compatibility
python -m pip install light-the-torch

ltt install -r requirements.txt