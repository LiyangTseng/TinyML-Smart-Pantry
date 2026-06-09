# Create and populate a Python virtual environment for this project on Windows (PowerShell)
$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $Here ".venv"

# We check for compatible python installations.
# TensorFlow 2.15+ works best with Python 3.10 or 3.11.
# We look for 'py' launcher to select python 3.10 or 3.11, or default to 'python'.
$PythonCmd = "python"
if (Get-Command "py" -ErrorAction SilentlyContinue) {
    $Versions = & py --list
    if ($Versions -match "3\.10") {
        $PythonCmd = "py -3.10"
        Write-Host "Using Python 3.10 via launcher"
    } elseif ($Versions -match "3\.11") {
        $PythonCmd = "py -3.11"
        Write-Host "Using Python 3.11 via launcher"
    }
}

if (Test-Path $VenvDir) {
    Write-Host "Using existing virtualenv at $VenvDir"
} else {
    Write-Host "Creating virtualenv at $VenvDir using $PythonCmd..."
    Invoke-Expression "$PythonCmd -m venv `"$VenvDir`""
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "Upgrading pip, setuptools, and wheel inside virtualenv..."
& $VenvPython -m pip install --upgrade pip setuptools wheel

Write-Host "Installing project requirements into virtualenv..."
$ReqPath = Join-Path $Here "requirements.txt"
& $VenvPython -m pip install -r $ReqPath

Write-Host ""
Write-Host "Virtual environment created and configured!"
Write-Host "To activate locally in PowerShell run:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then run commands normally, e.g.:"
Write-Host "  python data/download_food101.py --output-dir artifacts/food101_full --label-map data/label_map.json --max-per-class 1000"
Write-Host ""
Write-Host "To remove the virtualenv run:"
Write-Host "  Remove-Item -Recurse -Force .venv"
