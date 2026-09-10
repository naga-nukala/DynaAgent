$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

$requiredPython = "3.11"
$pythonCheck = & py -$requiredPython --version 2>$null
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($pythonCheck)) {
    throw "This project requires Python $requiredPython.x. Install Python 3.11 and then rerun this script."
}

$venvPath = Join-Path $projectDir "venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment with Python $requiredPython..."
    py -$requiredPython -m venv $venvPath
}

$pythonExe = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment Python not found: $pythonExe"
}

Write-Host "Installing requirements with Python $requiredPython..."
& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r requirements.txt

