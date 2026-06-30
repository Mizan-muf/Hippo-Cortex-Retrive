#Requires -Version 5.1
<#
.SYNOPSIS
    One-click setup for Hippo-Cortex-Retrieval.
.DESCRIPTION
    Checks prerequisites, creates a virtual environment, installs Python
    dependencies, downloads spaCy models, pulls required Ollama models, and
    starts the Qdrant Docker container.
.PARAMETER SkipOllama
    Skip Ollama model pulls (useful if you only use Gemini and have no local models).
#>
param([switch]$SkipOllama)

Set-Location $PSScriptRoot
$ErrorActionPreference = "Stop"

function Step  { Write-Host "" ; Write-Host ">> $args" -ForegroundColor Cyan }
function OK    { Write-Host "   [OK]   $args" -ForegroundColor Green }
function Warn  { Write-Host "   [WARN] $args" -ForegroundColor Yellow }
function Fail  { Write-Host "   [FAIL] $args" -ForegroundColor Red; exit 1 }
function Info  { Write-Host "         $args" -ForegroundColor Gray }

Write-Host ""
Write-Host "  Hippo-Cortex-Retrieval - Setup" -ForegroundColor Magenta
Write-Host "  ==============================" -ForegroundColor Magenta
Write-Host ""

# --- 1. Python 3.10+ ----------------------------------------------------------
Step "Python version check"
try {
    $pyver = python --version 2>&1
    if ($pyver -match "Python (\d+)\.(\d+)") {
        $maj = [int]$Matches[1]; $min = [int]$Matches[2]
        if ($maj -lt 3 -or ($maj -eq 3 -and $min -lt 10)) {
            Fail "Python 3.10+ required, found: $pyver. Install from https://python.org"
        }
        OK $pyver
    } else {
        Fail "Could not parse Python version output: $pyver"
    }
} catch {
    Fail "Python not found in PATH. Install from https://python.org and re-run."
}

# --- 2. Docker ----------------------------------------------------------------
Step "Docker Desktop check"
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        Fail "Docker is installed but not running. Start Docker Desktop and re-run."
    }
    OK "Docker is running"
} catch {
    Fail "Docker not found. Install Docker Desktop from https://docker.com and re-run."
}

# --- 3. Read .env for LLM provider + model names -----------------------------
Step "Reading .env configuration"
$llmProvider = "gemini"
$embedModel  = "nomic-embed-text"
$lightLLM    = "llama3.2"
$mem0LLM     = "phi3.5"

if (Test-Path ".env") {
    $raw = Get-Content ".env" -Raw
    if ($raw -match "(?m)^LLM_PROVIDER\s*=\s*(.+)$")    { $llmProvider = $Matches[1].Trim() }
    if ($raw -match "(?m)^EMBED_MODEL\s*=\s*(.+)$")      { $embedModel  = $Matches[1].Trim() }
    if ($raw -match "(?m)^LIGHTWEIGHT_LLM\s*=\s*(.+)$")  { $lightLLM    = $Matches[1].Trim() }
    if ($raw -match "(?m)^MEM0_LLM\s*=\s*(.+)$")         { $mem0LLM     = $Matches[1].Trim() }
    OK "LLM_PROVIDER=$llmProvider  EMBED_MODEL=$embedModel"
} else {
    Warn ".env not found - defaults used (LLM_PROVIDER=gemini). Configure .env before running."
}

# --- 4. Ollama ----------------------------------------------------------------
Step "Ollama check"
$ollamaOK = $false
try {
    $r = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
    if ($r.StatusCode -eq 200) { $ollamaOK = $true; OK "Ollama daemon is running at http://localhost:11434" }
} catch {}

if (-not $ollamaOK) {
    if ($llmProvider -eq "local") {
        Fail "Ollama required for LLM_PROVIDER=local. Install from https://ollama.com and start the daemon."
    } else {
        Warn "Ollama not running (OK for Gemini provider, but embeddings may still need it)."
        Warn "If you switch to LLM_PROVIDER=local, install Ollama from https://ollama.com"
    }
}

# --- 5. Virtual environment ---------------------------------------------------
Step "Virtual environment"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Info "Creating .venv ..."
    python -m venv .venv
    OK "Created .venv"
} else {
    OK ".venv already exists"
}

& ".venv\Scripts\Activate.ps1"
OK "Activated .venv"

# --- 6. Python dependencies ---------------------------------------------------
Step "Installing Python dependencies (this takes a few minutes on first run)"
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
OK "All packages installed"

# --- 7. spaCy model -----------------------------------------------------------
Step "Downloading spaCy language model (en_core_web_trf ~500 MB, requires torch)"
python -m spacy download en_core_web_trf
OK "en_core_web_trf ready"

# --- 8. Ollama models ---------------------------------------------------------
if ($ollamaOK -and -not $SkipOllama) {
    Step "Pulling Ollama models"

    Info "Pulling embedding model: $embedModel"
    ollama pull $embedModel
    OK "$embedModel pulled"

    if ($llmProvider -eq "local") {
        Info "Pulling lightweight LLM: $lightLLM"
        ollama pull $lightLLM
        OK "$lightLLM pulled"

        if ($mem0LLM -ne $lightLLM) {
            Info "Pulling Mem0 LLM: $mem0LLM"
            ollama pull $mem0LLM
            OK "$mem0LLM pulled"
        }
    } else {
        Info "LLM_PROVIDER=$llmProvider - skipping LLM model pulls (embedding only)."
    }
}

# --- 9. Qdrant container ------------------------------------------------------
Step "Starting Qdrant (docker-compose)"
docker-compose up -d
OK "Container started"

Info "Waiting for Qdrant health check ..."
$ready = $false
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:6333/healthz" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
}

if ($ready) { OK "Qdrant is healthy at http://localhost:6333" }
else        { Warn "Qdrant did not respond within 40 s. Check: docker logs app-qdrant" }

# --- Done ---------------------------------------------------------------------
Write-Host ""
Write-Host "  ==============================" -ForegroundColor Magenta
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "  ==============================" -ForegroundColor Magenta
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Cyan
Write-Host "    1. Edit .env - set GEMINI_API_KEY if using the Gemini provider."
Write-Host "    2. Run the app:  .\start.ps1          (smoke test + REPL)"
Write-Host "    3. Smoke only:   .\start.ps1 --smoke"
Write-Host "    4. REPL only:    .\start.ps1 --repl"
Write-Host "    5. With ingest:  .\start.ps1 --ingest"
Write-Host "    6. Ingest EPUB:  python scripts\ingest_epub.py samples\The_Strongest_Gene.epub --chapters 1-3"
Write-Host ""
Write-Host "  NOTE: On first run, fastcoref downloads its neural model (approx. 200 MB)." -ForegroundColor Yellow
Write-Host ""
