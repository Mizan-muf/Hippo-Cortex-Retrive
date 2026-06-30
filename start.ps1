#Requires -Version 5.1
<#
.SYNOPSIS
    Start Hippo-Cortex-Retrieval.
.DESCRIPTION
    Ensures Qdrant is running, activates the virtual environment, and launches
    main.py. Any arguments are forwarded to main.py.
.EXAMPLE
    .\start.ps1              # smoke test + interactive REPL
    .\start.ps1 --smoke      # smoke test only
    .\start.ps1 --repl       # REPL only
    .\start.ps1 --ingest     # include full pipeline ingest demo
#>
param([Parameter(ValueFromRemainingArguments)][string[]]$AppArgs)

Set-Location $PSScriptRoot
$ErrorActionPreference = "Continue"

function Step { Write-Host "" ; Write-Host ">> $args" -ForegroundColor Cyan }
function OK   { Write-Host "   [OK]   $args" -ForegroundColor Green }
function Warn { Write-Host "   [WARN] $args" -ForegroundColor Yellow }
function Fail { Write-Host "   [FAIL] $args" -ForegroundColor Red; exit 1 }
function Info { Write-Host "         $args" -ForegroundColor Gray }

Write-Host ""
Write-Host "  Hippo-Cortex-Retrieval - Start" -ForegroundColor Magenta
Write-Host "  ==============================" -ForegroundColor Magenta
Write-Host ""

# --- 1. Virtual environment ---------------------------------------------------
Step "Virtual environment"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Fail ".venv not found. Run .\setup.ps1 (or setup.bat) first."
}
& ".venv\Scripts\Activate.ps1"
OK "Activated .venv"

# --- 2. Qdrant ----------------------------------------------------------------
Step "Qdrant vector store"
$qdrantUp = $false
try {
    $r = Invoke-WebRequest -Uri "http://localhost:6333/healthz" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
    if ($r.StatusCode -eq 200) { $qdrantUp = $true }
} catch {}

if ($qdrantUp) {
    OK "Qdrant already running at http://localhost:6333"
} else {
    Info "Qdrant not detected - attempting docker-compose up -d ..."
    try {
        docker-compose up -d 2>&1 | Out-Null
        Info "Waiting for Qdrant to become healthy ..."
        for ($i = 0; $i -lt 15; $i++) {
            Start-Sleep -Seconds 2
            try {
                $r = Invoke-WebRequest -Uri "http://localhost:6333/healthz" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
                if ($r.StatusCode -eq 200) { $qdrantUp = $true; break }
            } catch {}
        }
        if ($qdrantUp) { OK "Qdrant started and healthy" }
        else {
            Warn "Qdrant did not respond after 30 s."
            Warn "Check Docker Desktop is running, then: docker logs app-qdrant"
        }
    } catch {
        Warn "Could not start Qdrant: $($_.Exception.Message)"
        Warn "Make sure Docker Desktop is running, then retry."
    }
}

# --- 3. Ollama (informational only) ------------------------------------------
Step "Ollama LLM daemon"
try {
    $r = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
    if ($r.StatusCode -eq 200) { OK "Ollama is running at http://localhost:11434" }
    else { throw }
} catch {
    $provider = "gemini"
    if (Test-Path ".env") {
        $raw = Get-Content ".env" -Raw
        if ($raw -match "(?m)^LLM_PROVIDER\s*=\s*(.+)$") { $provider = $Matches[1].Trim() }
    }
    if ($provider -eq "local") {
        Warn "Ollama not running - queries WILL FAIL (LLM_PROVIDER=local)."
        Warn "Start Ollama, then re-run this script."
    } else {
        Warn "Ollama not running (LLM_PROVIDER=$provider - LLM calls use Gemini)."
        Warn "Embeddings still need Ollama if EMBED_MODEL points to an Ollama model."
    }
}

# --- Launch -------------------------------------------------------------------
Write-Host ""
if ($AppArgs) {
    Write-Host "  Launching: python main.py $AppArgs" -ForegroundColor Green
} else {
    Write-Host "  Launching: python main.py" -ForegroundColor Green
}
Write-Host "  (Ctrl-C or type 'exit' in the REPL to quit)" -ForegroundColor Gray
Write-Host ""

if ($AppArgs) {
    python main.py @AppArgs
} else {
    python main.py
}
