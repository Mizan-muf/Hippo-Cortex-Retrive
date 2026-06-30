#Requires -Version 5.1
<#
.SYNOPSIS
    Reset Hippo-Cortex-Retrieval data stores.
.DESCRIPTION
    Deletes all three Qdrant collections (knowledge_base, knowledge_base_entities,
    mem0migrations), removes the Kuzu graph database directory, and removes the
    knowledge-store file store. Does NOT remove .venv or source files.
    Run setup.ps1 again after this to reinitialise services.
#>

$ErrorActionPreference = "SilentlyContinue"
$root = $PSScriptRoot

foreach ($c in @("knowledge_base", "knowledge_base_entities", "mem0migrations")) {
    $r = Invoke-RestMethod -Method Delete -Uri "http://localhost:6333/collections/$c"
    Write-Host "Qdrant $c`: $($r.status)"
}

Remove-Item -Recurse -Force "$root\data\graph"
Remove-Item -Recurse -Force "$root\knowledge-store"

$n            = (Invoke-RestMethod -Uri "http://localhost:6333/collections").result.collections.Count
$graphExists  = Test-Path "$root\data\graph"
$storeExists  = Test-Path "$root\knowledge-store"
Write-Host "Done - Qdrant: $n collections | graph: $graphExists | filestore: $storeExists"
