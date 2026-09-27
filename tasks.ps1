<#
.SYNOPSIS
    Canopy task runner for PowerShell. Mirrors the Makefile target for target.

.EXAMPLE
    .\tasks.ps1 check
    .\tasks.ps1 fly -Extra '--laps','3'
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'setup', 'lock', 'lint', 'format', 'typecheck',
                 'test', 'test-cov', 'test-all', 'check', 'fly', 'view', 'props', 'clean')]
    [string]$Task = 'help',

    # Extra arguments forwarded to the underlying command.
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra = @()
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Invoke-Step {
    param([string[]]$Command)
    Write-Host "> $($Command -join ' ')" -ForegroundColor Cyan
    & $Command[0] @($Command[1..($Command.Length - 1)])
    if ($LASTEXITCODE -ne 0) {
        throw "'$($Command -join ' ')' failed with exit code $LASTEXITCODE"
    }
}

switch ($Task) {
    'help' {
        Write-Host 'Canopy tasks:' -ForegroundColor Green
        @(
            @{ n = 'setup';     d = 'Create/refresh the environment from uv.lock' }
            @{ n = 'lock';      d = 'Re-resolve dependencies and update uv.lock' }
            @{ n = 'lint';      d = 'Lint and check formatting' }
            @{ n = 'format';    d = 'Auto-fix lint findings and format' }
            @{ n = 'typecheck'; d = 'Strict type check' }
            @{ n = 'test';      d = 'Run the fast tests' }
            @{ n = 'test-cov';  d = 'Run tests with a coverage report' }
            @{ n = 'test-all';  d = 'Include the slow tests' }
            @{ n = 'check';     d = 'Everything CI runs' }
            @{ n = 'fly';       d = 'Single-drone flight check, kinematic' }
            @{ n = 'view';      d = 'Open the Canopy desktop viewer' }
            @{ n = 'props';     d = 'Rebuild the model library props from scripts/build_props.py' }
            @{ n = 'clean';     d = 'Remove caches and build artefacts' }
        ) | ForEach-Object { Write-Host ('  {0,-12} {1}' -f $_.n, $_.d) }
    }
    'setup'     { Invoke-Step (@('uv', 'sync') + $Extra) }
    'lock'      { Invoke-Step (@('uv', 'lock') + $Extra) }
    'lint' {
        Invoke-Step @('uv', 'run', 'ruff', 'check', '.')
        Invoke-Step @('uv', 'run', 'ruff', 'format', '--check', '.')
    }
    'format' {
        Invoke-Step @('uv', 'run', 'ruff', 'check', '--fix', '.')
        Invoke-Step @('uv', 'run', 'ruff', 'format', '.')
    }
    'typecheck' { Invoke-Step (@('uv', 'run', 'mypy') + $Extra) }
    'test'      { Invoke-Step (@('uv', 'run', 'pytest') + $Extra) }
    'test-cov'  {
        Invoke-Step (@('uv', 'run', 'pytest', '--cov',
                       '--cov-report=term-missing', '--cov-report=xml') + $Extra)
    }
    'test-all'  { Invoke-Step (@('uv', 'run', 'pytest', '-m', 'slow or not slow') + $Extra) }
    'check' {
        Invoke-Step @('uv', 'run', 'ruff', 'check', '.')
        Invoke-Step @('uv', 'run', 'ruff', 'format', '--check', '.')
        Invoke-Step @('uv', 'run', 'mypy')
        Invoke-Step @('uv', 'run', 'pytest')
        Write-Host 'All checks passed.' -ForegroundColor Green
    }
    'fly'       { Invoke-Step (@('uv', 'run', 'canopy-fly') + $Extra) }
    'view' {
        # A terminal keeps the environment it was opened with, so a key saved
        # to the user environment since then would be missed and address
        # search would silently fall back to Photon.
        if (-not $env:GEOAPIFY_API_KEY) {
            $env:GEOAPIFY_API_KEY = [Environment]::GetEnvironmentVariable('GEOAPIFY_API_KEY', 'User')
        }
        $provider = if ($env:GEOAPIFY_API_KEY) { 'Geoapify' } else { 'Photon (no GEOAPIFY_API_KEY)' }
        Write-Host "Address search: $provider" -ForegroundColor DarkGray
        Invoke-Step (@('uv', 'run', 'canopy-view') + $Extra)
    }
    'props'     { Invoke-Step (@('uv', 'run', 'python', 'scripts/build_props.py', '--catalog') + $Extra) }
    'clean' {
        foreach ($p in '.pytest_cache', '.ruff_cache', '.mypy_cache', 'htmlcov',
                       'coverage.xml', '.coverage') {
            if (Test-Path $p) { Remove-Item -Recurse -Force $p }
        }
        Get-ChildItem -Recurse -Directory -Filter __pycache__ |
            Where-Object { $_.FullName -notlike '*\.venv\*' } |
            Remove-Item -Recurse -Force
        Write-Host 'Cleaned.' -ForegroundColor Green
    }
}
