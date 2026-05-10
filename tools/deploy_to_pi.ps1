<#
.SYNOPSIS
    Push the NeoMscope repo to a Raspberry Pi 5 and run setup.sh remotely.

.DESCRIPTION
    Reads tools/pi-config.env (or the example file as fallback) for connection
    details. The Pi side either:
      * already has a git clone -> we git pull on the Pi
      * has nothing -> we git clone <GIT_REMOTE> on the Pi

    Then we run bash setup.sh with whatever SETUP_FLAGS the config file sets.

.PARAMETER Host
    Override PI_HOST from the config file.

.PARAMETER User
    Override PI_USER.

.PARAMETER Flags
    Override SETUP_FLAGS, e.g. -Flags "--no-gui --no-self-test"

.PARAMETER Probe
    Just verify connectivity (ping + ssh + lspci) and exit. No deploy.

.EXAMPLE
    .\tools\deploy_to_pi.ps1
    # uses tools/pi-config.env

.EXAMPLE
    .\tools\deploy_to_pi.ps1 -Host 192.168.1.50 -User domam -Flags "--no-gui"

.EXAMPLE
    .\tools\deploy_to_pi.ps1 -Probe
    # ping + ssh + lspci + exit
#>

[CmdletBinding()]
param(
    [string]$PiHost,
    [string]$User,
    [int]$Port,
    [string]$ProjectDir,
    [string]$Flags,
    [switch]$Probe,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repoRoot  = Split-Path -Parent $PSScriptRoot
$configEnv = Join-Path $repoRoot 'tools\pi-config.env'
$configExample = Join-Path $repoRoot 'tools\pi-config.example.env'

# ---- Load config ----
$cfg = @{
    PI_HOST         = '192.168.123.110'
    PI_USER         = 'pi'
    PI_PORT         = 22
    PI_PROJECT_DIR  = 'NeoMscope'
    GIT_REMOTE      = 'https://github.com/domafordarwin/NeoMscope.git'
    GIT_BRANCH      = 'main'
    SETUP_FLAGS     = ''
}

$cfgPath = if (Test-Path $configEnv) { $configEnv } else { $configExample }
if (Test-Path $cfgPath) {
    Write-Host "[deploy] Loading config: $cfgPath" -ForegroundColor DarkGray
    Get-Content $cfgPath | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith('#')) {
            $kv = $line -split '=', 2
            if ($kv.Count -eq 2) {
                $cfg[$kv[0].Trim()] = $kv[1].Trim()
            }
        }
    }
}

# CLI parameter overrides
if ($PiHost)     { $cfg.PI_HOST         = $PiHost }
if ($User)       { $cfg.PI_USER         = $User }
if ($Port)       { $cfg.PI_PORT         = $Port }
if ($ProjectDir) { $cfg.PI_PROJECT_DIR  = $ProjectDir }
if ($PSBoundParameters.ContainsKey('Flags')) { $cfg.SETUP_FLAGS = $Flags }

$target = "$($cfg.PI_USER)@$($cfg.PI_HOST)"
Write-Host ""
Write-Host "[deploy] target:    $target (port $($cfg.PI_PORT))" -ForegroundColor Cyan
Write-Host "[deploy] dir:       ~/$($cfg.PI_PROJECT_DIR)"       -ForegroundColor Cyan
Write-Host "[deploy] git:       $($cfg.GIT_REMOTE) [$($cfg.GIT_BRANCH)]" -ForegroundColor Cyan
Write-Host "[deploy] flags:     $($cfg.SETUP_FLAGS)"            -ForegroundColor Cyan
Write-Host ""

# ---- Connectivity probe ----
function Test-PiReachable {
    Write-Host "[deploy] ping $($cfg.PI_HOST)..." -ForegroundColor DarkGray
    if (-not (Test-Connection -ComputerName $cfg.PI_HOST -Count 1 -Quiet)) {
        throw "Pi at $($cfg.PI_HOST) does not respond to ping. Check network / Pi power."
    }
    Write-Host "[deploy]   ping OK" -ForegroundColor Green

    Write-Host "[deploy] ssh probe..." -ForegroundColor DarkGray
    $probeCmd = "uname -a; cat /proc/device-tree/model 2>/dev/null | tr -d '\0' && echo; lspci 2>/dev/null | grep -i hailo || echo '  (no Hailo on lspci yet)'"
    & ssh -p $cfg.PI_PORT -o ConnectTimeout=8 -o BatchMode=no $target $probeCmd
    if ($LASTEXITCODE -ne 0) {
        throw "ssh failed. First time? Run:  ssh $target  manually to accept host key."
    }
    Write-Host "[deploy]   ssh OK" -ForegroundColor Green
}

Test-PiReachable

if ($Probe) {
    Write-Host "[deploy] probe complete (no deploy)." -ForegroundColor Green
    exit 0
}

# ---- Deploy ----
$piDir = $cfg.PI_PROJECT_DIR
$gitRemote = $cfg.GIT_REMOTE
$gitBranch = $cfg.GIT_BRANCH
$setupFlags = $cfg.SETUP_FLAGS

$bashScript = @"
set -e
cd "`$HOME"
if [ -d "$piDir/.git" ]; then
    echo "[pi] git pull..."
    cd "$piDir"
    git fetch origin
    git checkout "$gitBranch"
    git pull --ff-only origin "$gitBranch"
else
    echo "[pi] git clone..."
    git clone --branch "$gitBranch" "$gitRemote" "$piDir"
    cd "$piDir"
fi
echo "[pi] running setup.sh $setupFlags ..."
bash setup.sh $setupFlags
"@

if ($DryRun) {
    Write-Host "[deploy] (dry run) would execute on Pi:" -ForegroundColor Yellow
    Write-Host $bashScript -ForegroundColor DarkGray
    exit 0
}

Write-Host "[deploy] running on Pi (output streams below)..." -ForegroundColor Cyan
Write-Host "─" * 60 -ForegroundColor DarkGray

& ssh -p $cfg.PI_PORT -tt $target "bash -s" -- <<< $bashScript

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "─" * 60 -ForegroundColor DarkGray
    Write-Host "[deploy] ✅ Done." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next, on the Pi (open another terminal or ssh into it):" -ForegroundColor Cyan
    Write-Host "  ssh $target"
    Write-Host "  cd ~/$piDir && source .venv/bin/activate"
    Write-Host "  neomscope-live    --camera /dev/video0      # CLI 실시간"
    Write-Host "  neomscope-batch   --input <dir> --output <dir>   # CLI 일괄"
    Write-Host "  neomscope-capture --camera 0                # CLI 캡처+검출"
    Write-Host "  neomscope-gui                               # PySide6 GUI" -ForegroundColor Yellow
} else {
    throw "Remote setup.sh failed with exit code $LASTEXITCODE."
}
