param(
    [string]$Base = 'D:\Research\ODE-FFN',
    [Parameter(Mandatory=$true)][string]$RuntimeRoot,
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$OutputName = 'recurrent_depth_20261026_product16',
    [switch]$Worker
)
$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$python = Join-Path $Base 'envs\synthetic\Scripts\python.exe'
$output = Join-Path $Base "runs\$OutputName"
if ($Worker) {
    $process = Start-Process -FilePath $python -ArgumentList @(
        '-u', "$RuntimeRoot\scripts\run_recurrent_depth.py", 'run',
        '--output', $output, '--datasets', 'rotated_product_bimodal_16d',
        '--data-v1', "$Base\datasets\synthetic\v1",
        '--data-product', "$Base\datasets\synthetic\product16_v1"
    ) -WorkingDirectory $RuntimeRoot -NoNewWindow -Wait -PassThru `
      -RedirectStandardOutput "$Base\logs\$OutputName.out.log" `
      -RedirectStandardError "$Base\logs\$OutputName.err.log"
    exit $process.ExitCode
}
if (Test-Path $output) { throw "Output already exists: $output" }
if (-not (Test-Path "$RuntimeRoot\preflight\PASS.json")) { throw 'Native preflight receipt missing' }
$taskName = "ODEFFN_$OutputName"
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Task already exists' }
$arguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' +
    "$RuntimeRoot\scripts\start_recurrent_depth.ps1" + '" -Base "' + $Base +
    '" -RuntimeRoot "' + $RuntimeRoot + '" -OutputName "' + $OutputName + '" -Worker'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments -WorkingDirectory $RuntimeRoot
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings `
    -Description 'Frozen seed-0 recurrent ODE/SwiGLU depth ablation with process monitoring' | Out-Null
Start-ScheduledTask -TaskName $taskName
@{task=$taskName; output=$output; time=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json
