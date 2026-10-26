param(
    [string]$Base = 'D:\Research\ODE-FFN',
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$OutputName = 'clock_20261024'
)
$ErrorActionPreference = 'Stop'
$output = Join-Path "$Base\runs" $OutputName
if (Test-Path $output) { throw "Output already exists: $output" }
$taskName = "ODEFFN_$OutputName"
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw "Scheduled task already exists: $taskName"
}
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument (
    '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' +
    "$Base\repo\scripts\run_clock_study_windows.ps1" +
    '" -Base "' + $Base + '" -OutputName "' + $OutputName + '"'
) -WorkingDirectory "$Base\repo"
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal `
    -Settings $settings -Description 'Seed-0 ODELayer centered clock study; logs on D drive' | Out-Null
Start-ScheduledTask -TaskName $taskName
@{ task = $taskName; output = $output; started_utc = [DateTime]::UtcNow.ToString('o') } |
    ConvertTo-Json | Set-Content -Encoding UTF8 "$Base\logs\$OutputName-task-launch.json"
Get-Content "$Base\logs\$OutputName-task-launch.json"
