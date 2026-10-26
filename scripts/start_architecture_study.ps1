param(
    [string]$Base = 'D:\Research\ODE-FFN',
    [Parameter(Mandatory=$true)][string]$RuntimeRoot,
    [Parameter(Mandatory=$true)][string]$Plan,
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_-]+$')][string]$OutputName
)
$ErrorActionPreference='Stop'
$taskName="ODEFFN_$OutputName"
if(Test-Path "$Base\runs\$OutputName"){throw 'Output directory exists'}
if(Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue){throw 'Task exists'}
$arguments='-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' +
    "$RuntimeRoot\scripts\run_architecture_study_windows.ps1" + '" -Base "'+$Base+
    '" -RuntimeRoot "'+$RuntimeRoot+'" -Plan "'+$Plan+'" -OutputName "'+$OutputName+'"'
$action=New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments -WorkingDirectory $RuntimeRoot
$principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$settings=New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 4) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Description 'Frozen hypothesis-specific ODE-FFN study' | Out-Null
Start-ScheduledTask -TaskName $taskName
@{task=$taskName;runtime=$RuntimeRoot;plan=$Plan;output="$Base\runs\$OutputName";time=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -Encoding UTF8 "$Base\logs\$OutputName-launch.json"
Get-Content "$Base\logs\$OutputName-launch.json"
