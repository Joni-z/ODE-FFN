param([string]$Base,[string]$RuntimeRoot,[string]$Plan,[string]$OutputName)
$ErrorActionPreference='Stop'
$env:PATH="$Base\tools\git\cmd;"+$env:PATH
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONUNBUFFERED='1'
$env:PYTHONUTF8='1'
$process=Start-Process -FilePath "$Base\envs\synthetic\Scripts\python.exe" `
    -ArgumentList @('-u',"$RuntimeRoot\scripts\run_architecture_study.py",'--plan',$Plan,
        '--data-root',"$Base\datasets\synthetic\v1",'--output',"$Base\runs\$OutputName") `
    -WorkingDirectory $RuntimeRoot -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput "$Base\logs\$OutputName.out.log" `
    -RedirectStandardError "$Base\logs\$OutputName.err.log"
exit $process.ExitCode
