# TEMPORARY: register cdb.exe as non-interactive post-mortem debugger
$cdb = "${env:ProgramFiles(x86)}\Windows Kits\10\Debuggers\x64\cdb.exe"
if (!(Test-Path $cdb)) {
  Write-Output "cdb.exe not found at $cdb, searching ..."
  Get-ChildItem -Path "${env:ProgramFiles(x86)}\Windows Kits", "${env:ProgramFiles}\WindowsApps" -Recurse -Filter cdb.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
  Exit 1
}
$key = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\AeDebug"
Write-Output "previous AeDebug:"
Get-ItemProperty $key | Format-List
$log = "$env:RUNNER_TEMP\aedebug_cdb.log"
$cmd = "`"$cdb`" -p %ld -e %ld -loga `"$log`" -lines -c `".jdinfo 0x%p; .lines -e; kpn 30; qd`""
Set-ItemProperty $key -Name Debugger -Value $cmd
Set-ItemProperty $key -Name Auto -Value "1"
Write-Output "new AeDebug:"
Get-ItemProperty $key | Format-List
