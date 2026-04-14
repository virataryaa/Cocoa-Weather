$bat = "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\Non Fundamental\Weather\Cocoa\daily_push.bat"
$action   = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$bat`""
$trigger  = New-ScheduledTaskTrigger -Daily -At 07:15AM
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable
Register-ScheduledTask -TaskName "Cocoa Weather Daily Update" -Action $action -Trigger $trigger -Settings $settings -Force
Write-Host "Task Scheduler job created successfully."
