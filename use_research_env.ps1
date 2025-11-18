# use_research_env.ps1
# Force Python & Pip to always come from research_env

$env:Path = "D:\Capstone_research\research_env\Scripts;" + $env:Path

Set-Alias python "D:\Capstone_research\research_env\Scripts\python.exe"
Set-Alias pip "D:\Capstone_research\research_env\Scripts\pip.exe"

Write-Host "✅ Research environment is now active. Using:" -ForegroundColor Green
& D:\Capstone_research\research_env\Scripts\python.exe --version
& D:\Capstone_research\research_env\Scripts\pip.exe --version
