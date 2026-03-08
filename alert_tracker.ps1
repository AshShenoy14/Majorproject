param (
    [int]$ProcessId
)

while ($true) {
    if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
        Write-Host "Training process has finished!"
        [System.Media.SystemSounds]::Beep.Play()
        Start-Sleep -Seconds 1
        [System.Media.SystemSounds]::Beep.Play()
        Start-Sleep -Seconds 1
        [System.Media.SystemSounds]::Beep.Play()
        
        # Optional: Show a popup message
        Add-Type -AssemblyName PresentationFramework
        [System.Windows.MessageBox]::Show("The ESM-MLP model training has finished! You can now run the ensemble script and compare models.", "Training Complete")
        break
    }
    Start-Sleep -Seconds 30
}
