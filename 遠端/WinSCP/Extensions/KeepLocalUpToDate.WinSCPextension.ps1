# @name         &Keep Local Directory up to Date...
# @command      powershell.exe -ExecutionPolicy Bypass -File "%EXTENSION_PATH%" ^
#                   -sessionUrl "!E" -localPath "%LocalPath%" -remotePath "%RemotePath%" ^
#                   %Delete% %Beep% %ContinueOnError% -interval "%Interval%" ^
#                   -fileMask "%FileMask%" -pause -sessionLogPath "%SessionLogPath%"
# @description  Periodically scans for changes in a remote directory and ^
#                   reflects them on a local directory
# @version      11
# @homepage     https://winscp.net/eng/docs/library_example_keep_local_directory_up_to_date
# @require      WinSCP 5.16
# @option       - -run group "Directories"
# @option         RemotePath -run textbox "&Watch for changes in the remote directory:" "!/"
# @option         LocalPath -run textbox ^
#                     "... &and automatically reflect them on the local directory:" "!\"
# @option       - -config -run group "Options"
# @option         Delete -config -run checkbox "&Delete files" "" -delete 
# @option         Beep -config -run checkbox "&Beep on change" "" -beep
# @option         ContinueOnError -config -run checkbox "Continue on &error" "" -continueOnError
# @option         Interval -config -run textbox "&Interval (in seconds):" "30"
# @option         FileMask -config -run textbox "File &mask:" ""
# @option       - -config group "Logging"
# @option         SessionLogPath -config sessionlogfile
# @optionspage  ^
#     https://winscp.net/eng/docs/library_example_keep_local_directory_up_to_date#options

param (
    # Use Generate Session URL function to obtain a value for -sessionUrl parameter.
    $sessionUrl = "sftp://user:mypassword;fingerprint=ssh-rsa-xxxxxxxxxxx...@example.com/",
    [Parameter(Mandatory = $True)]
    $localPath,
    [Parameter(Mandatory = $True)]
    $remotePath,
    [Switch]
    $delete,
    [Switch]
    $beep,
    [Switch]
    $continueOnError,
    $sessionLogPath = $Null,
    $interval = 30,
    $fileMask = $Null,
    [Switch]
    $pause
)

function Beep()
{
    if ($beep)
    {
        [System.Console]::Beep()
    }
}

function HandleException ($e)
{
    if ($continueOnError)
    {
        Write-Host -ForegroundColor Red $_.Exception.Message
        Beep
    }
    else
    {
        throw $e
    }
}

function SetConsoleTitle ($status)
{
    if ($sessionOptions)
    {
        $status = "$sessionOptions - $status"
    }
    $Host.UI.RawUI.WindowTitle = $status
}

try
{
    # Load WinSCP .NET assembly
    $assemblyPath = if ($env:WINSCP_PATH) { $env:WINSCP_PATH } else { $PSScriptRoot }
    Add-Type -Path (Join-Path $assemblyPath "WinSCPnet.dll")

    # Setup session options
    $sessionOptions = New-Object WinSCP.SessionOptions
    $sessionOptions.ParseUrl($sessionUrl)

    $transferOptions = New-Object WinSCP.TransferOptions -Property @{ FileMask = $fileMask };

    $session = New-Object WinSCP.Session
    
    try
    {
        $session.SessionLogPath = $sessionLogPath

        Write-Host "Connecting..."
        SetConsoleTitle "Connecting"
        $session.Open($sessionOptions)

        while ($True)
        {
            Write-Host -NoNewline "Looking for changes..."
            SetConsoleTitle "Looking for changes"
            try
            {
                $differences =
                    $session.CompareDirectories(
                        [WinSCP.SynchronizationMode]::Local, $localPath, $remotePath, $delete,
                        $False, [WinSCP.SynchronizationCriteria]::Time, $transferOptions)

                Write-Host
                if ($differences.Count -eq 0)
                {
                    Write-Host "No changes found."   
                }
                else
                {
                    Write-Host "Synchronizing $($differences.Count) change(s)..."
                    SetConsoleTitle "Synchronizing changes"
                    Beep

                    foreach ($difference in $differences)
                    {
                        $action = $difference.Action
                        if ($action -eq [WinSCP.SynchronizationAction]::DownloadNew)
                        {
                            $message = "Downloading new $($difference.Remote.FileName)..."
                        }
                        elseif ($action -eq [WinSCP.SynchronizationAction]::DownloadUpdate)
                        {
                            $message = "Downloading updated $($difference.Remote.FileName)..."
                        }
                        elseif ($action -eq [WinSCP.SynchronizationAction]::DeleteLocal)
                        {
                            $message = "Deleting $($difference.Local.FileName)..."
                        }
                        else
                        {
                            throw "Unexpected difference $action"
                        }

                        Write-Host -NoNewline $message

                        try
                        {
                            $difference.Resolve($session, $transferOptions) | Out-Null
                            Write-Host " Done."
                        }
                        catch
                        {
                            Write-Host
                            HandleException $_
                        }
                    }
                }
            }
            catch
            {
                Write-Host
                HandleException $_
            }

            SetConsoleTitle "Waiting"
            $wait = [int]$interval
            # Wait for 1 second in a loop, to make the waiting breakable
            while ($wait -gt 0)
            {
                Write-Host -NoNewLine "`rWaiting for $wait seconds, press Ctrl+C to abort... "
                Start-Sleep -Seconds 1
                $wait--
            }

            Write-Host
            Write-Host
        }
    }
    finally
    {
        Write-Host # to break after "Waiting..." status
        Write-Host "Disconnecting..."
        # Disconnect, clean up
        $session.Dispose()
    }
}
catch
{
    $continueOnError = $True
    HandleException $_
    SetConsoleTitle "Error"
}

# Pause if -pause switch was used
if ($pause)
{
    Write-Host "Press any key to exit..."
    [System.Console]::ReadKey() | Out-Null
}

# Never exits cleanly
exit 1
