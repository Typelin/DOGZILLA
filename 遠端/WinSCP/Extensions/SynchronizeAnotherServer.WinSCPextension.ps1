# @name         &Synchronize with Another Remote Server...
# @command      powershell.exe -ExecutionPolicy Bypass -File "%EXTENSION_PATH%" ^
#                   -sessionUrl1 "!E" -remotePath1 "!/" ^
#                   -sessionUrl2 "%SessionUrl2%" %PasswordPrompt% -remotePath2 "%RemotePath2%" ^
#                   %Delete% %Preview% %ContinueOnError% -pause -sessionLogPath ^
#                   "%SessionLogPath%"
# @description  Synchronizes a directory on another server (or another directory on this ^
#               server) against a directory on this server
# @version      2
# @homepage     https://winscp.net/eng/docs/extension_synchronize_another_server
# @require      WinSCP 5.18
# @require      PowerShell 3.0
# @option       - -run group "Synchronize directory from &this server:"
# @option       RemotePath1 -run textbox "&Directory:" "!/"
# @option       - -run group "... to &another server:"
# @option       SessionUrl2 -run textbox "&Session:" "!S"
# @option       PasswordPrompt -run checkbox "&Prompt for session password" ^
#                 -passwordPrompt  -passwordPrompt 
# @option       RemotePath2 -run textbox "Di&rectory:" "!/"
# @option       - -config -run group "Options"
# @option         Delete -config -run checkbox "&Delete files" "" -delete 
# @option         Preview -config -run checkbox "&Preview changes" -preview -preview
# @option         ContinueOnError -config -run checkbox "Continue on &error" "" -continueOnError
# @option       - -config group "Logging"
# @option         SessionLogPath -config sessionlogfile
# @optionspage  https://winscp.net/eng/docs/extension_synchronize_another_server#options

param (
    # Use Generate Session URL function to obtain a value
    # for -sessionUrl1 and -sessionUrl2 parameters.
    $sessionUrl1 = "sftp://user:mypassword;fingerprint=ssh-rsa-xxxxxxxxxx...@one.example.com/",
    [Parameter(Mandatory = $True)]
    $remotePath1,
    $sessionUrl2 = "sftp://user:mypassword;fingerprint=ssh-rsa-xxxxxxxxxx...@two.example.com/",
    [Switch]
    $passwordPrompt,
    [Parameter(Mandatory = $True)]
    $remotePath2,
    [Switch]
    $delete,
    [Switch]
    $preview,
    [Switch]
    $continueOnError,
    [Switch]
    $pause,
    $sessionLogPath = $Null
)

Set-StrictMode -Version 3.0

function SetConsoleTitle ($status)
{
    $Host.UI.RawUI.WindowTitle = $status
}

function HandleException ($e)
{
    if ($continueOnError)
    {
        Write-Host -ForegroundColor Red $_.Exception.Message
    }
    else
    {
        throw $e
    }
}

function CompareDirectories ($remotePath1, $remotePath2)
{
    Write-Host -NoNewline "Comparing $remotePath1 with $remotePath2..."
    try
    {
        $enumerationOptions = [WinSCP.EnumerationOptions]::MatchDirectories
        $files1 = $session1.EnumerateRemoteFiles($remotePath1, $Null, $enumerationOptions)
        $files2 = $session2.EnumerateRemoteFiles($remotePath2, $Null, $enumerationOptions)
        $first = $True

        $directories = [System.Collections.ArrayList]@()
        $existing2 = [System.Collections.ArrayList]@()

        foreach ($file1 in $files1)
        {
            $file2 = $files2 | Where-Object Name -EQ $file1.Name
            $modified = $False
            if ($file2 -eq $Null)
            {
                $modified = $True
            }
            else
            {
                $existing2.Add($file2.Name) | Out-Null
                if ($file1.IsDirectory -and $file2.IsDirectory)
                {
                    $file1 | Add-Member -NotePropertyName _Other -NotePropertyValue $file2
                    $directories.Add($file1) | Out-Null
                }
                else
                {
                    if ($file1.LastWriteTime -gt $file2.LastWriteTime)
                    {
                        $modified = $True
                    }
                }
            }

            if ($modified)
            {
                if ($first)
                {
                    Write-Host
                    $first = $False
                }
                if ($file2 -eq $Null)
                {
                    Write-Host "$($file1.FullName) is new"
                }
                else
                {
                    Write-Host "$($file1.FullName) is modified comparing to $($file2.FullName)"
                }
                $file1 | Add-Member -NotePropertyName _Target -NotePropertyValue $remotePath2
                $changes.Add($file1) | Out-Null
            }
        }

        if ($delete)
        {
            foreach ($file2 in $files2)
            {
                if (-not $existing2.Contains($file2.Name))
                {
                    if ($first)
                    {
                        Write-Host
                        $first = $False
                    }
                    Write-Host "$($file2.FullName) is orphan"
                    $file2 | Add-Member -NotePropertyName _Target -NotePropertyValue $Null
                    $changes.Add($file2) | Out-Null
                }
            }
        }

        if ($first)
        {
            Write-Host
        }

        foreach ($directory1 in $directories)
        { 
            $directory2 = $directory1._Other
            if ($directory1.IsDirectory -and $directory2.IsDirectory)
            {
                CompareDirectories $directory1.FullName $directory2.FullName
            }
        }
    }
    catch
    {
        Write-Host
        HandleException $_
    }
}

try
{
    # Load WinSCP .NET assembly
    $assemblyPath = if ($env:WINSCP_PATH) { $env:WINSCP_PATH } else { $PSScriptRoot }
    Add-Type -Path (Join-Path $assemblyPath "WinSCPnet.dll")

    # Setup session options
    $sessionOptions1 = New-Object WinSCP.SessionOptions
    $sessionOptions1.ParseUrl($sessionUrl1)
    $sessionOptions2 = New-Object WinSCP.SessionOptions
    $sessionOptions2.ParseUrl($sessionUrl2)
    if ((($sessionOptions2.Protocol -eq [WinSCP.Protocol]::Sftp) -or
         ($sessionOptions2.Protocol -eq [WinSCP.Protocol]::Scp)) -and
        (-not $sessionOptions2.SshHostKeyFingerprint))
    {
        $sessionOptions2.SshHostKeyPolicy = [WinSCP.SshHostKeyPolicy]::AcceptNew
    }

    $session1 = New-Object WinSCP.Session
    $session2 = New-Object WinSCP.Session

    try
    {
        SetConsoleTitle "Connecting..."

        if ($sessionLogPath)
        {
            $session1.SessionLogPath = $sessionLogPath + ".1"
            $session2.SessionLogPath = $sessionLogPath + ".2"
        }

        Write-Host "Connecting to $sessionOptions1..."
        $session1.Open($sessionOptions1)

        Write-Host "Connecting to $sessionOptions2..."
        if ($passwordPrompt -and
            (-not $sessionOptions2.SecurePassword) -and
            (-not $sessionOptions2.SshPrivateKeyPath))
        {
            $sessionOptions2.SecurePassword = Read-Host "Password" -AsSecureString
        }
        $session2.Open($sessionOptions2)

        Write-Host "Comparing..."
        SetConsoleTitle "Comparing..."

        $changes = [System.Collections.ArrayList]@()
        CompareDirectories $remotePath1 $remotePath2

        if ($changes.Count -eq 0)
        {
            Write-Host "No changes found"
        }
        else
        {
            $continue = $True
            if ($preview)
            {
                Write-Host -NoNewline "Continue? y/N "
                $key = [System.Console]::ReadKey()
                Write-Host
                $continue = ($key.KeyChar -eq "y")
                if (!$continue)
                {
                    $pause = $False
                }
            }

            if ($continue)
            {
                Write-Host "Synchronizing..."
                SetConsoleTitle "Synchronizing..."
                $tempName = [System.IO.Path]::GetRandomFileName()
                $tempPath = Join-Path ([System.IO.Path]::GetTempPath()) $tempName
                New-Item -ItemType Directory -Path $tempPath | Out-Null

                try
                {
                    foreach ($change in $changes)
                    {
                        $fullName = $change.FullName
                        if ($change._Target)
                        {
                            $remotePath1 = [WinSCP.RemotePath]::GetDirectoryName($fullName)
                            $remotePath2 = $change._Target

                            Write-Host -NoNewline "Synchronizing $fullName to $remotePath2..."
                            $filemask = [WinSCP.RemotePath]::EscapeFileMask($change.Name)
                            try
                            {
                                $session1.GetFilesToDirectory(
                                    $remotePath1, $tempPath, $filemask).Check()
                                $session2.PutFilesToDirectory(
                                    $tempPath, $remotePath2, $filemask).Check()
                                Write-Host " Done."
                            }
                            catch
                            {
                                Write-Host
                                HandleException $_
                            }

                            Remove-Item -Recurse -Force (Join-Path $tempPath $change.Name)
                        }
                        else
                        {
                            Write-Host -NoNewline "Removing orphan $fullName..."
                            try
                            {
                                $path = [WinSCP.RemotePath]::EscapeFileMask($fullName)
                                $session2.RemoveFiles($path).Check()
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
                finally
                {
                    Remove-Item -Recurse -Force $tempPath
                }
            }
        }
    }
    finally
    {
        # Disconnect, clean up
        $session1.Dispose()
        $session2.Dispose()
    }

    SetConsoleTitle "Done"
    $result = 0
}
catch
{
    $continueOnError = $True
    HandleException $_
    SetConsoleTitle "Error"
    $result = 1
}

# Pause if -pause switch was used
if ($pause)
{
    Write-Host "Press any key to exit..."
    [System.Console]::ReadKey() | Out-Null
}

exit $result
