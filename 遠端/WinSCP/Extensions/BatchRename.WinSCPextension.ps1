# @name         Batch &Rename...
# @command      powershell.exe -ExecutionPolicy Bypass -File "%EXTENSION_PATH%" ^
#                   -sessionUrl "!E" -remotePath "!/" -pattern "%Pattern%" ^
#                   -replacement "%Replacement%" -refresh -pause -sessionLogPath "%SessionLogPath%" ^
#                   %PreviewMode% !& 
# @description  Renames remote files using a regular expression
# @flag         RemoteFiles
# @version      7
# @homepage     https://winscp.net/eng/docs/library_example_advanced_rename
# @require      WinSCP 5.19
# @option       - -run group "Rename"
# @option         Pattern -run textbox "Replace file name part matching this pattern:"
# @option         Replacement -run textbox "with:"
# @option       - -run -config group "Options"
# @option         PreviewMode -run -config checkbox "&Preview changes" "-previewMode" ^
#                     "-previewMode"
# @option       - -config group "Logging"
# @option         SessionLogPath -config sessionlogfile
# @optionspage  https://winscp.net/eng/docs/library_example_advanced_rename#options

param (
    # Use Generate Session URL function to obtain a value for -sessionUrl parameter.
    $sessionUrl = "sftp://user:mypassword;fingerprint=ssh-rsa-xxxxxxxxxxx...@example.com/",
    [Parameter(Mandatory = $True)]
    $remotePath,
    [Parameter(Mandatory = $True)]
    $pattern,
    $replacement,
    [Switch]
    $pause,
    [Switch]
    $refresh,
    $sessionLogPath = $Null,
    [Switch]
    $previewMode,
    [Parameter(Mandatory = $True, ValueFromRemainingArguments = $True, Position = 0)]
    $files
)

try
{
    if ($previewMode)
    {
        $anyChange = $False
        foreach ($file in $files)
        {
            $newName = $file -replace $pattern, $replacement
            if ($newName -eq $file)
            {
                Write-Host "$file not changed"
            }
            else
            {
                Write-Host "$file => $newName"
                $anyChange = $True
            }
        }

        Write-Host

        if (!$anyChange)
        {
            Write-Host "No change to be made"
            $continue = $False
        }
        else
        {
            Write-Host -NoNewline "Continue? y/N "
            $key = [System.Console]::ReadKey()
            Write-Host
            Write-Host
            $continue = ($key.KeyChar -eq "y")
            if (!$continue)
            {
                $pause = $False
            }
        }
    }
    else
    {
        $continue = $True
    }

    if (!$continue)
    {
        $result = 1
    }
    else
    {
        # Load WinSCP .NET assembly
        $assemblyPath = if ($env:WINSCP_PATH) { $env:WINSCP_PATH } else { $PSScriptRoot }
        Add-Type -Path (Join-Path $assemblyPath "WinSCPnet.dll")

        # Setup session options
        $sessionOptions = New-Object WinSCP.SessionOptions
        $sessionOptions.ParseUrl($sessionUrl)

        $session = New-Object WinSCP.Session

        try
        {
            $session.SessionLogPath = $sessionLogPath
     
            Write-Host "Connecting..."
            $session.Open($sessionOptions)

            Write-Host "Renaming..."
            foreach ($file in $files)
            {
                $newName = $file -replace $pattern, $replacement
                if ($newName -eq $file)
                {
                    Write-Host "$file not changed"
                }
                else
                {
                    Write-Host "$file => $newName"
                    $fileMask = [WinSCP.RemotePath]::EscapeFileMask($file)
                    $sourcePath = [WinSCP.RemotePath]::Combine($remotePath, $fileMask)
                    $operationMask = [WinSCP.RemotePath]::EscapeOperationMask($newName)
                    $targetPath = [WinSCP.RemotePath]::Combine($remotePath, $operationMask)
                    $session.MoveFile($sourcePath, $targetPath)
                }
            }
        }
        finally
        {
            # Disconnect, clean up
            $session.Dispose()
        }

        if ($refresh)
        {
            & "$env:WINSCP_PATH\WinSCP.exe" "$sessionUrl" /refresh "$remotePath"
        }
    }

    $result = 0
}
catch
{
    Write-Host "Error: $($_.Exception.Message)"
    $result = 1
}

if ($pause)
{
    Write-Host "Press any key to exit..."
    [System.Console]::ReadKey() | Out-Null
}

exit $result
