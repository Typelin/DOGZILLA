# @name         &Archive and Download...
# @command      powershell.exe -ExecutionPolicy Bypass -File "%EXTENSION_PATH%" ^
#                   -sessionUrl "!E" -remotePath "!/" -localPath "!\" ^
#                   -archiveName "%ArchiveName%" -pause %Extract% ^
#                   -sessionLogPath "%SessionLogPath%" ^
#                   %Use7zip% -path7zip "%Path7zip%" ^
#                   -archiveType %ArchiveType% -archiveCommand "%ArchiveCommand%" !&
# @description  Packs the selected files to an archive, downloads it, ^
#                   and optionally extracts the archive to the current local directory
# @flag         ApplyToDirectories
# @flag         RemoteFiles
# @version      1
# @homepage     https://winscp.net/eng/docs/extension_archive_and_download
# @require      WinSCP 5.15
# @option       ArchiveName -run textbox "&Archive name:" "archive"
# @option       ArchiveType -config -run combobox "Archive &type:" ^
#                   zip zip tar/gzip
# @option       - -config group "Packing"
# @option         ArchiveCommand -config textbox "Custom archive &command:"
# @option       - -config -run group "Extracting"
# @option         Extract -config -run checkbox "&Extract after download" "" -extract
# @option         Use7zip -config -run checkbox "Use &7-zip for extracting" "" -use7zip
# @option         Path7zip -config file "7-zip &path (7z.exe/7za.exe):" ^
#                     "C:\Program Files\7-Zip\7z.exe"
# @option       - -config group "Logging"
# @option         SessionLogPath -config sessionlogfile
# @optionspage  https://winscp.net/eng/docs/extension_archive_and_download#options
 
param (
    # Use Generate Session URL function to obtain a value for -sessionUrl parameter.
    $sessionUrl = "sftp://user:mypassword;fingerprint=ssh-rsa-xxxxxxxxxxx...@example.com/",
    [Parameter(Mandatory = $True)]
    $remotePath,
    [Parameter(Mandatory = $True)]
    $localPath,
    [Switch]
    $pause,
    $archiveName,
    [Switch]
    $use7Zip,
    # The 7z.exe can be replaced with portable 7za.exe
    $path7zip = "C:\Program Files\7-Zip\7z.exe",
    $archiveType = "zip",
    [Switch]
    $extract,
    $archiveCommand,
    $sessionLogPath = $Null,
    [Parameter(Mandatory = $True, ValueFromRemainingArguments = $True, Position = 0)]
    $remotePaths
)
 
try
{
    switch ($archiveType)
    {
        "tar/gzip" { $ext = "tar.gz" }
        default { $ext = $archiveType }
    }
    $archiveName += "." + $ext

    if (-not $archiveCommand)
    {
        switch ($archiveType)
        {
            "zip" { $archiveCommand = "zip -r" }
            "tar/gzip" { $archiveCommand = "tar -czvf" }

            default
            {
                throw ("Custom archive type was selected, " +
                       "but custom archive command was not specified.")
            }
        }
    }

    # Load WinSCP .NET assembly
    $assemblyPath = if ($env:WINSCP_PATH) { $env:WINSCP_PATH } else { $PSScriptRoot }
    Add-Type -Path (Join-Path $assemblyPath "WinSCPnet.dll")
 
    # Setup session options
    $sessionOptions = New-Object WinSCP.SessionOptions
    $sessionOptions.ParseUrl($sessionUrl)

    if (($sessionOptions.Protocol -ne [WinSCP.Protocol]::Sftp) -and
        ($sessionOptions.Protocol -ne [WinSCP.Protocol]::Scp))
    {
        throw "Shell access is required, which is provided by SFTP and SCP protocols only."
    }
 
    if ($extract)
    {
        if (-not $use7Zip)
        {
            if ($PSVersionTable.PSVersion.Major -lt 5)
            {
                throw ("PowerShell 5.0 or newer required to extract an archive without 7-zip." +
                       "Please, upgrade PowerShell or use the 7-zip mode instead.")
            }
            if ($archiveType -ne "zip")
            {
                throw ("Only ZIP format is supported to extract an archive without 7-zip. " +
                       "Please, select ZIP archive type or use the 7-zip mode instead.")
            }
        }

        $downloadPath = $env:TEMP
    }
    else
    {
        $downloadPath = $localPath
    }

    $session = New-Object WinSCP.Session
 
    try
    {
        $session.SessionLogPath = $sessionLogPath
 
        Write-Host "Connecting..."
        $Host.UI.RawUI.WindowTitle = "Connecting"
        $session.Open($sessionOptions)
 
        Write-Host "Archiving $($remotePaths.Count) files to archive $archiveName..."
        $Host.UI.RawUI.WindowTitle = "Archiving"

        $filesArgs = ""
        foreach ($s in $remotePaths)
        {
            $s = ($s -replace '[\`$"]', "`\`\")
            $filesArgs += "`"$s`" "
        }

        $archiveNameArg = "`"$archiveName`""

        $commands = "cd `"$remotePath`" ; $archiveCommand $archiveNameArg $filesArgs"

        $result = $session.ExecuteCommand($commands)
        $result.Check()

        Write-Host $result.ErrorOutput
        Write-Host $result.Output
 
        Write-Host "Archive $archiveName created."

        Write-Host "Downloading..."
        $Host.UI.RawUI.WindowTitle = "Downloading"

        $source = [WinSCP.RemotePath]::EscapeFileMask($archiveName)
        $source = [WinSCP.RemotePath]::Combine($remotePath, $source)
        $target = (Join-Path $downloadPath "*")
        $session.GetFiles($source, $target, $True).Check()
    }
    finally
    {
        # Disconnect, clean up
        $session.Dispose()
    }

    if ($extract)
    {
        try
        {
            Write-Host "Extracting..."
            $Host.UI.RawUI.WindowTitle = "Extracting"

            $archiveTempPath = (Join-Path $downloadPath $archiveName)

            if ($use7Zip)
            {
                if ($archiveType -eq "tar/gzip")
                {
                    # https://stackoverflow.com/q/1359793/850848#14699663
                    & cmd.exe (
                        "/C `"`"$path7zip`" x -y `"$archiveTempPath`" -so | " +
                        "`"$path7zip`" x -y -ttar -si `"-o$localPath`"`"")
                }
                else
                {
                    & "$path7zip" x -y "$archiveTempPath" "-o$localPath"
                }
         
                if ($LASTEXITCODE -gt 0)
                {
                    throw "Extracting failed."
                }
            }
            else
            {
                Expand-Archive $archiveTempPath $localPath
            }
        }
        finally
        {
            Remove-Item $archiveTempPath
        }
    }

    Write-Host "Done."
    $Host.UI.RawUI.WindowTitle = "Done"

    $result = 0
}
catch
{
    Write-Host "Error: $($_.Exception.Message)"
    $Host.UI.RawUI.WindowTitle = "Error"
    $result = 1
}
 
# Pause if -pause switch was used
if ($pause)
{
    Write-Host
    Write-Host "Press any key to exit..."
    [System.Console]::ReadKey() | Out-Null
}
 
exit $result
