[CmdletBinding()]
param(
    [string]$InstallRoot = "",
    [string]$RepoUrl = "https://github.com/AkaneTendo25/musubi-tuner.git",
    [string]$Branch = "ltx-2",
    [string]$RepoDir = "",
    [ValidateSet("cu124", "cu128", "cu130", "cpu")]
    [string]$Cuda = "cu128",
    [ValidateSet("3.10", "3.11", "3.12", "3.13")]
    [string]$PythonVersion = "3.12",
    [int]$Port = 7860,
    [string]$DashboardHost = "127.0.0.1",
    [ValidateSet("Safe", "StashAndUpdate")]
    [string]$UpdatePolicy = "Safe",
    [switch]$NonInteractive,
    [switch]$StrictPreflight,
    [switch]$PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:LogFile = Join-Path $env:TEMP ("musubi_ltx2_install_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
New-Item -Path $script:LogFile -ItemType File -Force | Out-Null
$script:SessionId = [Guid]::NewGuid().ToString()
$script:CurrentStep = "bootstrap"
$script:LastRepositorySyncSucceeded = $false
$script:LastRepositoryBackup = $null
$script:InstallStateFileName = ".musubi_install_state.json"
$script:DashboardLauncherName = "launch_musubi_dashboard.cmd"
$script:SetupLauncherName = "launch_musubi_setup.cmd"
$script:DashboardShortcutName = "Musubi Tuner Dashboard.lnk"
$script:SetupShortcutName = "Musubi Tuner Setup and Update.lnk"

function Write-Log {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [ValidateSet("INFO", "WARN", "ERROR", "OK")][string]$Level = "INFO"
    )

    $line = "{0} [{1}] {2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Level, $Message
    Add-Content -Path $script:LogFile -Value $line -Encoding ASCII

    $color = switch ($Level) {
        "WARN" { "Yellow" }
        "ERROR" { "Red" }
        "OK" { "Green" }
        default { "Gray" }
    }
    Write-Host $line -ForegroundColor $color
}

function Write-Section {
    param([Parameter(Mandatory = $true)][string]$Title)
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host (" {0}" -f $Title) -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    $previous = $script:CurrentStep
    $failed = $false
    $script:CurrentStep = $Name
    Write-Log ("STEP_START [{0}]" -f $Name)
    try {
        & $Action
        Write-Log ("STEP_OK [{0}]" -f $Name) "OK"
    } catch {
        $failed = $true
        $script:CurrentStep = $Name
        throw [System.Exception]::new(("[{0}] {1}" -f $Name, $_.Exception.Message), $_.Exception)
    } finally {
        if (-not $failed) {
            $script:CurrentStep = $previous
        }
    }
}

function Write-SupportBundle {
    param(
        [string]$Outcome = "unknown",
        [System.Management.Automation.ErrorRecord]$ErrorRecord = $null
    )

    Write-Log "----- SUPPORT_BUNDLE_BEGIN -----"
    Write-Log ("SessionId: {0}" -f $script:SessionId)
    Write-Log ("Outcome: {0}" -f $Outcome)
    Write-Log ("CurrentStep: {0}" -f $script:CurrentStep)
    Write-Log ("Timestamp: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"))
    Write-Log ("PowerShell: {0}" -f $PSVersionTable.PSVersion)
    Write-Log ("OS: {0}" -f [Environment]::OSVersion.VersionString)
    Write-Log ("User: {0}" -f $env:USERNAME)
    Write-Log ("Machine: {0}" -f $env:COMPUTERNAME)

    if ($ErrorRecord) {
        Write-Log ("ErrorMessage: {0}" -f $ErrorRecord.Exception.Message) "ERROR"
        Write-Log ("ExceptionType: {0}" -f $ErrorRecord.Exception.GetType().FullName) "ERROR"
        Write-Log ("Category: {0}" -f $ErrorRecord.CategoryInfo) "ERROR"
        if ($ErrorRecord.InvocationInfo) {
            Write-Log ("ScriptName: {0}" -f $ErrorRecord.InvocationInfo.ScriptName) "ERROR"
            Write-Log ("ScriptLineNumber: {0}" -f $ErrorRecord.InvocationInfo.ScriptLineNumber) "ERROR"
            Write-Log ("PositionMessage: {0}" -f $ErrorRecord.InvocationInfo.PositionMessage) "ERROR"
        }
    }
    Write-Log ("LogFile: {0}" -f $script:LogFile)
    Write-Log "----- SUPPORT_BUNDLE_END -----"
}

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $rendered = $Arguments | ForEach-Object {
        if ($_ -match "\s") { '"{0}"' -f $_ } else { $_ }
    }
    Write-Log ("Running: {0} {1}" -f $FilePath, ($rendered -join " "))

    & $FilePath @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $FilePath $($Arguments -join ' ')"
    }
}

function Write-ExternalOutputLines {
    param(
        [object[]]$Output = @(),
        [ValidateSet("INFO", "WARN", "ERROR", "OK")][string]$Level = "WARN"
    )

    foreach ($line in @($Output)) {
        $text = ([string]$line).TrimEnd()
        if (-not [string]::IsNullOrWhiteSpace($text)) {
            Write-Log ("  {0}" -f $text) $Level
        }
    }
}

function Refresh-Path {
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $segments = @()
    if ($machinePath) { $segments += $machinePath }
    if ($userPath) { $segments += $userPath }
    if ($env:Path) { $segments += $env:Path }
    $env:Path = ($segments -join ";")
}

function Test-Command {
    param([Parameter(Mandatory = $true)][string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-WindowsHost {
    try {
        return [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
            [System.Runtime.InteropServices.OSPlatform]::Windows
        )
    } catch {
        return $env:OS -eq "Windows_NT"
    }
}

function Test-AnyCommand {
    param([Parameter(Mandatory = $true)][string[]]$Names)
    foreach ($name in $Names) {
        if (Test-Command $name) {
            return $true
        }
    }
    return $false
}

function Test-IsAdmin {
    try {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = [Security.Principal.WindowsPrincipal]::new($identity)
        return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    } catch {
        return $false
    }
}

function Test-TcpEndpoint {
    param(
        [Parameter(Mandatory = $true)][string]$HostName,
        [int]$PortNumber = 443,
        [int]$TimeoutMs = 2500
    )

    try {
        [void][System.Net.Dns]::GetHostAddresses($HostName)
    } catch {
        return $false
    }

    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        try {
            $async = $client.BeginConnect($HostName, $PortNumber, $null, $null)
            if (-not $async.AsyncWaitHandle.WaitOne($TimeoutMs, $false)) {
                return $false
            }
            $client.EndConnect($async) | Out-Null
            return $true
        } finally {
            $client.Close()
        }
    } catch {
        return $false
    }
}

function Add-CheckResult {
    param(
        [Parameter(Mandatory = $true)]$Bucket,
        [Parameter(Mandatory = $true)][string]$Message
    )
    if (-not ($Bucket -is [System.Collections.IList])) {
        throw "Internal error: Add-CheckResult requires an IList bucket."
    }
    [void]$Bucket.Add($Message)
}

function Ensure-Tls12ForLegacyPowerShell {
    if ($PSVersionTable.PSVersion.Major -gt 5) {
        return
    }
    try {
        $hasTls12 = ([Net.ServicePointManager]::SecurityProtocol -band [Net.SecurityProtocolType]::Tls12) -ne 0
        if (-not $hasTls12) {
            [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
            Write-Log "Enabled TLS 1.2 for this PowerShell session." "OK"
        }
    } catch {
        Write-Log "Could not configure TLS 1.2 automatically; HTTPS downloads may fail." "WARN"
    }
}

function Get-InstallStatePath {
    param([Parameter(Mandatory = $true)][string]$RepoPath)
    return Join-Path $RepoPath $script:InstallStateFileName
}

function Get-InstallState {
    param([Parameter(Mandatory = $true)][string]$RepoPath)

    if ([string]::IsNullOrWhiteSpace($RepoPath)) {
        return $null
    }

    $statePath = Get-InstallStatePath -RepoPath $RepoPath
    if (-not (Test-Path $statePath)) {
        return $null
    }

    try {
        $raw = Get-Content -Path $statePath -Raw -Encoding UTF8
        if ([string]::IsNullOrWhiteSpace($raw)) {
            return $null
        }
        return $raw | ConvertFrom-Json
    } catch {
        Write-Log ("Could not parse install state at {0}: {1}" -f $statePath, $_.Exception.Message) "WARN"
        return $null
    }
}

function Get-GitStatusEntryPath {
    param([Parameter(Mandatory = $true)][string]$Line)

    if ($Line.Length -lt 4) {
        return ""
    }
    return $Line.Substring(3)
}

function Format-GitStatusEntry {
    param([Parameter(Mandatory = $true)][string]$Line)

    $statusCode = if ($Line.Length -ge 2) { $Line.Substring(0, 2) } else { $Line }
    $path = Get-GitStatusEntryPath -Line $Line
    $label = switch -Exact ($statusCode) {
        "??" { "untracked" }
        " M" { "modified" }
        "M " { "staged" }
        "MM" { "modified" }
        " A" { "added" }
        "A " { "staged add" }
        " D" { "deleted" }
        "D " { "staged delete" }
        default {
            $trimmed = $statusCode.Trim()
            if ([string]::IsNullOrWhiteSpace($trimmed)) { "changed" } else { $trimmed }
        }
    }

    return "{0}: {1}" -f $label, $path
}

function Format-StatusFileList {
    param(
        [object[]]$Items = @(),
        [int]$MaxItems = 5
    )

    $values = @(@($Items) | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
    if ($values.Count -eq 0) {
        return ""
    }

    $shown = @($values | Select-Object -First $MaxItems)
    $suffix = if ($values.Count -gt $MaxItems) { " (+{0} more)" -f ($values.Count - $MaxItems) } else { "" }
    return "{0}{1}" -f ($shown -join ", "), $suffix
}

function Get-StashRecoveryCommands {
    param([string]$StashTarget = "")

    if ([string]::IsNullOrWhiteSpace($StashTarget)) {
        return @()
    }

    return @(
        "git stash show --include-untracked --name-only $StashTarget",
        "git stash show --include-untracked -p $StashTarget",
        "git stash apply $StashTarget",
        "git stash pop $StashTarget"
    )
}

function Get-GitStatusPorcelain {
    param(
        [Parameter(Mandatory = $true)][string]$GitExe,
        [Parameter(Mandatory = $true)][string]$RepoPath
    )

    $output = & $GitExe -C $RepoPath status --porcelain -- . 2>$null
    $exitCode = $LASTEXITCODE
    $trackedOutput = [System.Collections.ArrayList]::new()
    $untrackedOutput = [System.Collections.ArrayList]::new()

    if ($exitCode -eq 0) {
        foreach ($line in @($output)) {
            if ([string]::IsNullOrWhiteSpace($line)) {
                continue
            }

            $statusCode = if ($line.Length -ge 2) { $line.Substring(0, 2) } else { $line }
            if ($statusCode -eq "??") {
                [void]$untrackedOutput.Add($line)
            } else {
                [void]$trackedOutput.Add($line)
            }
        }
    }

    $trackedLines = @($trackedOutput | ForEach-Object { [string]$_ })
    $untrackedLines = @($untrackedOutput | ForEach-Object { [string]$_ })
    $allLines = @($output | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })

    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = $allLines
        TrackedOutput = $trackedLines
        UntrackedOutput = $untrackedLines
        ChangedFiles = @($trackedLines | ForEach-Object { Format-GitStatusEntry -Line $_ })
        UntrackedFiles = @($untrackedLines | ForEach-Object { Format-GitStatusEntry -Line $_ })
        LocalFiles = @($allLines | ForEach-Object { Format-GitStatusEntry -Line $_ })
    }
}

function Save-InstallState {
    param(
        [Parameter(Mandatory = $true)][string]$RepoPath,
        [Parameter(Mandatory = $true)][hashtable]$State
    )

    $statePath = Get-InstallStatePath -RepoPath $RepoPath
    $State["state_updated_utc"] = (Get-Date).ToUniversalTime().ToString("o")
    $State["repo_dir"] = $RepoPath

    try {
        $json = $State | ConvertTo-Json -Depth 8
        Set-Content -Path $statePath -Value $json -Encoding UTF8
    } catch {
        Write-Log ("Failed to save install state at {0}: {1}" -f $statePath, $_.Exception.Message) "WARN"
    }
}

function Get-OptionalPropertyValue {
    param(
        $Object,
        [Parameter(Mandatory = $true)][string]$Name,
        $Default = $null
    )

    if ($null -eq $Object) {
        return $Default
    }

    if ($Object -is [System.Collections.IDictionary]) {
        if ($Object.Contains($Name)) {
            $value = $Object[$Name]
            if ($null -ne $value) {
                return $value
            }
        }
        return $Default
    }

    $property = $Object.PSObject.Properties[$Name]
    if ($property) {
        $value = $property.Value
        if ($null -ne $value) {
            return $value
        }
    }

    return $Default
}

function Get-NestedPropertyValue {
    param(
        $Object,
        [Parameter(Mandatory = $true)][string[]]$Path,
        $Default = $null
    )

    $current = $Object
    foreach ($segment in $Path) {
        $current = Get-OptionalPropertyValue -Object $current -Name $segment -Default $null
        if ($null -eq $current) {
            return $Default
        }
    }

    return $current
}

function Format-StateTimestamp {
    param([string]$Timestamp)

    if ([string]::IsNullOrWhiteSpace($Timestamp)) {
        return "never"
    }

    try {
        return ([DateTimeOffset]::Parse($Timestamp)).ToLocalTime().ToString("yyyy-MM-dd HH:mm")
    } catch {
        return $Timestamp
    }
}

function Get-RepositoryStatus {
    param(
        [string]$GitExe = "",
        [Parameter(Mandatory = $true)][string]$RepoPath,
        [Parameter(Mandatory = $true)][string]$BranchName,
        [string]$RemoteUrl = ""
    )

    $status = [ordered]@{
        exists = Test-Path $RepoPath
        git_available = -not [string]::IsNullOrWhiteSpace($GitExe)
        is_git_repo = $false
        origin_configured = $false
        origin_error = ""
        head = ""
        head_short = ""
        branch = ""
        dirty = $false
        fetch_attempted = $false
        fetch_succeeded = $false
        offline = $false
        remote_head = ""
        remote_head_short = ""
        local_ahead_count = 0
        remote_ahead_count = 0
        diverged = $false
        update_available = $false
        can_auto_update = $false
        tracked_dirty = $false
        has_untracked = $false
        changed_files = @()
        untracked_files = @()
        local_files = @()
        summary = "Repository not found"
        error = ""
    }

    if (-not $status.exists) {
        return [pscustomobject]$status
    }
    if (-not $status.git_available) {
        $status.summary = "Git is not available"
        return [pscustomobject]$status
    }
    if (-not (Test-Path (Join-Path $RepoPath ".git"))) {
        $status.summary = "Repository exists but is not a git checkout"
        return [pscustomobject]$status
    }

    $status.is_git_repo = $true

    $head = (& $GitExe -C $RepoPath rev-parse HEAD 2>$null | Select-Object -First 1)
    if ($LASTEXITCODE -eq 0 -and $head) {
        $status.head = $head.Trim()
        if ($status.head.Length -ge 8) {
            $status.head_short = $status.head.Substring(0, 8)
        }
    }

    $branch = (& $GitExe -C $RepoPath rev-parse --abbrev-ref HEAD 2>$null | Select-Object -First 1)
    if ($LASTEXITCODE -eq 0 -and $branch) {
        $status.branch = $branch.Trim()
    }

    $porcelain = Get-GitStatusPorcelain -GitExe $GitExe -RepoPath $RepoPath
    if ($porcelain.ExitCode -eq 0) {
        $status.changed_files = @($porcelain.ChangedFiles)
        $status.untracked_files = @($porcelain.UntrackedFiles)
        $status.local_files = @($porcelain.LocalFiles)
        $status.tracked_dirty = @($porcelain.TrackedOutput).Count -gt 0
        $status.has_untracked = @($porcelain.UntrackedOutput).Count -gt 0
        $status.dirty = $status.tracked_dirty
    }

    $originUrl = (& $GitExe -C $RepoPath remote get-url origin 2>$null | Select-Object -First 1)
    if ($LASTEXITCODE -eq 0 -and $originUrl) {
        $status.origin_configured = $true
    }

    $canReachOrigin = $true
    $originHostHint = if (-not [string]::IsNullOrWhiteSpace($RemoteUrl)) { $RemoteUrl } elseif ($originUrl) { $originUrl.Trim() } else { "" }
    if ($originHostHint -match "github\.com") {
        $canReachOrigin = Test-TcpEndpoint -HostName "github.com" -PortNumber 443 -TimeoutMs 2000
    }

    if (-not $status.origin_configured) {
        $status.origin_error = "No origin remote is configured"
    } elseif ($canReachOrigin) {
        $status.fetch_attempted = $true
        try {
            $null = & $GitExe -C $RepoPath fetch --quiet origin $BranchName 2>$null
            if ($LASTEXITCODE -eq 0) {
                $status.fetch_succeeded = $true
            }
        } catch {
            $status.error = $_.Exception.Message
        }
        if (-not $status.fetch_succeeded -and [string]::IsNullOrWhiteSpace($status.origin_error)) {
            $status.origin_error = "Fetching origin failed. Check repository access, branch name, and network connectivity."
        }
    } else {
        $status.offline = $true
        $status.origin_error = "Could not reach the origin host to check for updates"
    }

    if ($status.fetch_succeeded) {
        $remoteRef = "origin/$BranchName"
        $remoteHead = (& $GitExe -C $RepoPath rev-parse $remoteRef 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -eq 0 -and $remoteHead) {
            $status.remote_head = $remoteHead.Trim()
            if ($status.remote_head.Length -ge 8) {
                $status.remote_head_short = $status.remote_head.Substring(0, 8)
            }
        }

        $counts = (& $GitExe -C $RepoPath rev-list --left-right --count "HEAD...$remoteRef" 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -eq 0 -and $counts) {
            $parts = ($counts -replace "\s+", " ").Trim().Split(" ")
            if ($parts.Count -ge 2) {
                $status.local_ahead_count = [int]$parts[0]
                $status.remote_ahead_count = [int]$parts[1]
                $status.diverged = ($status.local_ahead_count -gt 0 -and $status.remote_ahead_count -gt 0)
                $status.update_available = ($status.remote_ahead_count -gt 0)
            }
        }
    }

    if ($status.tracked_dirty) {
        $status.summary = "Tracked local changes detected ($(@($status.changed_files).Count) file(s)); safe auto-update is disabled"
    } elseif ($status.diverged) {
        $status.summary = "Local branch and origin have diverged"
    } elseif ($status.update_available) {
        $status.summary = "Update available: origin/$BranchName is $($status.remote_ahead_count) commit(s) ahead"
    } elseif ($status.local_ahead_count -gt 0) {
        $status.summary = "Local branch is $($status.local_ahead_count) commit(s) ahead of origin"
    } elseif (-not $status.origin_configured) {
        $status.summary = "No origin remote configured; update checks are unavailable"
    } elseif ($status.fetch_attempted -and (-not $status.fetch_succeeded)) {
        $status.summary = "Origin fetch failed; review git remote configuration and network access"
    } elseif ($status.has_untracked) {
        $status.summary = "Repository is up to date; untracked local files are present and will be left in place"
    } else {
        $status.summary = "Repository is up to date"
    }

    $status.can_auto_update = $status.update_available -and (-not $status.tracked_dirty) -and (-not $status.diverged)
    return [pscustomobject]$status
}

function Test-GitPathsChanged {
    param(
        [Parameter(Mandatory = $true)][string]$GitExe,
        [Parameter(Mandatory = $true)][string]$RepoPath,
        [string]$FromRef = "",
        [string]$ToRef = "",
        [string[]]$PathSpecs = @()
    )

    if ([string]::IsNullOrWhiteSpace($FromRef) -or [string]::IsNullOrWhiteSpace($ToRef) -or $PathSpecs.Count -eq 0) {
        return $false
    }

    $null = & $GitExe -C $RepoPath rev-parse --verify $FromRef 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    $null = & $GitExe -C $RepoPath rev-parse --verify $ToRef 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $false
    }

    $diffArgs = @("-C", $RepoPath, "diff", "--name-only", "$FromRef..$ToRef", "--") + $PathSpecs
    $changed = & $GitExe @diffArgs 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $false
    }

    return -not [string]::IsNullOrWhiteSpace(($changed | Out-String).Trim())
}

function Write-InstallerOverview {
    param([string[]]$Lines = @())

    if ($Lines.Count -eq 0) {
        return
    }

    Write-Host ""
    Write-Host "Current Setup State" -ForegroundColor DarkGray
    foreach ($line in $Lines) {
        Write-Host (" - {0}" -f $line) -ForegroundColor DarkGray
    }
}

function Get-HostSetForActions {
    param([Parameter(Mandatory = $true)][System.Collections.ArrayList]$Actions)
    $hosts = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

    if (Get-ActionState -Actions $Actions -Key "CloneOrUpdate") {
        [void]$hosts.Add("github.com")
    }
    if (Get-ActionState -Actions $Actions -Key "InstallDeps") {
        [void]$hosts.Add("pypi.org")
        [void]$hosts.Add("files.pythonhosted.org")
        [void]$hosts.Add("download.pytorch.org")
    }
    if (Get-ActionState -Actions $Actions -Key "BuildFrontend") {
        [void]$hosts.Add("registry.npmjs.org")
    }
    if (
        (Get-ActionState -Actions $Actions -Key "InstallGit") -or
        (Get-ActionState -Actions $Actions -Key "InstallPython") -or
        (Get-ActionState -Actions $Actions -Key "InstallNode")
    ) {
        [void]$hosts.Add("api.github.com")
        [void]$hosts.Add("github.com")
    }
    return $hosts
}

function Invoke-Preflight {
    param(
        [Parameter(Mandatory = $true)][System.Collections.ArrayList]$Actions,
        [Parameter(Mandatory = $true)][string]$InstallPath,
        [Parameter(Mandatory = $true)][string]$RepositoryPath,
        [Parameter(Mandatory = $true)][string]$PreferredPythonVersion,
        [Parameter(Mandatory = $true)][int]$DashboardPort,
        [switch]$StrictMode
    )

    $errors = [System.Collections.ArrayList]::new()
    $warnings = [System.Collections.ArrayList]::new()

    Write-Log "Running preflight checks..."

    if (-not (Test-WindowsHost)) {
        Add-CheckResult -Bucket $errors -Message "Windows host check failed. This installer supports Windows only."
    }

    if ($PSVersionTable.PSVersion.Major -lt 5) {
        Add-CheckResult -Bucket $errors -Message "PowerShell 5.1+ is required."
    }

    if ($DashboardPort -lt 1 -or $DashboardPort -gt 65535) {
        Add-CheckResult -Bucket $errors -Message "Port $DashboardPort is invalid. Use 1..65535."
    }

    $pm = Get-PackageManager
    $pmName = if ($pm) { $pm.Name } else { "none" }
    Write-Log "Detected package manager: $pmName"

    $gitInstalled = Test-Command "git"
    $npmInstalled = Test-AnyCommand @("npm.cmd", "npm")
    $pythonExact = Get-PythonExecutable -PreferredVersion $PreferredPythonVersion
    $pythonAny = Get-PythonExecutable -PreferredVersion $PreferredPythonVersion -AllowAny
    $pythonAnyOk = $false
    if ($pythonAny) {
        $pythonAnyOk = Test-PythonMinimumVersion -PythonExe $pythonAny -MinMajor 3 -MinMinor 10
    }

    if ((Get-ActionState -Actions $Actions -Key "InstallGit") -and (-not $gitInstalled) -and (-not $pm)) {
        Add-CheckResult -Bucket $errors -Message "Git is missing and no package manager (winget/choco/scoop) is available. Install Git manually: https://git-scm.com/download/win"
    }
    if ((Get-ActionState -Actions $Actions -Key "CloneOrUpdate") -and (-not $gitInstalled) -and (-not (Get-ActionState -Actions $Actions -Key "InstallGit"))) {
        Add-CheckResult -Bucket $errors -Message "Clone/update is enabled but git is missing and 'Install Git' is disabled."
    }

    $needsPython = (Get-ActionState -Actions $Actions -Key "CreateVenv") -or (Get-ActionState -Actions $Actions -Key "InstallDeps")
    if ((Get-ActionState -Actions $Actions -Key "InstallPython") -and (-not $pythonExact) -and (-not $pm)) {
        Add-CheckResult -Bucket $errors -Message "Python $PreferredPythonVersion is missing and no package manager is available. Install manually: https://www.python.org/downloads/windows/"
    }
    if ($needsPython -and (-not (Get-ActionState -Actions $Actions -Key "InstallPython")) -and (-not $pythonAnyOk)) {
        Add-CheckResult -Bucket $errors -Message "Python 3.10+ is required for selected actions, but no compatible Python is installed and 'Install Python' is disabled."
    }
    if ($pythonAny -and (-not $pythonExact) -and $pythonAnyOk) {
        Add-CheckResult -Bucket $warnings -Message "Preferred Python $PreferredPythonVersion is not present. A compatible fallback ($pythonAny) will be used."
    }

    if ((Get-ActionState -Actions $Actions -Key "BuildFrontend") -and (-not $npmInstalled) -and (-not (Get-ActionState -Actions $Actions -Key "InstallNode"))) {
        Add-CheckResult -Bucket $errors -Message "Frontend build is enabled but npm is missing and 'Install Node.js' is disabled."
    }
    if ((Get-ActionState -Actions $Actions -Key "InstallNode") -and (-not $npmInstalled) -and (-not $pm)) {
        Add-CheckResult -Bucket $errors -Message "Node.js/npm is missing and no package manager is available. Install manually: https://nodejs.org/"
    }

    if (Test-Path $RepositoryPath) {
        if ((Get-ActionState -Actions $Actions -Key "CloneOrUpdate") -and (-not (Test-Path (Join-Path $RepositoryPath ".git")))) {
            Add-CheckResult -Bucket $errors -Message "Repository path exists but is not a git repository: $RepositoryPath"
        }
    } elseif (-not (Get-ActionState -Actions $Actions -Key "CloneOrUpdate")) {
        Add-CheckResult -Bucket $errors -Message "Repository path does not exist and 'Clone/update repository' is disabled: $RepositoryPath"
    }

    $installRootPath = [Environment]::ExpandEnvironmentVariables($InstallPath)
    try {
        if (-not (Test-Path $installRootPath)) {
            New-Item -Path $installRootPath -ItemType Directory -Force | Out-Null
        }
    } catch {
        Add-CheckResult -Bucket $errors -Message "Cannot create/access install root: $installRootPath. Error: $($_.Exception.Message)"
    }

    try {
        $root = [System.IO.Path]::GetPathRoot($installRootPath)
        if ($root) {
            $driveName = $root.TrimEnd('\').TrimEnd(':')
            $drive = Get-PSDrive -Name $driveName -ErrorAction SilentlyContinue
            if ($drive) {
                $freeGiB = [math]::Round($drive.Free / 1GB, 2)
                if ($freeGiB -lt 8) {
                    $msg = "Low disk space on $root (${freeGiB} GiB free). At least ~8 GiB is recommended."
                    if ($StrictMode) {
                        Add-CheckResult -Bucket $errors -Message $msg
                    } else {
                        Add-CheckResult -Bucket $warnings -Message $msg
                    }
                } elseif ($freeGiB -lt 16) {
                    Add-CheckResult -Bucket $warnings -Message "Disk space on $root is ${freeGiB} GiB. Installation may fail depending on caches/build artifacts."
                }
            }
        }
    } catch {
        Add-CheckResult -Bucket $warnings -Message "Unable to determine free disk space."
    }

    if ($pm -and $pm.Name -eq "choco" -and (-not (Test-IsAdmin))) {
        Add-CheckResult -Bucket $warnings -Message "Chocolatey detected without elevated shell. Some package installs may fail."
    }

    $hosts = Get-HostSetForActions -Actions $Actions
    foreach ($endpoint in $hosts) {
        if (-not (Test-TcpEndpoint -HostName $endpoint -PortNumber 443 -TimeoutMs 2500)) {
            $msg = "Cannot reach $endpoint:443 from this machine."
            if ($StrictMode) {
                Add-CheckResult -Bucket $errors -Message $msg
            } else {
                Add-CheckResult -Bucket $warnings -Message $msg
            }
        }
    }

    if ($warnings.Count -gt 0) {
        Write-Log "Preflight warnings:" "WARN"
        foreach ($w in $warnings) {
            Write-Log " - $w" "WARN"
        }
    }

    if ($errors.Count -gt 0) {
        Write-Log "Preflight blockers:" "ERROR"
        foreach ($e in $errors) {
            Write-Log " - $e" "ERROR"
        }
        throw "Preflight failed with $($errors.Count) blocker(s). Fix listed issues and rerun."
    }

    Write-Log "Preflight passed." "OK"
}

function Invoke-ExternalWithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [int]$MaxAttempts = 3,
        [int]$DelaySeconds = 4
    )

    if ($MaxAttempts -lt 1) {
        $MaxAttempts = 1
    }

    $attempt = 1
    while ($attempt -le $MaxAttempts) {
        try {
            Invoke-External -FilePath $FilePath -Arguments $Arguments
            return
        } catch {
            if ($attempt -ge $MaxAttempts) {
                throw
            }
            Write-Log "Attempt $attempt failed. Retrying in $DelaySeconds second(s)..." "WARN"
            Start-Sleep -Seconds $DelaySeconds
            $attempt++
        }
    }
}

function Get-PackageManager {
    if (Test-Command "winget") {
        return [pscustomobject]@{ Name = "winget"; Command = (Get-Command winget).Source }
    }
    if (Test-Command "choco") {
        return [pscustomobject]@{ Name = "choco"; Command = (Get-Command choco).Source }
    }
    if (Test-Command "scoop") {
        return [pscustomobject]@{ Name = "scoop"; Command = (Get-Command scoop).Source }
    }
    return $null
}

function Get-PythonExecutable {
    param(
        [Parameter(Mandatory = $true)][string]$PreferredVersion,
        [switch]$AllowAny
    )

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        try {
            $pyArgs = @()
            if ($AllowAny) {
                $pyArgs = @("-c", "import sys; print(sys.executable)")
            } else {
                $pyArgs = @("-$PreferredVersion", "-c", "import sys; print(sys.executable)")
            }
            $pyOut = & $pyCmd.Source @pyArgs 2>$null
            if ($LASTEXITCODE -eq 0 -and $pyOut) {
                $candidate = ($pyOut | Select-Object -Last 1).Trim()
                if (Test-Path $candidate) {
                    return $candidate
                }
            }
        } catch {
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        try {
            $out = & $pythonCmd.Source -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}'); print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                $lines = @($out)
                if ($lines.Count -ge 2) {
                    $ver = $lines[0].Trim()
                    $exe = $lines[1].Trim()
                    if (($AllowAny -or $ver -eq $PreferredVersion) -and (Test-Path $exe)) {
                        return $exe
                    }
                }
            }
        } catch {
        }
    }

    $pyTag = $PreferredVersion.Replace(".", "")
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python$pyTag\python.exe"),
        (Join-Path $env:ProgramFiles "Python$pyTag\python.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Python$pyTag\python.exe")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    return $null
}

function Test-PythonMinimumVersion {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][int]$MinMajor,
        [Parameter(Mandatory = $true)][int]$MinMinor
    )

    $result = & $PythonExe -c "import sys; print(1 if sys.version_info >= ($MinMajor, $MinMinor) else 0)"
    return (($LASTEXITCODE -eq 0) -and (($result | Select-Object -Last 1).Trim() -eq "1"))
}

function Get-ExecutionPolicySnapshot {
    $rows = [System.Collections.ArrayList]::new()
    foreach ($scope in @("MachinePolicy", "UserPolicy", "Process", "CurrentUser", "LocalMachine")) {
        $policy = ""
        try {
            $policy = Get-ExecutionPolicy -Scope $scope -ErrorAction SilentlyContinue
        } catch {
            $policy = "Unknown"
        }
        [void]$rows.Add([pscustomobject]@{
            Scope = $scope
            Policy = if ([string]::IsNullOrWhiteSpace($policy)) { "Undefined" } else { $policy }
        })
    }
    return $rows
}

function Get-InitialEnvironmentScan {
    param([Parameter(Mandatory = $true)][string]$PreferredPythonVersion)

    $pm = Get-PackageManager
    $pythonExact = Get-PythonExecutable -PreferredVersion $PreferredPythonVersion
    $pythonAny = Get-PythonExecutable -PreferredVersion $PreferredPythonVersion -AllowAny
    $pythonAnySupported = $false
    if ($pythonAny) {
        $pythonAnySupported = Test-PythonMinimumVersion -PythonExe $pythonAny -MinMajor 3 -MinMinor 10
    }

    $effectivePolicy = ""
    try {
        $effectivePolicy = Get-ExecutionPolicy
    } catch {
        $effectivePolicy = "Unknown"
    }

    $policyWarnings = [System.Collections.ArrayList]::new()
    if ($effectivePolicy -in @("Restricted", "AllSigned")) {
        [void]$policyWarnings.Add("ExecutionPolicy is '$effectivePolicy'. One-liner or script execution may be blocked in some shells.")
    }
    if ($PSVersionTable.PSVersion.Major -le 5) {
        try {
            $hasTls12 = ([Net.ServicePointManager]::SecurityProtocol -band [Net.SecurityProtocolType]::Tls12) -ne 0
            if (-not $hasTls12) {
                [void]$policyWarnings.Add("TLS 1.2 is not enabled in current PowerShell session. HTTPS downloads can fail.")
            }
        } catch {
            [void]$policyWarnings.Add("Could not verify TLS 1.2 status in this PowerShell session.")
        }
    }

    return [pscustomobject]@{
        IsWindows = Test-WindowsHost
        PsVersion = $PSVersionTable.PSVersion
        IsAdmin = Test-IsAdmin
        PackageManager = if ($pm) { $pm.Name } else { "none" }
        GitInstalled = Test-Command "git"
        NodeInstalled = Test-AnyCommand @("npm.cmd", "npm")
        PythonExactInstalled = [bool]$pythonExact
        PythonExactPath = $pythonExact
        PythonAnyPath = $pythonAny
        PythonAnySupported = $pythonAnySupported
        ExecutionPolicyEffective = $effectivePolicy
        ExecutionPolicySnapshot = Get-ExecutionPolicySnapshot
        PolicyWarnings = $policyWarnings
    }
}

function Show-InitialEnvironmentScan {
    param([Parameter(Mandatory = $true)]$Scan)

    Write-Section "Initial System Scan"
    Write-Log ("SessionId: {0}" -f $script:SessionId)
    Write-Log ("Windows: {0}" -f $Scan.IsWindows)
    Write-Log ("PowerShell: {0}" -f $Scan.PsVersion)
    Write-Log ("Admin: {0}" -f $Scan.IsAdmin)
    Write-Log ("Package manager: {0}" -f $Scan.PackageManager)
    Write-Log ("Git installed: {0}" -f $Scan.GitInstalled)
    Write-Log ("Python exact installed ({0}): {1}" -f $PythonVersion, $Scan.PythonExactInstalled)
    if ($Scan.PythonExactPath) {
        Write-Log ("Python exact path: {0}" -f $Scan.PythonExactPath)
    }
    if ($Scan.PythonAnyPath -and (-not $Scan.PythonExactInstalled)) {
        Write-Log ("Python fallback path: {0}" -f $Scan.PythonAnyPath)
    }
    Write-Log ("Node/npm installed: {0}" -f $Scan.NodeInstalled)
    Write-Log ("ExecutionPolicy effective: {0}" -f $Scan.ExecutionPolicyEffective)
    foreach ($ep in $Scan.ExecutionPolicySnapshot) {
        Write-Log ("ExecutionPolicy[{0}]: {1}" -f $ep.Scope, $ep.Policy)
    }
    if ($Scan.PolicyWarnings.Count -gt 0) {
        foreach ($warn in $Scan.PolicyWarnings) {
            Write-Log ("Policy warning: {0}" -f $warn) "WARN"
        }
    }
}

function Ensure-PackageInstall {
    param(
        [Parameter(Mandatory = $true)][string]$DisplayName,
        [string]$WingetId = "",
        [string]$ChocoId = "",
        [string]$ScoopId = "",
        [string]$ManualUrl = ""
    )

    $pm = Get-PackageManager
    if (-not $pm) {
        $hint = if ($ManualUrl) { " Install manually: $ManualUrl" } else { "" }
        throw "No supported package manager found (winget/choco/scoop).$hint"
    }

    switch ($pm.Name) {
        "winget" {
            if ([string]::IsNullOrWhiteSpace($WingetId)) {
                throw "winget is available but no winget package ID was provided for $DisplayName."
            }
            Write-Log "$DisplayName is missing. Installing via winget..."
            Invoke-ExternalWithRetry -FilePath $pm.Command -Arguments @(
                "install",
                "--id", $WingetId,
                "--exact",
                "--accept-package-agreements",
                "--accept-source-agreements",
                "--silent",
                "--scope", "user"
            ) -MaxAttempts 2 -DelaySeconds 4
        }
        "choco" {
            if ([string]::IsNullOrWhiteSpace($ChocoId)) {
                throw "choco is available but no chocolatey package ID was provided for $DisplayName."
            }
            Write-Log "$DisplayName is missing. Installing via chocolatey..."
            Invoke-ExternalWithRetry -FilePath $pm.Command -Arguments @("install", $ChocoId, "-y") -MaxAttempts 2 -DelaySeconds 4
        }
        "scoop" {
            if ([string]::IsNullOrWhiteSpace($ScoopId)) {
                throw "scoop is available but no scoop package ID was provided for $DisplayName."
            }
            Write-Log "$DisplayName is missing. Installing via scoop..."
            Invoke-ExternalWithRetry -FilePath $pm.Command -Arguments @("install", $ScoopId) -MaxAttempts 2 -DelaySeconds 4
        }
        default {
            throw "Unsupported package manager backend: $($pm.Name)"
        }
    }

    Refresh-Path
}

function Ensure-Git {
    if (Test-Command "git") {
        Write-Log "Git already available."
        return
    }
    Ensure-PackageInstall `
        -DisplayName "Git" `
        -WingetId "Git.Git" `
        -ChocoId "git" `
        -ScoopId "git" `
        -ManualUrl "https://git-scm.com/download/win"
    if (-not (Test-Command "git")) {
        throw "Git installation did not make git available on PATH."
    }
    Write-Log "Git installed successfully." "OK"
}

function Ensure-Node {
    if (Test-AnyCommand @("npm.cmd", "npm")) {
        Write-Log "Node.js/npm already available."
        return
    }
    Ensure-PackageInstall `
        -DisplayName "Node.js LTS" `
        -WingetId "OpenJS.NodeJS.LTS" `
        -ChocoId "nodejs-lts" `
        -ScoopId "nodejs-lts" `
        -ManualUrl "https://nodejs.org/"
    if (-not (Test-AnyCommand @("npm.cmd", "npm"))) {
        throw "Node.js installation did not make npm available on PATH."
    }
    Write-Log "Node.js installed successfully." "OK"
}

function Ensure-Python {
    param([Parameter(Mandatory = $true)][string]$Version)

    $py = Get-PythonExecutable -PreferredVersion $Version
    if ($py) {
        Write-Log "Python $Version already available at $py"
        return $py
    }

    Ensure-PackageInstall `
        -DisplayName "Python $Version" `
        -WingetId "Python.Python.$Version" `
        -ChocoId "python" `
        -ScoopId "python" `
        -ManualUrl "https://www.python.org/downloads/windows/"

    Write-Log "Python package-manager install is per-user/system tooling, not repo-local. The project itself still runs inside its own venv." "WARN"

    $py = Get-PythonExecutable -PreferredVersion $Version
    if (-not $py) {
        $py = Get-PythonExecutable -PreferredVersion $Version -AllowAny
        if ($py -and (Test-PythonMinimumVersion -PythonExe $py -MinMajor 3 -MinMinor 10)) {
            Write-Log "Exact Python $Version was not found after installation; using compatible version at $py" "WARN"
            return $py
        }
    }
    if (-not $py) {
        throw "Python $Version install completed but executable was not found."
    }
    Write-Log "Python $Version installed at $py" "OK"
    return $py
}

function Resolve-PythonForBuild {
    param([Parameter(Mandatory = $true)][string]$PreferredVersion)

    $py = Get-PythonExecutable -PreferredVersion $PreferredVersion
    if ($py) {
        return $py
    }

    $fallback = Get-PythonExecutable -PreferredVersion $PreferredVersion -AllowAny
    if (-not $fallback) {
        throw "No Python interpreter found. Enable 'Install Python' or install Python manually."
    }

    if (-not (Test-PythonMinimumVersion -PythonExe $fallback -MinMajor 3 -MinMinor 10)) {
        throw "Found Python at $fallback, but version is below 3.10."
    }

    Write-Log "Preferred Python $PreferredVersion not found. Using compatible fallback: $fallback" "WARN"
    return $fallback
}

function Sync-Repository {
    param(
        [Parameter(Mandatory = $true)][string]$GitExe,
        [Parameter(Mandatory = $true)][string]$RepoPath,
        [Parameter(Mandatory = $true)][string]$RemoteUrl,
        [Parameter(Mandatory = $true)][string]$BranchName,
        [ValidateSet("Safe", "StashAndUpdate")][string]$UpdatePolicy = "Safe"
    )

    $script:LastRepositorySyncSucceeded = $false
    $script:LastRepositoryBackup = $null
    $parent = Split-Path -Parent $RepoPath
    if (-not (Test-Path $parent)) {
        New-Item -Path $parent -ItemType Directory -Force | Out-Null
    }

    if (-not (Test-Path $RepoPath)) {
        Write-Log "Cloning $RemoteUrl ($BranchName) into $RepoPath..."
        Invoke-ExternalWithRetry -FilePath $GitExe -Arguments @(
            "clone",
            "--branch", $BranchName,
            "--single-branch",
            $RemoteUrl,
            $RepoPath
        ) -MaxAttempts 3 -DelaySeconds 5
        $script:LastRepositorySyncSucceeded = $true
        return
    }

    if (-not (Test-Path (Join-Path $RepoPath ".git"))) {
        throw "Repo directory exists but is not a git repository: $RepoPath"
    }

    Write-Log "Updating existing repository at $RepoPath..."
    $statusOut = Get-GitStatusPorcelain -GitExe $GitExe -RepoPath $RepoPath
    if ($statusOut.ExitCode -ne 0) {
        throw "Failed to inspect git status in $RepoPath"
    }
    if (@($statusOut.TrackedOutput).Count -gt 0 -and $UpdatePolicy -eq "Safe") {
        $changeSummary = Format-StatusFileList -Items $statusOut.ChangedFiles
        Write-Log "Tracked local changes were found. Safe update is leaving the repository unchanged." "WARN"
        Write-Log ("Changed files: {0}" -f $changeSummary) "WARN"
        Write-Log "Run setup again with -UpdatePolicy StashAndUpdate to back up these changes with git stash and update." "WARN"
        return
    }
    if (@($statusOut.UntrackedOutput).Count -gt 0 -and $UpdatePolicy -eq "Safe") {
        $untrackedSummary = Format-StatusFileList -Items $statusOut.UntrackedFiles
        Write-Log ("Untracked local files will be left in place during the update: {0}" -f $untrackedSummary) "INFO"
        $conflictMessage = "If Git reports that an untracked file would be overwritten, rerun with -UpdatePolicy StashAndUpdate or move that file."
        Write-Log $conflictMessage "INFO"
    }
    if (@($statusOut.Output).Count -gt 0 -and $UpdatePolicy -eq "StashAndUpdate") {
        $backupStamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $backupUtc = (Get-Date).ToUniversalTime().ToString("o")
        $stashMessage = "musubi setup backup $backupStamp"
        $backupFiles = @($statusOut.LocalFiles)
        Write-Log "Backing up local changes with git stash before updating."
        Write-Log "This backup will not be reapplied automatically after the update."
        $stashOutput = & $GitExe -C $RepoPath stash push --include-untracked -m $stashMessage 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Could not create the git stash backup. The repository was not updated." "WARN"
            Write-ExternalOutputLines -Output $stashOutput -Level "WARN"
            return
        }
        Write-ExternalOutputLines -Output $stashOutput -Level "INFO"
        $stashRef = (& $GitExe -C $RepoPath stash list --format="%gd%x09%H%x09%s" -n 1 2>$null | Select-Object -First 1)
        $stashName = ""
        $stashSha = ""
        $stashSubject = ""
        if ($LASTEXITCODE -eq 0 -and $stashRef) {
            $stashParts = $stashRef.Trim().Split("`t", 3)
            if ($stashParts.Count -ge 1) {
                $stashName = $stashParts[0]
            }
            if ($stashParts.Count -ge 2) {
                $stashSha = $stashParts[1]
            }
            if ($stashParts.Count -ge 3) {
                $stashSubject = $stashParts[2]
            }
            Write-Log ("Backup created: {0} {1}" -f $stashName, $stashSubject).Trim() "OK"
        } else {
            Write-Log ("Backup created with message: {0}" -f $stashMessage) "OK"
        }
        $stashTarget = if (-not [string]::IsNullOrWhiteSpace($stashSha)) { $stashSha } else { $stashName }
        $script:LastRepositoryBackup = [ordered]@{
            created_utc = $backupUtc
            message = $stashMessage
            stash_ref = $stashName
            stash_sha = $stashSha
            stash_subject = $stashSubject
            recovery_target = $stashTarget
            files = $backupFiles
            commands = Get-StashRecoveryCommands -StashTarget $stashTarget
        }
        if (@($script:LastRepositoryBackup["commands"]).Count -ge 3) {
            Write-Log "To see backed-up files:" "INFO"
            Write-Log ("  {0}" -f $script:LastRepositoryBackup["commands"][0]) "INFO"
            Write-Log "To restore them into the updated repo:" "INFO"
            Write-Log ("  {0}" -f $script:LastRepositoryBackup["commands"][2]) "INFO"
        }
    }

    Invoke-ExternalWithRetry -FilePath $GitExe -Arguments @("-C", $RepoPath, "fetch", "origin", $BranchName) -MaxAttempts 3 -DelaySeconds 5
    $checkoutOutput = & $GitExe -C $RepoPath checkout $BranchName 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Git could not check out the requested branch. The repository was not updated." "WARN"
        Write-ExternalOutputLines -Output $checkoutOutput -Level "WARN"
        Write-Log "No local files were deleted by setup. Resolve the Git message above and rerun setup." "WARN"
        return
    }
    $pullOutput = & $GitExe -C $RepoPath pull --ff-only origin $BranchName 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Git could not fast-forward the repository. The repository was not updated." "WARN"
        Write-ExternalOutputLines -Output $pullOutput -Level "WARN"
        Write-Log "No local files were deleted by setup. Untracked files may conflict with incoming repository files." "WARN"
        Write-Log "Move the conflicting files or rerun setup with -UpdatePolicy StashAndUpdate." "WARN"
        return
    }
    Write-ExternalOutputLines -Output $pullOutput -Level "INFO"
    if ($UpdatePolicy -eq "StashAndUpdate" -and @($statusOut.Output).Count -gt 0) {
        $stashReminder = "Local changes remain backed up in git stash. Review them with 'git stash list' before applying or dropping them."
        Write-Log $stashReminder "WARN"
    }
    $script:LastRepositorySyncSucceeded = $true
}

function Ensure-Venv {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string]$VenvPath
    )

    $venvPython = Join-Path $VenvPath "Scripts\python.exe"
    if (Test-Path $venvPython) {
        Write-Log "Virtual environment already exists: $VenvPath"
        return
    }

    Write-Log "Creating virtual environment: $VenvPath"
    Invoke-External -FilePath $PythonExe -Arguments @("-m", "venv", $VenvPath)
}

function Get-TorchInstallArgs {
    param([Parameter(Mandatory = $true)][string]$CudaFlavor)

    switch ($CudaFlavor) {
        "cu124" {
            return @("install", "torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu124")
        }
        "cu128" {
            return @("install", "torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu128")
        }
        "cu130" {
            return @("install", "torch==2.9.1", "torchvision==0.24.1", "torchaudio==2.9.1", "--index-url", "https://download.pytorch.org/whl/cu130")
        }
        "cpu" {
            return @("install", "torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0", "--index-url", "https://download.pytorch.org/whl/cpu")
        }
        default {
            throw "Unsupported CUDA flavor: $CudaFlavor"
        }
    }
}

function Install-PythonDependencies {
    param(
        [Parameter(Mandatory = $true)][string]$VenvPath,
        [Parameter(Mandatory = $true)][string]$RepoPath,
        [Parameter(Mandatory = $true)][string]$CudaFlavor
    )

    $venvPython = Join-Path $VenvPath "Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        throw "python not found in venv: $venvPython"
    }

    Write-Log "Upgrading pip/setuptools/wheel..."
    Invoke-ExternalWithRetry -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel") -MaxAttempts 3 -DelaySeconds 5

    Write-Log "Installing PyTorch for $CudaFlavor..."
    Invoke-ExternalWithRetry -FilePath $venvPython -Arguments (@("-m", "pip") + (Get-TorchInstallArgs -CudaFlavor $CudaFlavor)) -MaxAttempts 3 -DelaySeconds 6

    Write-Log "Installing musubi-tuner dashboard dependencies..."
    Invoke-ExternalWithRetry -FilePath $venvPython -Arguments @("-m", "pip", "install", "-e", "$RepoPath[dashboard]") -MaxAttempts 3 -DelaySeconds 6

    Write-Log "Installing optional utility packages..."
    Invoke-ExternalWithRetry -FilePath $venvPython -Arguments @("-m", "pip", "install", "ascii-magic", "matplotlib", "tensorboard", "prompt-toolkit") -MaxAttempts 3 -DelaySeconds 6

    Write-Log "Running quick import health check..."
    Invoke-External -FilePath $venvPython -Arguments @("-c", "import musubi_tuner, fastapi; print('healthcheck: ok')")
}

function Build-Frontend {
    param([Parameter(Mandatory = $true)][string]$RepoPath)

    $frontendDir = Join-Path $RepoPath "src\musubi_tuner\gui_dashboard\frontend"
    if (-not (Test-Path $frontendDir)) {
        throw "Frontend directory not found: $frontendDir"
    }

    $npmCmd = Get-Command "npm.cmd" -ErrorAction SilentlyContinue
    if (-not $npmCmd) {
        $npmCmd = Get-Command "npm" -ErrorAction SilentlyContinue
    }
    if (-not $npmCmd) {
        throw "npm is not available. Install Node.js or disable frontend build."
    }

    Push-Location $frontendDir
    try {
        if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
            Write-Log "Installing frontend dependencies..."
            Invoke-ExternalWithRetry -FilePath $npmCmd.Source -Arguments @("install") -MaxAttempts 3 -DelaySeconds 5
        }
        Write-Log "Building frontend..."
        Invoke-ExternalWithRetry -FilePath $npmCmd.Source -Arguments @("run", "build") -MaxAttempts 2 -DelaySeconds 5
    } finally {
        Pop-Location
    }
}

function Write-LauncherScript {
    param(
        [Parameter(Mandatory = $true)][string]$RepoPath,
        [Parameter(Mandatory = $true)][string]$HostValue,
        [Parameter(Mandatory = $true)][int]$PortValue
    )

    $launcherPath = Join-Path $RepoPath $script:DashboardLauncherName
    $browserUrl = Get-BrowserUrl -HostValue $HostValue -PortValue $PortValue
    $contents = @(
        "@echo off",
        "setlocal",
        "for %%I in (""%~dp0."") do set ""REPO_DIR=%%~fI""",
        "set ""PYTHONPATH=%REPO_DIR%\src;%PYTHONPATH%""",
        "set ""VENV_PY=%REPO_DIR%\venv\Scripts\python.exe""",
        "if not exist ""%VENV_PY%"" (",
        "  echo Virtual environment python not found: %VENV_PY%",
        "  pause",
        "  exit /b 1",
        ")",
        "set ""DASHBOARD_URL=$browserUrl""",
        "echo [dashboard] Starting Musubi dashboard at %DASHBOARD_URL%",
        "start """" ""%DASHBOARD_URL%""",
        """%VENV_PY%"" -m musubi_tuner.gui_dashboard --host $HostValue --port $PortValue",
        "set ""EXIT_CODE=%ERRORLEVEL%""",
        "if not ""%EXIT_CODE%""==""0"" pause",
        "exit /b %EXIT_CODE%"
    )
    Set-Content -Path $launcherPath -Value $contents -Encoding ASCII
    return $launcherPath
}

function Write-SetupLauncherScript {
    param(
        [Parameter(Mandatory = $true)][string]$RepoPath,
        [Parameter(Mandatory = $true)][string]$RepoUrlValue,
        [Parameter(Mandatory = $true)][string]$BranchValue,
        [Parameter(Mandatory = $true)][string]$CudaValue,
        [Parameter(Mandatory = $true)][string]$PythonVersionValue,
        [Parameter(Mandatory = $true)][string]$HostValue,
        [Parameter(Mandatory = $true)][int]$PortValue
    )

    $launcherPath = Join-Path $RepoPath $script:SetupLauncherName
    $contents = @(
        "@echo off",
        "setlocal",
        "for %%I in (""%~dp0."") do set ""REPO_DIR=%%~fI""",
        "for %%I in (""%REPO_DIR%\.."") do set ""INSTALL_ROOT=%%~fI""",
        "set ""INSTALL_PS=%REPO_DIR%\scripts\install.ps1""",
        "if not exist ""%INSTALL_PS%"" (",
        "  echo Installer script not found: %INSTALL_PS%",
        "  pause",
        "  exit /b 1",
        ")",
        "powershell.exe -NoProfile -ExecutionPolicy Bypass -File ""%INSTALL_PS%"" -InstallRoot ""%INSTALL_ROOT%"" -RepoDir ""%REPO_DIR%"" -RepoUrl ""$RepoUrlValue"" -Branch ""$BranchValue"" -Cuda ""$CudaValue"" -PythonVersion ""$PythonVersionValue"" -DashboardHost ""$HostValue"" -Port $PortValue %*",
        "set ""EXIT_CODE=%ERRORLEVEL%""",
        "if not ""%EXIT_CODE%""==""0"" pause",
        "exit /b %EXIT_CODE%"
    )
    Set-Content -Path $launcherPath -Value $contents -Encoding ASCII
    return $launcherPath
}

function Get-BrowserUrl {
    param(
        [Parameter(Mandatory = $true)][string]$HostValue,
        [Parameter(Mandatory = $true)][int]$PortValue
    )

    $browserHost = $HostValue
    if ($browserHost -in @("0.0.0.0", "::", "*", "+")) {
        $browserHost = "127.0.0.1"
    }
    return "http://{0}:{1}/" -f $browserHost, $PortValue
}

function Create-DesktopShortcut {
    param(
        [Parameter(Mandatory = $true)][string]$ShortcutName,
        [Parameter(Mandatory = $true)][string]$ShortcutTarget,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$Description,
        [string]$IconLocation = "$env:SystemRoot\System32\shell32.dll,13"
    )

    $desktop = [Environment]::GetFolderPath("Desktop")
    $shortcutPath = Join-Path $desktop $ShortcutName

    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $ShortcutTarget
    $shortcut.WorkingDirectory = $WorkingDirectory
    $shortcut.Description = $Description
    $shortcut.IconLocation = $IconLocation
    $shortcut.Save()

    Write-Log "Desktop shortcut created: $shortcutPath" "OK"
    return $shortcutPath
}

function Read-SelectionInput {
    try {
        if (-not [Console]::IsInputRedirected) {
            $key = [Console]::ReadKey($true)
            if ($key.Key -eq [ConsoleKey]::Enter) {
                return "C"
            }
            if ($key.KeyChar) {
                return $key.KeyChar.ToString()
            }
        }
    } catch {
    }

    return (Read-Host "Selection").Trim()
}

function Select-Branch {
    param(
        [Parameter(Mandatory = $true)][string]$CurrentBranch,
        [Parameter(Mandatory = $true)][string]$RepositoryPath,
        [switch]$SkipPrompt
    )

    if ($SkipPrompt) {
        return $CurrentBranch
    }

    $actualBranch = ""
    $gitExeObj = Get-Command git -ErrorAction SilentlyContinue
    if ($gitExeObj -and (Test-Path (Join-Path $RepositoryPath ".git"))) {
        $actualBranch = (& $gitExeObj.Source -C $RepositoryPath rev-parse --abbrev-ref HEAD 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -eq 0 -and $actualBranch) {
            $actualBranch = $actualBranch.Trim()
        } else {
            $actualBranch = ""
        }
    }

    while ($true) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host " Musubi LTX-2 Setup / Update - Branch Selection" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host (" Selected branch: {0}" -f $CurrentBranch) -ForegroundColor DarkGray
        if ($actualBranch) {
            Write-Host (" Checked-out repo branch: {0}" -f $actualBranch) -ForegroundColor DarkGray
        }
        Write-Host ""
        Write-Host " 1. ltx-2"
        Write-Host " 2. ltx-2-dev"
        Write-Host ""
        Write-Host "Press [Enter]/[C] to continue with the current selection." -ForegroundColor DarkGray
        $choice = (Read-SelectionInput).Trim()

        if ([string]::IsNullOrWhiteSpace($choice)) {
            return $CurrentBranch
        }

        $upper = $choice.ToUpperInvariant()
        if ($upper -eq "C") {
            return $CurrentBranch
        }
        if ($upper -eq "Q") {
            throw "Installation cancelled by user."
        }
        if ($choice -eq "1") {
            $CurrentBranch = "ltx-2"
            continue
        }
        if ($choice -eq "2") {
            $CurrentBranch = "ltx-2-dev"
            continue
        }

        Write-Host "Invalid selection." -ForegroundColor Yellow
    }
}

function Select-UpdatePolicy {
    param(
        [Parameter(Mandatory = $true)][string]$CurrentPolicy,
        [Parameter(Mandatory = $true)]$RepoStatus,
        [Parameter(Mandatory = $true)][bool]$BranchSwitchRequested,
        [switch]$SkipPrompt
    )

    if ($SkipPrompt -or $CurrentPolicy -eq "StashAndUpdate" -or (-not $RepoStatus.tracked_dirty)) {
        return $CurrentPolicy
    }
    if ((-not $RepoStatus.update_available) -and (-not $BranchSwitchRequested)) {
        return $CurrentPolicy
    }

    while ($true) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host " Musubi LTX-2 Setup / Update - Local Changes" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "Tracked local changes were found. Safe update will not overwrite or delete them." -ForegroundColor Yellow
        Write-Host ("Changed files: {0}" -f (Format-StatusFileList -Items $RepoStatus.changed_files)) -ForegroundColor DarkGray
        Write-Host ""
        Write-Host " S. Skip repository update for this run (safe default)"
        Write-Host " B. Back up local changes with git stash, then update"
        Write-Host ""
        Write-Host "The stash backup will not be reapplied automatically after setup finishes." -ForegroundColor DarkGray
        Write-Host "Press [Enter]/[S] to keep the safe default, or [Q] to quit." -ForegroundColor DarkGray

        $choice = (Read-SelectionInput).Trim().ToUpperInvariant()
        if ([string]::IsNullOrWhiteSpace($choice) -or $choice -eq "S") {
            return "Safe"
        }
        if ($choice -eq "B") {
            return "StashAndUpdate"
        }
        if ($choice -eq "Q") {
            throw "Installation cancelled by user."
        }
        Write-Host "Invalid selection." -ForegroundColor Yellow
    }
}

function Select-Actions {
    param(
        [Parameter(Mandatory = $true)][System.Collections.ArrayList]$Actions,
        [Parameter(Mandatory = $true)]$InitialScan,
        [Parameter(Mandatory = $true)][string]$RepositoryPath,
        [Parameter(Mandatory = $true)][string]$VenvPath,
        [string[]]$OverviewLines = @(),
        [switch]$SkipPrompt
    )

    Update-ActionConstraints -Actions $Actions -InitialScan $InitialScan -RepositoryPath $RepositoryPath -VenvPath $VenvPath

    if ($SkipPrompt) {
        return
    }

    while ($true) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host " Musubi LTX-2 Setup / Update - Action Selection" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-InstallerOverview -Lines $OverviewLines
        for ($i = 0; $i -lt $Actions.Count; $i++) {
            $item = $Actions[$i]
            $mark = if ($item.Selected) { "x" } else { " " }
            $lockTag = if ($item.Locked) { " [locked]" } else { "" }
            $reason = ""
            if ($item.PSObject.Properties.Name -contains "Reason") {
                if (-not [string]::IsNullOrWhiteSpace($item.Reason)) {
                    $reason = " -- $($item.Reason)"
                }
            }
            Write-Host (" {0}. [{1}] {2}{3}{4}" -f ($i + 1), $mark, $item.Label, $lockTag, $reason)
        }
        Write-Host ""
        Write-Host "Press number to toggle immediately. [Enter]/[C] Continue, [A]ll, [N]one, [Q]uit" -ForegroundColor DarkGray
        $choice = (Read-SelectionInput).Trim()

        if ([string]::IsNullOrWhiteSpace($choice)) {
            continue
        }

        $upper = $choice.ToUpperInvariant()
        if ($upper -eq "C") {
            if ($Actions.Where({ $_.Selected }).Count -eq 0) {
                Write-Host "At least one action must be selected." -ForegroundColor Yellow
                continue
            }
            break
        }
        if ($upper -eq "Q") {
            throw "Installation cancelled by user."
        }
        if ($upper -eq "A") {
            foreach ($action in $Actions) {
                if (-not $action.Locked) {
                    $action.Selected = $true
                }
            }
            Update-ActionConstraints -Actions $Actions -InitialScan $InitialScan -RepositoryPath $RepositoryPath -VenvPath $VenvPath
            continue
        }
        if ($upper -eq "N") {
            foreach ($action in $Actions) {
                if (-not $action.Locked) {
                    $action.Selected = $false
                }
            }
            Update-ActionConstraints -Actions $Actions -InitialScan $InitialScan -RepositoryPath $RepositoryPath -VenvPath $VenvPath
            continue
        }

        $num = 0
        if ([int]::TryParse($choice, [ref]$num)) {
            if ($num -ge 1 -and $num -le $Actions.Count) {
                $target = $Actions[$num - 1]
                if ($target.Locked) {
                    Write-Host "That action is currently required and cannot be turned off." -ForegroundColor Yellow
                    continue
                }
                $target.Selected = -not $target.Selected
                Update-ActionConstraints -Actions $Actions -InitialScan $InitialScan -RepositoryPath $RepositoryPath -VenvPath $VenvPath
                continue
            }
        }

        Write-Host "Invalid selection." -ForegroundColor Yellow
    }
}

function Get-ActionState {
    param(
        [Parameter(Mandatory = $true)][System.Collections.ArrayList]$Actions,
        [Parameter(Mandatory = $true)][string]$Key
    )
    $item = $Actions | Where-Object { $_.Key -eq $Key } | Select-Object -First 1
    if (-not $item) { return $false }
    return [bool]$item.Selected
}

function Get-ActionItem {
    param(
        [Parameter(Mandatory = $true)][System.Collections.ArrayList]$Actions,
        [Parameter(Mandatory = $true)][string]$Key
    )

    return $Actions | Where-Object { $_.Key -eq $Key } | Select-Object -First 1
}

function Update-ActionConstraints {
    param(
        [Parameter(Mandatory = $true)][System.Collections.ArrayList]$Actions,
        [Parameter(Mandatory = $true)]$InitialScan,
        [Parameter(Mandatory = $true)][string]$RepositoryPath,
        [Parameter(Mandatory = $true)][string]$VenvPath
    )

    foreach ($action in $Actions) {
        $action.Locked = $false
        $action.Reason = $action.BaseReason
    }

    $repoExists = Test-Path $RepositoryPath
    $venvExists = Test-Path (Join-Path $VenvPath "Scripts\python.exe")
    $frontendDistExists = Test-Path (Join-Path $RepositoryPath "src\musubi_tuner\gui_dashboard\frontend\dist\index.html")

    $cloneAction = Get-ActionItem -Actions $Actions -Key "CloneOrUpdate"
    if ($cloneAction -and (-not $repoExists)) {
        $cloneAction.Selected = $true
        $cloneAction.Locked = $true
        $cloneAction.Reason = "required because repository does not exist yet"
    }

    $installGitAction = Get-ActionItem -Actions $Actions -Key "InstallGit"
    if ($installGitAction -and (Get-ActionState -Actions $Actions -Key "CloneOrUpdate") -and (-not $InitialScan.GitInstalled)) {
        $installGitAction.Selected = $true
        $installGitAction.Locked = $true
        $installGitAction.Reason = "required because git is needed for clone/update"
    }

    $needsPython = (Get-ActionState -Actions $Actions -Key "CreateVenv") -or (Get-ActionState -Actions $Actions -Key "InstallDeps")
    $installPythonAction = Get-ActionItem -Actions $Actions -Key "InstallPython"
    if ($installPythonAction -and $needsPython -and (-not $InitialScan.PythonAnySupported)) {
        $installPythonAction.Selected = $true
        $installPythonAction.Locked = $true
        $installPythonAction.Reason = "required because no compatible Python 3.10+ was found"
    }

    $createVenvAction = Get-ActionItem -Actions $Actions -Key "CreateVenv"
    if ($createVenvAction -and (Get-ActionState -Actions $Actions -Key "InstallDeps") -and (-not $venvExists)) {
        $createVenvAction.Selected = $true
        $createVenvAction.Locked = $true
        $createVenvAction.Reason = "required because the virtual environment does not exist yet"
    }

    $installNodeAction = Get-ActionItem -Actions $Actions -Key "InstallNode"
    if ($installNodeAction -and (Get-ActionState -Actions $Actions -Key "BuildFrontend") -and (-not $InitialScan.NodeInstalled)) {
        $installNodeAction.Selected = $true
        $installNodeAction.Locked = $true
        $installNodeAction.Reason = "required because npm is needed for frontend build"
    }

    $buildFrontendAction = Get-ActionItem -Actions $Actions -Key "BuildFrontend"
    if ($buildFrontendAction -and $repoExists -and (-not $frontendDistExists)) {
        $buildFrontendAction.Selected = $true
        $buildFrontendAction.Locked = $true
        $buildFrontendAction.Reason = "required because the frontend dist folder is missing"
    }
}

try {
    if (-not (Test-WindowsHost)) {
        throw "This installer currently supports Windows only."
    }

    if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
        $InstallRoot = (Get-Location).ProviderPath
    }

    if ([string]::IsNullOrWhiteSpace($RepoDir)) {
        $RepoDir = Join-Path $InstallRoot "musubi-tuner-ltx2"
    }

    $InstallRoot = [Environment]::ExpandEnvironmentVariables($InstallRoot)
    $RepoDir = [Environment]::ExpandEnvironmentVariables($RepoDir)
    $VenvDir = Join-Path $RepoDir "venv"
    $branchWasProvided = $PSBoundParameters.ContainsKey("Branch")
    $existingState = Get-InstallState -RepoPath $RepoDir
    if (-not $branchWasProvided) {
        $savedBranch = [string](Get-OptionalPropertyValue -Object $existingState -Name "branch" -Default "")
        if (-not [string]::IsNullOrWhiteSpace($savedBranch)) {
            $Branch = $savedBranch
        }
    }

    Write-Log "Installer log file: $script:LogFile"
    Write-Log "Install root: $InstallRoot"
    Write-Log "Repository path: $RepoDir"
    Write-Log "Repository URL: $RepoUrl"
    Write-Log "Branch: $Branch"
    Write-Log "CUDA target: $Cuda"
    Write-Log "Preferred Python: $PythonVersion"
    Write-Log "Dashboard host: $DashboardHost"
    Write-Log "Update policy: $UpdatePolicy"
    Write-Log ("Browser URL: {0}" -f (Get-BrowserUrl -HostValue $DashboardHost -PortValue $Port))
    Ensure-Tls12ForLegacyPowerShell

    $initialScan = Get-InitialEnvironmentScan -PreferredPythonVersion $PythonVersion
    Show-InitialEnvironmentScan -Scan $initialScan
    $Branch = Select-Branch -CurrentBranch $Branch -RepositoryPath $RepoDir -SkipPrompt:$NonInteractive

    $defaultInstallGit = -not $initialScan.GitInstalled
    $defaultInstallPython = -not $initialScan.PythonAnySupported
    $defaultInstallNode = -not $initialScan.NodeInstalled

    $repoExists = Test-Path $RepoDir
    $venvPythonPath = Join-Path $VenvDir "Scripts\python.exe"
    $venvExists = Test-Path $venvPythonPath
    $frontendDistPath = Join-Path $RepoDir "src\musubi_tuner\gui_dashboard\frontend\dist\index.html"
    $frontendDistExists = Test-Path $frontendDistPath
    $desktop = [Environment]::GetFolderPath("Desktop")
    $dashboardShortcutPath = Join-Path $desktop $script:DashboardShortcutName
    $setupShortcutPath = Join-Path $desktop $script:SetupShortcutName
    $dashboardShortcutExists = Test-Path $dashboardShortcutPath
    $setupShortcutExists = Test-Path $setupShortcutPath
    $gitExeObj = Get-Command git -ErrorAction SilentlyContinue
    $repoStatus = Get-RepositoryStatus -GitExe $(if ($gitExeObj) { $gitExeObj.Source } else { "" }) -RepoPath $RepoDir -BranchName $Branch -RemoteUrl $RepoUrl

    $stateLastSuccessUtc = ""
    $stateDepsCommit = ""
    $stateDepsTimestampUtc = ""
    $stateFrontendCommit = ""
    $stateFrontendTimestampUtc = ""
    $stateCuda = ""
    $stateLastBackup = $null
    if ($existingState) {
        $stateLastSuccessUtc = [string](Get-NestedPropertyValue -Object $existingState -Path @("install", "last_success_utc") -Default "")
        $stateDepsCommit = [string](Get-NestedPropertyValue -Object $existingState -Path @("install", "deps_commit") -Default "")
        $stateDepsTimestampUtc = [string](Get-NestedPropertyValue -Object $existingState -Path @("install", "deps_timestamp_utc") -Default "")
        $stateFrontendCommit = [string](Get-NestedPropertyValue -Object $existingState -Path @("install", "frontend_commit") -Default "")
        $stateFrontendTimestampUtc = [string](Get-NestedPropertyValue -Object $existingState -Path @("install", "frontend_timestamp_utc") -Default "")
        $stateCuda = [string](Get-OptionalPropertyValue -Object $existingState -Name "cuda" -Default "")
        $stateLastBackup = Get-NestedPropertyValue -Object $existingState -Path @("install", "last_backup") -Default $null
    }

    $branchSwitchRequested = $repoExists -and $repoStatus.is_git_repo -and (-not [string]::IsNullOrWhiteSpace($repoStatus.branch)) -and ($repoStatus.branch -ne $Branch)
    $updatePolicyArgs = @{
        CurrentPolicy = $UpdatePolicy
        RepoStatus = $repoStatus
        BranchSwitchRequested = $branchSwitchRequested
        SkipPrompt = [bool]$NonInteractive
    }
    $UpdatePolicy = Select-UpdatePolicy @updatePolicyArgs
    $stashCanHandleTrackedChanges = $UpdatePolicy -eq "StashAndUpdate" -and $repoStatus.tracked_dirty

    $comparisonTargetRef = ""
    $canCompareRemote = $repoStatus.can_auto_update -or ($UpdatePolicy -eq "StashAndUpdate" -and $repoStatus.update_available)
    if ($canCompareRemote -and $repoStatus.remote_head) {
        $comparisonTargetRef = $repoStatus.remote_head
    } elseif ($repoStatus.head) {
        $comparisonTargetRef = $repoStatus.head
    }

    $dependencyPathSpecs = @("pyproject.toml", "uv.lock")
    $frontendPathSpecs = @(
        "src/musubi_tuner/gui_dashboard/frontend/package.json",
        "src/musubi_tuner/gui_dashboard/frontend/package-lock.json",
        "src/musubi_tuner/gui_dashboard/frontend/src",
        "src/musubi_tuner/gui_dashboard/frontend/svelte.config.js",
        "src/musubi_tuner/gui_dashboard/frontend/vite.config.js"
    )

    $depsChanged = $false
    $frontendChanged = $false
    if ($gitExeObj -and $repoStatus.is_git_repo -and $comparisonTargetRef) {
        if ($stateDepsCommit) {
            $depsChanged = Test-GitPathsChanged -GitExe $gitExeObj.Source -RepoPath $RepoDir -FromRef $stateDepsCommit -ToRef $comparisonTargetRef -PathSpecs $dependencyPathSpecs
        } elseif ($canCompareRemote -and $repoStatus.head) {
            $depsChanged = Test-GitPathsChanged -GitExe $gitExeObj.Source -RepoPath $RepoDir -FromRef $repoStatus.head -ToRef $comparisonTargetRef -PathSpecs $dependencyPathSpecs
        }

        if ($stateFrontendCommit) {
            $frontendChanged = Test-GitPathsChanged -GitExe $gitExeObj.Source -RepoPath $RepoDir -FromRef $stateFrontendCommit -ToRef $comparisonTargetRef -PathSpecs $frontendPathSpecs
        } elseif ($canCompareRemote -and $repoStatus.head) {
            $frontendChanged = Test-GitPathsChanged -GitExe $gitExeObj.Source -RepoPath $RepoDir -FromRef $repoStatus.head -ToRef $comparisonTargetRef -PathSpecs $frontendPathSpecs
        }
    }

    $defaultCloneOrUpdate = (
        (-not $repoExists) -or
        ($branchSwitchRequested -and ((-not $repoStatus.tracked_dirty) -or $stashCanHandleTrackedChanges)) -or
        $repoStatus.can_auto_update -or
        ($stashCanHandleTrackedChanges -and $repoStatus.update_available)
    )
    $defaultCreateVenv = -not $venvExists
    $defaultInstallDeps = (-not $venvExists) -or $depsChanged -or (($stateCuda) -and ($stateCuda -ne $Cuda))
    $defaultBuildFrontend = ($repoExists -and (-not $frontendDistExists)) -or $frontendChanged
    $defaultCreateShortcut = -not ($dashboardShortcutExists -and $setupShortcutExists)
    $localChangeSummary = Format-StatusFileList -Items $repoStatus.changed_files -MaxItems 3

    $cloneBaseReason = if (-not $repoExists) {
        "required because repository does not exist yet"
    } elseif ($branchSwitchRequested -and $repoStatus.tracked_dirty -and $UpdatePolicy -eq "StashAndUpdate") {
        "branch switch requested; tracked changes will be backed up with git stash before checkout"
    } elseif ($branchSwitchRequested -and $repoStatus.tracked_dirty) {
        "branch switch requested, but tracked local changes block safe checkout: $localChangeSummary"
    } elseif ($branchSwitchRequested) {
        "recommended because setup is configured for branch '$Branch' while the repo is on '$($repoStatus.branch)'"
    } elseif ($repoStatus.tracked_dirty -and $repoStatus.update_available -and $UpdatePolicy -eq "StashAndUpdate") {
        "tracked local changes will be backed up with git stash before updating"
    } elseif ($repoStatus.tracked_dirty -and $repoStatus.update_available) {
        "tracked local changes detected; safe update is not auto-selected: $localChangeSummary"
    } elseif ($repoStatus.tracked_dirty) {
        "tracked local changes detected; repo sync is left off to avoid conflicts: $localChangeSummary"
    } elseif (-not $repoStatus.origin_configured -and $repoStatus.is_git_repo) {
        "origin remote is missing; automatic update checks are unavailable"
    } elseif ($repoStatus.update_available) {
        "recommended because origin/$Branch is $($repoStatus.remote_ahead_count) commit(s) ahead"
    } elseif ($repoStatus.fetch_attempted -and (-not $repoStatus.fetch_succeeded)) {
        "origin fetch failed; leaving repo sync off"
    } else {
        "repository is already up to date"
    }

    $createVenvBaseReason = if ($defaultCreateVenv) {
        "missing (auto-selected)"
    } else {
        "existing virtual environment detected"
    }

    $installDepsBaseReason = if (-not $venvExists) {
        "required because the virtual environment is missing"
    } elseif ($stateCuda -and ($stateCuda -ne $Cuda)) {
        "recommended because the CUDA target changed from $stateCuda to $Cuda"
    } elseif ($depsChanged) {
        "recommended because Python dependency inputs changed"
    } else {
        "dependency inputs look unchanged"
    }

    $buildFrontendBaseReason = if ($repoExists -and (-not $frontendDistExists)) {
        "required because the frontend dist folder is missing"
    } elseif ($frontendChanged) {
        "recommended because dashboard frontend sources changed"
    } else {
        "frontend build looks current"
    }

    $shortcutBaseReason = if ($dashboardShortcutExists -and $setupShortcutExists) {
        "both desktop shortcuts already exist"
    } elseif ($dashboardShortcutExists -or $setupShortcutExists) {
        "recommended to create the missing desktop shortcut"
    } else {
        "recommended"
    }

    $overviewLines = @(
        ("Mode: {0}" -f $(if ($repoExists) { "manage existing install" } else { "first-time setup" })),
        ("Configured branch: {0}" -f $Branch),
        ("Checked-out repo branch: {0}" -f $(if ($repoStatus.branch) { $repoStatus.branch } else { "unknown" })),
        ("Update policy: {0}" -f $UpdatePolicy),
        ("Repo: {0}" -f $repoStatus.summary),
        ("Venv: {0}" -f $(if ($venvExists) { "ready" } else { "missing" })),
        ("Python deps: {0}" -f $(if ($defaultInstallDeps) { $installDepsBaseReason } else { "no reinstall recommended" })),
        ("Frontend: {0}" -f $(if ($defaultBuildFrontend) { $buildFrontendBaseReason } else { "no rebuild recommended" })),
        ("Shortcuts: dashboard={0}, setup={1}" -f $(if ($dashboardShortcutExists) { "present" } else { "missing" }), $(if ($setupShortcutExists) { "present" } else { "missing" })),
        ("Last successful setup: {0}" -f (Format-StateTimestamp -Timestamp $stateLastSuccessUtc))
    )
    if (@($repoStatus.changed_files).Count -gt 0) {
        $overviewLines += ("Tracked changes: {0}" -f (Format-StatusFileList -Items $repoStatus.changed_files))
    }
    if (@($repoStatus.untracked_files).Count -gt 0) {
        $untrackedSummary = Format-StatusFileList -Items $repoStatus.untracked_files
        $overviewLines += ("Untracked files: {0}" -f $untrackedSummary)
    }

    $actions = [System.Collections.ArrayList]::new()
    [void]$actions.Add([pscustomobject]@{
            Key = "InstallGit"; Label = "Install Git"; Selected = $defaultInstallGit
            BaseReason = if ($defaultInstallGit) { "missing (auto-selected)" } else { "already installed" }
            Reason = ""
            Locked = $false
        })
    [void]$actions.Add([pscustomobject]@{
            Key = "InstallPython"; Label = "Install preferred Python $PythonVersion (user-level, optional)"; Selected = $defaultInstallPython
            BaseReason = if ($defaultInstallPython) { "no compatible Python 3.10+ found (auto-selected)" } elseif ($initialScan.PythonExactInstalled) { "preferred version already installed" } else { "compatible Python 3.10+ already installed" }
            Reason = ""
            Locked = $false
        })
    [void]$actions.Add([pscustomobject]@{
            Key = "InstallNode"; Label = "Install Node.js LTS (optional)"; Selected = $defaultInstallNode
            BaseReason = if ($defaultInstallNode) { "missing (auto-selected)" } else { "already installed" }
            Reason = ""
            Locked = $false
        })
    [void]$actions.Add([pscustomobject]@{
            Key = "CloneOrUpdate"; Label = "Clone/update repository ($Branch branch)"; Selected = $defaultCloneOrUpdate
            BaseReason = $cloneBaseReason
            Reason = ""
            Locked = $false
        })
    [void]$actions.Add([pscustomobject]@{
            Key = "CreateVenv"; Label = "Create/reuse virtual environment"; Selected = $defaultCreateVenv
            BaseReason = $createVenvBaseReason
            Reason = ""
            Locked = $false
        })
    [void]$actions.Add([pscustomobject]@{
            Key = "InstallDeps"; Label = "Install Python dependencies (torch + musubi + dashboard)"; Selected = $defaultInstallDeps
            BaseReason = $installDepsBaseReason
            Reason = ""
            Locked = $false
        })
    [void]$actions.Add([pscustomobject]@{
            Key = "BuildFrontend"; Label = "Build dashboard frontend (npm run build)"; Selected = $defaultBuildFrontend
            BaseReason = $buildFrontendBaseReason
            Reason = ""
            Locked = $false
        })
    [void]$actions.Add([pscustomobject]@{
            Key = "CreateShortcut"; Label = "Create desktop shortcuts (dashboard + setup/update)"; Selected = $defaultCreateShortcut
            BaseReason = $shortcutBaseReason
            Reason = ""
            Locked = $false
        })
    [void]$actions.Add([pscustomobject]@{
            Key = "LaunchDashboard"; Label = "Launch dashboard after setup"; Selected = $false
            BaseReason = "optional"
            Reason = ""
            Locked = $false
        })

    Select-Actions -Actions $actions -InitialScan $initialScan -RepositoryPath $RepoDir -VenvPath $VenvDir -OverviewLines $overviewLines -SkipPrompt:$NonInteractive

    Write-Log "Final action set:"
    foreach ($action in $actions) {
        Write-Log (" - {0}: {1}" -f $action.Key, ($(if ($action.Selected) { "ON" } else { "OFF" })))
    }

    Invoke-Step -Name "Preflight" -Action {
        Invoke-Preflight -Actions $actions -InstallPath $InstallRoot -RepositoryPath $RepoDir -PreferredPythonVersion $PythonVersion -DashboardPort $Port -StrictMode:$StrictPreflight
    }
    if ($PreflightOnly) {
        Write-Log "Preflight-only mode requested. Exiting without making changes." "OK"
        Write-SupportBundle -Outcome "preflight_only"
        exit 0
    }

    if (Get-ActionState -Actions $actions -Key "InstallGit") {
        Invoke-Step -Name "InstallGit" -Action { Ensure-Git }
    }
    if (Get-ActionState -Actions $actions -Key "InstallPython") {
        Invoke-Step -Name "InstallPython" -Action { [void](Ensure-Python -Version $PythonVersion) }
    }
    if (Get-ActionState -Actions $actions -Key "InstallNode") {
        Invoke-Step -Name "InstallNode" -Action { Ensure-Node }
    }

    Invoke-Step -Name "RefreshPath" -Action { Refresh-Path }

    $gitExeObj = Get-Command git -ErrorAction SilentlyContinue
    $stateData = [ordered]@{
        schema_version = 1
        install_root = $InstallRoot
        repo_dir = $RepoDir
        repo_url = $RepoUrl
        branch = $Branch
        cuda = $Cuda
        python_version = $PythonVersion
        dashboard_host = $DashboardHost
        port = $Port
        update_policy = $UpdatePolicy
        repo = [ordered]@{
            last_checked_utc = ""
            head = ""
            head_short = ""
            branch = ""
            remote_head = ""
            remote_head_short = ""
            dirty = $false
            update_available = $false
            local_ahead_count = 0
            remote_ahead_count = 0
            diverged = $false
            tracked_dirty = $false
            has_untracked = $false
            last_sync_utc = ""
        }
        install = [ordered]@{
            last_success_utc = $stateLastSuccessUtc
            deps_commit = $stateDepsCommit
            deps_timestamp_utc = $stateDepsTimestampUtc
            frontend_commit = $stateFrontendCommit
            frontend_timestamp_utc = $stateFrontendTimestampUtc
            last_backup = $stateLastBackup
            venv_python = $venvPythonPath
            venv_exists = $venvExists
            dashboard_launcher_path = Join-Path $RepoDir $script:DashboardLauncherName
            setup_launcher_path = Join-Path $RepoDir $script:SetupLauncherName
            dashboard_shortcut_path = $dashboardShortcutPath
            setup_shortcut_path = $setupShortcutPath
        }
    }

    if (Get-ActionState -Actions $actions -Key "CloneOrUpdate") {
        if (-not $gitExeObj) {
            throw "git is required for clone/update. Enable 'Install Git' or install git manually."
        }
        $script:LastRepositorySyncSucceeded = $false
        Invoke-Step -Name "SyncRepository" -Action {
            $syncArgs = @{
                GitExe = $gitExeObj.Source
                RepoPath = $RepoDir
                RemoteUrl = $RepoUrl
                BranchName = $Branch
                UpdatePolicy = $UpdatePolicy
            }
            Sync-Repository @syncArgs
        }
    } elseif (-not (Test-Path $RepoDir)) {
        throw "Repository directory does not exist: $RepoDir"
    }

    $gitExeObj = Get-Command git -ErrorAction SilentlyContinue
    $repoStatus = Get-RepositoryStatus -GitExe $(if ($gitExeObj) { $gitExeObj.Source } else { "" }) -RepoPath $RepoDir -BranchName $Branch -RemoteUrl $RepoUrl
    $stateData["repo"]["last_checked_utc"] = (Get-Date).ToUniversalTime().ToString("o")
    $stateData["repo"]["head"] = $repoStatus.head
    $stateData["repo"]["head_short"] = $repoStatus.head_short
    $stateData["repo"]["branch"] = $repoStatus.branch
    $stateData["repo"]["remote_head"] = $repoStatus.remote_head
    $stateData["repo"]["remote_head_short"] = $repoStatus.remote_head_short
    $stateData["repo"]["dirty"] = [bool]$repoStatus.dirty
    $stateData["repo"]["tracked_dirty"] = [bool]$repoStatus.tracked_dirty
    $stateData["repo"]["has_untracked"] = [bool]$repoStatus.has_untracked
    $stateData["repo"]["update_available"] = [bool]$repoStatus.update_available
    $stateData["repo"]["local_ahead_count"] = [int]$repoStatus.local_ahead_count
    $stateData["repo"]["remote_ahead_count"] = [int]$repoStatus.remote_ahead_count
    $stateData["repo"]["diverged"] = [bool]$repoStatus.diverged
    if ((Get-ActionState -Actions $actions -Key "CloneOrUpdate") -and $script:LastRepositorySyncSucceeded) {
        $stateData["repo"]["last_sync_utc"] = (Get-Date).ToUniversalTime().ToString("o")
    }
    if ($script:LastRepositoryBackup) {
        $stateData["install"]["last_backup"] = $script:LastRepositoryBackup
    }
    Save-InstallState -RepoPath $RepoDir -State $stateData

    if (Get-ActionState -Actions $actions -Key "CreateVenv") {
        Invoke-Step -Name "CreateVenv" -Action {
            $pythonExe = Resolve-PythonForBuild -PreferredVersion $PythonVersion
            Ensure-Venv -PythonExe $pythonExe -VenvPath $VenvDir
        }
    }

    if (Get-ActionState -Actions $actions -Key "InstallDeps") {
        if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
            throw "Virtual environment not found at $VenvDir. Enable 'Create/reuse virtual environment' or create it manually."
        }
        Invoke-Step -Name "InstallDependencies" -Action {
            Install-PythonDependencies -VenvPath $VenvDir -RepoPath $RepoDir -CudaFlavor $Cuda
        }
        $stateData["install"]["deps_commit"] = $repoStatus.head
        $stateData["install"]["deps_timestamp_utc"] = (Get-Date).ToUniversalTime().ToString("o")
    }

    $frontendBuilt = $false
    if (Get-ActionState -Actions $actions -Key "BuildFrontend") {
        if (-not (Test-Path $RepoDir)) {
            throw "Repository directory does not exist: $RepoDir"
        }
        if (-not (Test-AnyCommand @("npm.cmd", "npm"))) {
            throw "Node.js/npm not found. Enable 'Install Node.js' or disable frontend build."
        }
        Invoke-Step -Name "BuildFrontend" -Action { Build-Frontend -RepoPath $RepoDir }
        $frontendBuilt = $true
    } elseif (-not (Test-Path $frontendDistPath)) {
        if (-not (Test-AnyCommand @("npm.cmd", "npm"))) {
            throw "Frontend dist is missing after repository sync. Enable 'Build dashboard frontend' or install Node.js."
        }
        Invoke-Step -Name "BuildFrontendAuto" -Action {
            Write-Log "Frontend dist is missing after repository sync. Building it automatically..." "WARN"
            Build-Frontend -RepoPath $RepoDir
        }
        $frontendBuilt = $true
    }

    if ($frontendBuilt) {
        $stateData["install"]["frontend_commit"] = $repoStatus.head
        $stateData["install"]["frontend_timestamp_utc"] = (Get-Date).ToUniversalTime().ToString("o")
    }

    $stateData["install"]["venv_exists"] = Test-Path (Join-Path $VenvDir "Scripts\python.exe")
    Save-InstallState -RepoPath $RepoDir -State $stateData

    $launcherPath = $null
    $setupLauncherPath = $null
    Invoke-Step -Name "WriteLauncherScript" -Action {
        $script:launcherPathInternal = Write-LauncherScript -RepoPath $RepoDir -HostValue $DashboardHost -PortValue $Port
    }
    $launcherPath = $script:launcherPathInternal
    Invoke-Step -Name "WriteSetupLauncherScript" -Action {
        $script:setupLauncherPathInternal = Write-SetupLauncherScript -RepoPath $RepoDir -RepoUrlValue $RepoUrl -BranchValue $Branch -CudaValue $Cuda -PythonVersionValue $PythonVersion -HostValue $DashboardHost -PortValue $Port
    }
    $setupLauncherPath = $script:setupLauncherPathInternal
    $stateData["install"]["dashboard_launcher_path"] = $launcherPath
    $stateData["install"]["setup_launcher_path"] = $setupLauncherPath
    Save-InstallState -RepoPath $RepoDir -State $stateData
    Write-Log "Dashboard launcher ready: $launcherPath" "OK"
    Write-Log "Setup/update launcher ready: $setupLauncherPath" "OK"

    if (Get-ActionState -Actions $actions -Key "CreateShortcut") {
        Invoke-Step -Name "CreateDesktopShortcuts" -Action {
            $script:dashboardShortcutCreated = Create-DesktopShortcut -ShortcutName $script:DashboardShortcutName -ShortcutTarget $launcherPath -WorkingDirectory $RepoDir -Description "Launch Musubi Tuner Dashboard" -IconLocation "$env:SystemRoot\System32\shell32.dll,13"
            $script:setupShortcutCreated = Create-DesktopShortcut -ShortcutName $script:SetupShortcutName -ShortcutTarget $setupLauncherPath -WorkingDirectory $RepoDir -Description "Open Musubi Tuner Setup and Update" -IconLocation "$env:SystemRoot\System32\shell32.dll,22"
        }
        $stateData["install"]["dashboard_shortcut_path"] = $script:dashboardShortcutCreated
        $stateData["install"]["setup_shortcut_path"] = $script:setupShortcutCreated
    }

    $stateData["install"]["last_success_utc"] = (Get-Date).ToUniversalTime().ToString("o")
    Save-InstallState -RepoPath $RepoDir -State $stateData

    if (Get-ActionState -Actions $actions -Key "LaunchDashboard") {
        Invoke-Step -Name "LaunchDashboard" -Action {
            Write-Log "Launching dashboard..."
            Start-Process -FilePath $launcherPath -WorkingDirectory $RepoDir
        }
    }

    Write-Log "Installation completed successfully." "OK"
    Write-Log "Dashboard launcher: $launcherPath"
    Write-Log "Setup/update launcher: $setupLauncherPath"
    Write-Log ("Open in your browser: {0}" -f (Get-BrowserUrl -HostValue $DashboardHost -PortValue $Port)) "OK"
    Write-SupportBundle -Outcome "success"
} catch {
    Write-Log ("Installation failed: {0}" -f $_.Exception.Message) "ERROR"
    Write-Log "See full log: $script:LogFile" "ERROR"
    Write-SupportBundle -Outcome "failure" -ErrorRecord $_
    throw
}
