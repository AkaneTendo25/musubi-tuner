[CmdletBinding()]
param(
    [string]$InstallRoot = "",
    [string]$RepoUrl = "https://github.com/AkaneTendo25/musubi-tuner.git",
    [string]$Branch = "ltx-2-dev",
    [string]$RepoDir = "",
    [ValidateSet("cu124", "cu128", "cu130", "cpu")]
    [string]$Cuda = "cu128",
    [ValidateSet("3.10", "3.11", "3.12", "3.13")]
    [string]$PythonVersion = "3.12",
    [int]$Port = 7860,
    [string]$DashboardHost = "127.0.0.1",
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
        [Parameter(Mandatory = $true)][string]$BranchName
    )

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
        return
    }

    if (-not (Test-Path (Join-Path $RepoPath ".git"))) {
        throw "Repo directory exists but is not a git repository: $RepoPath"
    }

    Write-Log "Updating existing repository at $RepoPath..."
    $statusOut = & $GitExe -C $RepoPath status --porcelain
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to inspect git status in $RepoPath"
    }
    if ($statusOut) {
        Write-Log "Repository has local changes. Skipping update to avoid conflicts." "WARN"
        return
    }

    Invoke-ExternalWithRetry -FilePath $GitExe -Arguments @("-C", $RepoPath, "fetch", "origin", $BranchName) -MaxAttempts 3 -DelaySeconds 5
    Invoke-External -FilePath $GitExe -Arguments @("-C", $RepoPath, "checkout", $BranchName)
    Invoke-ExternalWithRetry -FilePath $GitExe -Arguments @("-C", $RepoPath, "pull", "--ff-only", "origin", $BranchName) -MaxAttempts 3 -DelaySeconds 5
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

    $launcherPath = Join-Path $RepoPath "launch_musubi_dashboard.cmd"
    $contents = @(
        "@echo off",
        "setlocal",
        "set ""REPO_DIR=%~dp0""",
        "set ""PYTHONPATH=%REPO_DIR%src;%PYTHONPATH%""",
        "set ""VENV_PY=%REPO_DIR%venv\Scripts\python.exe""",
        "if not exist ""%VENV_PY%"" (",
        "  echo Virtual environment python not found: %VENV_PY%",
        "  pause",
        "  exit /b 1",
        ")",
        """%VENV_PY%"" -m musubi_tuner.gui_dashboard --host $HostValue --port $PortValue",
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
        [Parameter(Mandatory = $true)][string]$ShortcutTarget,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    $desktop = [Environment]::GetFolderPath("Desktop")
    $shortcutPath = Join-Path $desktop "Musubi Tuner Dashboard.lnk"

    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $ShortcutTarget
    $shortcut.WorkingDirectory = $WorkingDirectory
    $shortcut.Description = "Launch Musubi Tuner Dashboard"
    $shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,13"
    $shortcut.Save()

    Write-Log "Desktop shortcut created: $shortcutPath" "OK"
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

function Select-Actions {
    param(
        [Parameter(Mandatory = $true)][System.Collections.ArrayList]$Actions,
        [Parameter(Mandatory = $true)]$InitialScan,
        [Parameter(Mandatory = $true)][string]$RepositoryPath,
        [Parameter(Mandatory = $true)][string]$VenvPath,
        [switch]$SkipPrompt
    )

    Update-ActionConstraints -Actions $Actions -InitialScan $InitialScan -RepositoryPath $RepositoryPath -VenvPath $VenvPath

    if ($SkipPrompt) {
        return
    }

    while ($true) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host " Musubi LTX-2 Installer - Action Selection" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan
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

    Write-Log "Installer log file: $script:LogFile"
    Write-Log "Install root: $InstallRoot"
    Write-Log "Repository path: $RepoDir"
    Write-Log "Repository URL: $RepoUrl"
    Write-Log "Branch: $Branch"
    Write-Log "CUDA target: $Cuda"
    Write-Log "Preferred Python: $PythonVersion"
    Write-Log "Dashboard host: $DashboardHost"
    Write-Log ("Browser URL: {0}" -f (Get-BrowserUrl -HostValue $DashboardHost -PortValue $Port))
    Ensure-Tls12ForLegacyPowerShell

    $initialScan = Get-InitialEnvironmentScan -PreferredPythonVersion $PythonVersion
    Show-InitialEnvironmentScan -Scan $initialScan

    $defaultInstallGit = -not $initialScan.GitInstalled
    $defaultInstallPython = -not $initialScan.PythonAnySupported
    $defaultInstallNode = -not $initialScan.NodeInstalled

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
            Key = "CloneOrUpdate"; Label = "Clone/update repository ($Branch branch)"; Selected = $true
            BaseReason = "required for setup"
            Reason = ""
            Locked = $false
        })
    [void]$actions.Add([pscustomobject]@{
            Key = "CreateVenv"; Label = "Create/reuse virtual environment"; Selected = $true
            BaseReason = "required for isolated install"
            Reason = ""
            Locked = $false
        })
    [void]$actions.Add([pscustomobject]@{
            Key = "InstallDeps"; Label = "Install Python dependencies (torch + musubi + dashboard)"; Selected = $true
            BaseReason = "required to run dashboard"
            Reason = ""
            Locked = $false
        })
    [void]$actions.Add([pscustomobject]@{
            Key = "BuildFrontend"; Label = "Build dashboard frontend (npm run build)"; Selected = $false
            BaseReason = "optional if frontend is already bundled"
            Reason = ""
            Locked = $false
        })
    [void]$actions.Add([pscustomobject]@{
            Key = "CreateShortcut"; Label = "Create desktop shortcut"; Selected = $true
            BaseReason = "recommended"
            Reason = ""
            Locked = $false
        })
    [void]$actions.Add([pscustomobject]@{
            Key = "LaunchDashboard"; Label = "Launch dashboard after setup"; Selected = $false
            BaseReason = "optional"
            Reason = ""
            Locked = $false
        })

    Select-Actions -Actions $actions -InitialScan $initialScan -RepositoryPath $RepoDir -VenvPath $VenvDir -SkipPrompt:$NonInteractive

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
    if (Get-ActionState -Actions $actions -Key "CloneOrUpdate") {
        if (-not $gitExeObj) {
            throw "git is required for clone/update. Enable 'Install Git' or install git manually."
        }
        Invoke-Step -Name "SyncRepository" -Action {
            Sync-Repository -GitExe $gitExeObj.Source -RepoPath $RepoDir -RemoteUrl $RepoUrl -BranchName $Branch
        }
    } elseif (-not (Test-Path $RepoDir)) {
        throw "Repository directory does not exist: $RepoDir"
    }

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
    }

    if (Get-ActionState -Actions $actions -Key "BuildFrontend") {
        if (-not (Test-Path $RepoDir)) {
            throw "Repository directory does not exist: $RepoDir"
        }
        if (-not (Test-AnyCommand @("npm.cmd", "npm"))) {
            throw "Node.js/npm not found. Enable 'Install Node.js' or disable frontend build."
        }
        Invoke-Step -Name "BuildFrontend" -Action { Build-Frontend -RepoPath $RepoDir }
    }

    $launcherPath = $null
    Invoke-Step -Name "WriteLauncherScript" -Action {
        $script:launcherPathInternal = Write-LauncherScript -RepoPath $RepoDir -HostValue $DashboardHost -PortValue $Port
    }
    $launcherPath = $script:launcherPathInternal
    Write-Log "Launcher script ready: $launcherPath" "OK"

    if (Get-ActionState -Actions $actions -Key "CreateShortcut") {
        Invoke-Step -Name "CreateDesktopShortcut" -Action {
            Create-DesktopShortcut -ShortcutTarget $launcherPath -WorkingDirectory $RepoDir
        }
    }

    if (Get-ActionState -Actions $actions -Key "LaunchDashboard") {
        Invoke-Step -Name "LaunchDashboard" -Action {
            Write-Log "Launching dashboard..."
            Start-Process -FilePath $launcherPath -WorkingDirectory $RepoDir
        }
    }

    Write-Log "Installation completed successfully." "OK"
    Write-Log "You can launch the dashboard with: $launcherPath"
    Write-Log ("Open in your browser: {0}" -f (Get-BrowserUrl -HostValue $DashboardHost -PortValue $Port)) "OK"
    Write-SupportBundle -Outcome "success"
} catch {
    Write-Log ("Installation failed: {0}" -f $_.Exception.Message) "ERROR"
    Write-Log "See full log: $script:LogFile" "ERROR"
    Write-SupportBundle -Outcome "failure" -ErrorRecord $_
    throw
}
