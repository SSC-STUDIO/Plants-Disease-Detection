# run_training.ps1 - 植物病害检测训练入口脚本
# 用法:
#   .\run_training.ps1                    # 使用默认配置训练
#   .\run_training.ps1 -OfflineWeights    # 离线模式，不下载预训练权重
#   .\run_training.ps1 -ForceTrain        # 强制从头训练
#   .\run_training.ps1 -Epochs 100        # 指定训练轮数

param(
    [switch]$OfflineWeights,    # 离线模式：禁用预训练权重下载
    [switch]$ForceTrain,        # 强制从头训练：忽略已有检查点
    [int]$Epochs = 0,           # 训练轮数（0表示使用配置文件中的值）
    [string]$Model = "",        # 模型名称（空表示使用配置文件中的值）
    [int]$BatchSize = 0,        # 批次大小（0表示使用配置文件中的值）
    [string]$DataDir = "",      # 数据目录（空表示使用配置文件中的值）
    [switch]$Help              # 显示帮助信息
)

# 显示帮助
if ($Help) {
    Write-Host @"
植物病害检测训练脚本

用法:
    .\run_training.ps1 [选项]

选项:
    -OfflineWeights    离线模式：禁用预训练权重下载，适合网络受限环境
    -ForceTrain        强制从头训练：忽略已有的检查点文件
    -Epochs <n>        指定训练轮数（默认使用配置文件中的值）
    -Model <name>      指定模型名称（如 convnextv2_base_384, efficientnetv2_s）
    -BatchSize <n>     指定批次大小
    -DataDir <path>    指定数据目录
    -Help               显示此帮助信息

示例:
    .\run_training.ps1
    .\run_training.ps1 -OfflineWeights -Epochs 50
    .\run_training.ps1 -Model efficientnetv2_s -BatchSize 16
"@
    exit 0
}

# 设置错误处理
$ErrorActionPreference = "Stop"

# 获取脚本所在目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# 检查 Python 环境
Write-Host "检查 Python 环境..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python 版本: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "错误: 未找到 Python，请确保 Python 已安装并添加到 PATH" -ForegroundColor Red
    exit 1
}

# 检查必要的依赖
Write-Host "检查依赖包..." -ForegroundColor Cyan
$requiredPackages = @("torch", "torchvision", "timm", "PIL")
$missingPackages = @()

foreach ($pkg in $requiredPackages) {
    $check = python -c "import $pkg; print('ok')" 2>&1
    if ($check -ne "ok") {
        $missingPackages += $pkg
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host "警告: 以下依赖包未安装: $($missingPackages -join ', ')" -ForegroundColor Yellow
    Write-Host "正在尝试安装..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# 设置环境变量
Write-Host "设置环境变量..." -ForegroundColor Cyan

# 离线模式：设置环境变量阻止网络请求
if ($OfflineWeights) {
    Write-Host "离线模式已启用，将使用本地权重或随机初始化" -ForegroundColor Yellow
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    $env:TORCH_HOME = Join-Path $ScriptDir "checkpoints\pretrained"
    $env:HF_HOME = Join-Path $ScriptDir "checkpoints\pretrained"

    # 临时修改配置以禁用预训练
    $env:PLANT_DISEASE_PRETRAINED = "0"
} else {
    # 清除离线环境变量
    $env:HF_HUB_OFFLINE = $null
    $env:TRANSFORMERS_OFFLINE = $null
    $env:PLANT_DISEASE_PRETRAINED = "1"
}

# CUDA 设置
$env:CUDA_LAUNCH_BLOCKING = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# 构建训练命令
# main.py 使用子命令格式：python main.py train [options]
$trainArgs = @("train")

if ($ForceTrain) {
    $trainArgs += "--force-train"
}

if ($Epochs -gt 0) {
    $trainArgs += "--epochs"
    $trainArgs += $Epochs
}

if ($Model -ne "") {
    $trainArgs += "--model"
    $trainArgs += $Model
}

if ($BatchSize -gt 0) {
    $trainArgs += "--batch-size"
    $trainArgs += $BatchSize
}

if ($DataDir -ne "") {
    $trainArgs += "--dataset-path"
    $trainArgs += $DataDir
}

# 显示训练配置
Write-Host @"
========================================
训练配置
========================================
离线模式: $($OfflineWeights.IsPresent)
强制训练: $($ForceTrain.IsPresent)
训练轮数: $(if ($Epochs -gt 0) { $Epochs } else { "使用配置文件默认值" })
模型: $(if ($Model -ne "") { $Model } else { "使用配置文件默认值" })
批次大小: $(if ($BatchSize -gt 0) { $BatchSize } else { "使用配置文件默认值" })
数据目录: $(if ($DataDir -ne "") { $DataDir } else { "使用配置文件默认值" })
========================================
"@ -ForegroundColor Cyan

# 创建必要的目录
$directories = @("checkpoints", "checkpoints\best", "log", "submit")
foreach ($dir in $directories) {
    $dirPath = Join-Path $ScriptDir $dir
    if (-not (Test-Path $dirPath)) {
        New-Item -ItemType Directory -Path $dirPath -Force | Out-Null
        Write-Host "创建目录: $dirPath" -ForegroundColor Green
    }
}

# 运行训练
Write-Host "开始训练..." -ForegroundColor Green
Write-Host "命令: python main.py $($trainArgs -join ' ')" -ForegroundColor Cyan

try {
    & python main.py @trainArgs
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        Write-Host "`n训练完成!" -ForegroundColor Green
    } else {
        Write-Host "`n训练异常退出，退出码: $exitCode" -ForegroundColor Red
    }
} catch {
    Write-Host "`n训练过程中发生错误: $_" -ForegroundColor Red
    exit 1
}

# 清理环境变量
$env:HF_HUB_OFFLINE = $null
$env:TRANSFORMERS_OFFLINE = $null
$env:PLANT_DISEASE_PRETRAINED = $null

exit $exitCode
