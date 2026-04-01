# Security Fixes Batch 2

## 概述

本次安全修复批次解决了 **5 个关键安全漏洞**，包括 SSL 验证绕过、SSRF、缓冲区溢出和反序列化漏洞。

## 修复列表

### 1. ✅ SSL 验证完全禁用 (CRITICAL)
- **文件**: `models/model.py`
- **问题**: 代码使用 `ssl._create_default_https_context = ssl._create_unverified_context` 完全禁用了 SSL 证书验证
- **风险**: 允许中间人攻击，攻击者可拦截和篡改模型下载流量
- **修复**: 移除了危险的 SSL 禁用代码，恢复默认安全验证
- **Commit**: `8a1c783`

### 2. ✅ SSRF 漏洞 (CRITICAL)
- **文件**: `tools/dataset_collector/dataset_maker.py`
- **问题**: 图片下载功能未验证 URL，可请求内网资源（169.254.169.254 等）
- **风险**: 服务器端请求伪造，可能泄露敏感 metadata、攻击内部服务
- **修复**: 
  - 新增 `validate_url()` 函数，检测并阻止私有 IP 地址
  - 在所有 `urllib.request` 调用前添加 URL 验证
  - 移除了不安全的 `ssl._create_unverified_context()` 使用
- **Commit**: `ee24d21`

### 3. ✅ OpenCV WebP 缓冲区溢出 (CVE-2023-4863)
- **文件**: `requirements.txt`
- **问题**: OpenCV < 4.8.1.78 使用的 libwebp 存在堆缓冲区溢出
- **风险**: 处理恶意 WebP 图片时可导致任意代码执行
- **修复**: 将 `opencv-python` 从 `>=4.5.0` 升级到 `>=4.8.1.78`
- **Commit**: `ded6509`

### 4. ✅ 图像尺寸限制缺失 (HIGH)
- **文件**: `dataset/dataloader.py`
- **问题**: 加载图片时无尺寸限制，超大图片可导致内存耗尽
- **风险**: 拒绝服务攻击 (DoS)，系统 OOM 崩溃
- **修复**:
  - 新增 `MAX_IMAGE_SIZE = 100_000_000` (100MP) 常量
  - 在 `__getitem__` 中添加尺寸检查，超限图像抛出异常
- **Commit**: `7d8c993`

### 5. ✅ PyTorch 不安全反序列化 (CRITICAL)
- **文件**: `libs/training.py`
- **问题**: `torch.load()` 缺少 `weights_only=True`，使用 pickle 反序列化
- **风险**: 恶意 checkpoint 文件可执行任意 Python 代码
- **修复**: 为所有 `torch.load()` 调用添加 `weights_only=True`
- **Commit**: `f0a3fb6`

## 命令注入检查

**文件**: `build_dataset_bundle.py`

经检查，该文件使用的 `subprocess.run()` 已采用安全格式（无 `shell=True`），无需修复。

## 验证

### 代码扫描确认
```bash
# SSL 验证漏洞 - 已修复
grep -n "_create_unverified_context" models/model.py
# 无输出 (已移除)

# SSRF 防护 - 已添加
grep -n "validate_url" tools/dataset_collector/dataset_maker.py
# 输出: 多行匹配 (已添加)

# PyTorch weights_only - 已确认
grep -n "torch.load" libs/training.py
# 输出: 包含 weights_only=True
```

### 依赖版本确认
```bash
# OpenCV 版本要求
grep "opencv-python" requirements.txt
# 输出: opencv-python>=4.8.1.78
```

## 提交记录

```
f0a3fb6 Security: Fix PyTorch unsafe deserialization
7d8c993 Security: Add image size limits to prevent DoS
ded6509 Security: Fix OpenCV WebP buffer overflow (CVE-2023-4863)
ee24d21 Security: Fix SSRF vulnerability in image downloader
8a1c783 Security: Fix SSL verification disabled vulnerability
```

## 建议

1. **立即升级依赖**: 运行 `pip install --upgrade opencv-python>=4.8.1.78`
2. **审查自定义 checkpoint**: 验证所有加载的模型文件来源可信
3. **网络隔离**: 对于图片下载功能，考虑在网络层限制出站连接
4. **持续监控**: 启用 Dependabot 等工具自动检测新漏洞

---
*修复日期*: 2026-04-01  
*分支*: `security-fixes-batch2`
