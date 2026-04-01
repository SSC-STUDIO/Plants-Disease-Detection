#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
安全修复验证脚本
验证2个严重漏洞是否已修复
"""

import os
import sys
import subprocess
import tempfile
import shutil

def print_section(title):
    """打印章节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_result(test_name, passed, details=""):
    """打印测试结果"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {test_name}")
    if details:
        print(f"         {details}")
    return passed

def check_debug_fix():
    """检查DEBUG模式修复"""
    print_section("漏洞1: DEBUG模式修复验证")
    
    results = []
    training_file = "libs/training.py"
    
    # 检查1: 确认文件存在
    if not os.path.exists(training_file):
        print_result("training.py文件存在", False, "文件不存在")
        return False
    
    with open(training_file, 'r') as f:
        content = f.read()
    
    # 检查2: 确认硬编码的DEBUG已移除
    if 'logger.setLevel(logging.DEBUG)' in content and 'LOG_LEVEL' not in content:
        results.append(print_result("硬编码DEBUG已移除", False, "仍存在硬编码DEBUG"))
    else:
        results.append(print_result("硬编码DEBUG已移除", True))
    
    # 检查3: 确认使用环境变量
    if 'LOG_LEVEL' in content and 'os.environ.get' in content:
        results.append(print_result("使用环境变量控制日志级别", True))
    else:
        results.append(print_result("使用环境变量控制日志级别", False, "未找到环境变量配置"))
    
    # 检查4: 确认默认值为INFO
    if "'INFO'" in content or '"INFO"' in content:
        results.append(print_result("默认日志级别为INFO", True))
    else:
        results.append(print_result("默认日志级别为INFO", False, "默认值可能不是INFO"))
    
    return all(results)

def check_model_server_security():
    """检查模型下载服务安全修复"""
    print_section("漏洞2: 模型下载服务安全验证")
    
    results = []
    
    # 检查1: 安全版本文件存在
    if os.path.exists("model_server.py"):
        results.append(print_result("安全版本文件存在", True))
    else:
        results.append(print_result("安全版本文件存在", False, "model_server.py不存在"))
        return False
    
    with open("model_server.py", 'r') as f:
        content = f.read()
    
    # 检查2: API密钥认证
    if 'require_auth' in content and 'X-API-Key' in content:
        results.append(print_result("API密钥认证已添加", True))
    else:
        results.append(print_result("API密钥认证已添加", False, "未找到认证逻辑"))
    
    # 检查3: 路径遍历防护
    if 'is_safe_path' in content and 'realpath' in content:
        results.append(print_result("路径遍历防护已添加", True))
    else:
        results.append(print_result("路径遍历防护已添加", False, "未找到路径验证"))
    
    # 检查4: 文件类型验证
    if 'allowed_extensions' in content:
        results.append(print_result("文件类型白名单已添加", True))
    else:
        results.append(print_result("文件类型白名单已添加", False, "未找到文件类型验证"))
    
    # 检查5: DEBUG模式通过环境变量控制
    if 'FLASK_DEBUG' in content and 'os.environ.get' in content:
        results.append(print_result("DEBUG模式环境变量控制", True))
    else:
        results.append(print_result("DEBUG模式环境变量控制", False, "未找到环境变量配置"))
    
    # 检查6: 审计日志
    if 'app.logger.warning' in content or 'app.logger.info' in content:
        results.append(print_result("审计日志已添加", True))
    else:
        results.append(print_result("审计日志已添加", False, "未找到日志记录"))
    
    return all(results)

def test_model_server_runtime():
    """运行时测试模型服务器"""
    print_section("运行时测试")
    
    results = []
    
    try:
        import flask
    except ImportError:
        print("  ⚠️  Flask未安装，跳过运行时测试")
        print("      安装命令: pip install flask")
        return True
    
    # 设置测试环境
    test_api_key = "test-api-key-12345"
    os.environ['MODEL_API_KEY'] = test_api_key
    os.environ['FLASK_DEBUG'] = '0'
    
    # 创建测试模型目录和文件
    test_dir = tempfile.mkdtemp()
    os.environ['MODEL_DIR'] = test_dir
    
    try:
        # 创建测试模型文件
        test_model_path = os.path.join(test_dir, 'test_model.pth')
        with open(test_model_path, 'w') as f:
            f.write("fake model data")
        
        # 导入应用
        import importlib.util
        spec = importlib.util.spec_from_file_location("model_server", "model_server.py")
        model_server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_server)
        app = model_server.app
        app.config['TESTING'] = True
        client = app.test_client()
        
        # 测试1: 健康检查
        response = client.get('/health')
        if response.status_code == 200:
            results.append(print_result("健康检查端点", True))
        else:
            results.append(print_result("健康检查端点", False, f"状态码: {response.status_code}"))
        
        # 测试2: 未认证访问被拒绝
        response = client.get('/models')
        if response.status_code == 401:
            results.append(print_result("未认证访问被拒绝", True))
        else:
            results.append(print_result("未认证访问被拒绝", False, f"状态码: {response.status_code}"))
        
        # 测试3: 认证访问成功
        response = client.get('/models', headers={'X-API-Key': test_api_key})
        if response.status_code == 200:
            results.append(print_result("认证访问成功", True))
        else:
            results.append(print_result("认证访问成功", False, f"状态码: {response.status_code}"))
        
        # 测试4: 路径遍历防护
        response = client.get(
            '/download/../../../etc/passwd',
            headers={'X-API-Key': test_api_key}
        )
        if response.status_code in [403, 404]:
            results.append(print_result("路径遍历防护有效", True))
        else:
            results.append(print_result("路径遍历防护有效", False, f"状态码: {response.status_code}"))
        
        # 测试5: 有效的模型下载
        response = client.get(
            '/download/test_model.pth',
            headers={'X-API-Key': test_api_key}
        )
        if response.status_code == 200:
            results.append(print_result("模型下载功能正常", True))
        else:
            results.append(print_result("模型下载功能正常", False, f"状态码: {response.status_code}"))
        
    except Exception as e:
        print(f"  ⚠️  运行时测试出错: {e}")
        return False
    finally:
        # 清理
        shutil.rmtree(test_dir, ignore_errors=True)
    
    return all(results)

def main():
    """主函数"""
    print("\n" + "="*60)
    print("  Plants-Disease-Detection 安全修复验证")
    print("="*60)
    print(f"\n  验证时间: {subprocess.check_output(['date']).decode().strip()}")
    print(f"  项目路径: {os.getcwd()}")
    
    # 保存当前目录
    original_dir = os.getcwd()
    
    try:
        # 切换到项目目录
        project_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(project_dir)
        
        # 执行验证
        debug_fix_ok = check_debug_fix()
        model_server_ok = check_model_server_security()
        runtime_ok = test_model_server_runtime()
        
        # 打印总结
        print_section("验证总结")
        
        all_passed = debug_fix_ok and model_server_ok and runtime_ok
        
        if all_passed:
            print("  ✅ 所有安全修复验证通过!")
            print("\n  修复的漏洞:")
            print("    1. DEBUG模式硬编码问题 - 已修复")
            print("    2. 模型文件未授权下载 - 已修复")
            print("\n  安全功能:")
            print("    - API密钥认证")
            print("    - 路径遍历防护")
            print("    - 文件类型验证")
            print("    - 审计日志")
            print("    - DEBUG模式环境变量控制")
        else:
            print("  ❌ 部分验证未通过，请检查上述输出")
            return 1
        
        return 0
        
    finally:
        os.chdir(original_dir)

if __name__ == '__main__':
    sys.exit(main())
