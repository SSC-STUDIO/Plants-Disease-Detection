#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型下载服务 - 不安全版本（演示漏洞）
WARNING: 此文件仅用于演示漏洞，不要在生产环境使用！

漏洞说明:
1. DEBUG模式开启 - 会泄露敏感信息
2. 未授权下载 - 任何人都可以下载模型文件
3. 路径遍历漏洞 - 可能访问系统任意文件
"""

import os
from flask import Flask, send_file, jsonify

app = Flask(__name__)

# VULNERABILITY 1: DEBUG模式开启 - 会泄露敏感信息
DEBUG = True

# 模型文件存储目录
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({'status': 'ok'})


@app.route('/models', methods=['GET'])
def list_models():
    """
    列出可用的模型文件
    VULNERABILITY: 无需认证即可访问
    """
    models = []
    if os.path.exists(MODEL_DIR):
        for root, dirs, files in os.walk(MODEL_DIR):
            for file in files:
                if file.endswith(('.pth', '.pt', '.pth.tar', '.onnx')):
                    rel_path = os.path.relpath(os.path.join(root, file), MODEL_DIR)
                    models.append({'name': file, 'path': rel_path})
    return jsonify({'models': models})


@app.route('/download/<path:model_path>', methods=['GET'])
def download_model(model_path):
    """
    下载模型文件
    VULNERABILITY:
    1. 无需任何认证
    2. 没有路径遍历防护 - 可以访问 ../../../etc/passwd
    3. 没有文件类型验证
    """
    # VULNERABILITY: 直接使用用户输入的路径，没有验证
    file_path = os.path.join(MODEL_DIR, model_path)
    
    # VULNERABILITY: 仅检查文件是否存在，没有检查路径是否在允许范围内
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    # VULNERABILITY: 没有日志记录，无法审计
    
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    # VULNERABILITY: DEBUG模式开启，会显示详细的错误信息和调试信息
    app.run(host='0.0.0.0', port=5000, debug=DEBUG)
