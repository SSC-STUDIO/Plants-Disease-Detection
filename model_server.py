#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型下载服务 - 安全修复版
修复: 添加权限验证，只允许授权用户下载模型文件
"""

import os
import functools
from flask import Flask, send_file, request, jsonify, abort

app = Flask(__name__)

# 安全配置
# SECURITY FIX: 使用环境变量存储API密钥，不要硬编码
API_KEY = os.environ.get('MODEL_API_KEY', 'your-secret-api-key-here')
# SECURITY FIX: 生产环境必须设置为False
DEBUG_MODE = os.environ.get('FLASK_DEBUG', '0') == '1'

# 模型文件存储目录
MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints'))


def require_auth(f):
    """
    装饰器: 验证API密钥
    SECURITY FIX: 添加权限验证，只允许授权用户下载
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        # 从请求头获取API密钥
        auth_header = request.headers.get('X-API-Key')
        
        if not auth_header:
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Missing API key. Please provide X-API-Key header.'
            }), 401
        
        # 验证API密钥
        if auth_header != API_KEY:
            # SECURITY FIX: 记录失败的认证尝试
            app.logger.warning(f'Invalid API key attempt from {request.remote_addr}')
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Invalid API key.'
            }), 401
        
        return f(*args, **kwargs)
    return decorated_function


def is_safe_path(base_path, user_path):
    """
    验证文件路径是否安全（防止路径遍历攻击）
    SECURITY FIX: 路径遍历防护
    """
    try:
        # 规范化路径
        base = os.path.realpath(base_path)
        target = os.path.realpath(os.path.join(base, user_path))
        
        # 确保目标路径在允许的目录内
        return target.startswith(base)
    except (ValueError, OSError):
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点 - 无需认证"""
    return jsonify({'status': 'ok', 'service': 'model-download-service'})


@app.route('/models', methods=['GET'])
@require_auth
def list_models():
    """
    列出可用的模型文件
    SECURITY FIX: 需要认证
    """
    models = []
    if os.path.exists(MODEL_DIR):
        for root, dirs, files in os.walk(MODEL_DIR):
            for file in files:
                if file.endswith(('.pth', '.pt', '.pth.tar', '.onnx')):
                    rel_path = os.path.relpath(os.path.join(root, file), MODEL_DIR)
                    models.append({
                        'name': file,
                        'path': rel_path,
                        'size': os.path.getsize(os.path.join(root, file))
                    })
    return jsonify({'models': models})


@app.route('/download/<path:model_path>', methods=['GET'])
@require_auth
def download_model(model_path):
    """
    下载模型文件
    SECURITY FIX: 
    1. 需要API密钥认证
    2. 路径遍历防护
    3. 文件类型验证
    """
    # SECURITY FIX: 路径遍历防护
    if not is_safe_path(MODEL_DIR, model_path):
        app.logger.warning(f'Path traversal attempt: {model_path} from {request.remote_addr}')
        abort(403, 'Access denied: Invalid path')
    
    # 构建完整文件路径
    file_path = os.path.join(MODEL_DIR, model_path)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        abort(404, 'Model file not found')
    
    # SECURITY FIX: 验证文件类型
    allowed_extensions = ('.pth', '.pt', '.pth.tar', '.onnx', '.json')
    if not any(str(model_path).lower().endswith(ext) for ext in allowed_extensions):
        app.logger.warning(f'Invalid file type download attempt: {model_path}')
        abort(403, 'Access denied: Invalid file type')
    
    # SECURITY FIX: 记录下载日志
    app.logger.info(f'Model downloaded: {model_path} by {request.remote_addr}')
    
    # 发送文件
    return send_file(
        file_path,
        as_attachment=True,
        download_name=os.path.basename(model_path)
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(403)
def forbidden(error):
    return jsonify({'error': 'Forbidden'}), 403


@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server error: {str(error)}')
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # SECURITY FIX: 生产环境禁用DEBUG模式
    if DEBUG_MODE:
        print("WARNING: Running in DEBUG mode. Set FLASK_DEBUG=0 for production.")
    
    # 确保模型目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 启动服务
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=DEBUG_MODE  # SECURITY FIX: 从环境变量读取
    )
