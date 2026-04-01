"""
Cloud Configuration Module

This module provides secure access to cloud service credentials
using environment variables. Never hardcode credentials in code.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class AWSConfig:
    """AWS Configuration"""
    access_key_id: str
    secret_access_key: str
    region: str
    s3_bucket: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> Optional['AWSConfig']:
        """Load AWS config from environment variables"""
        access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        region = os.environ.get('AWS_REGION', 'us-east-1')
        
        if not access_key or not secret_key:
            return None
            
        return cls(
            access_key_id=access_key,
            secret_access_key=secret_key,
            region=region,
            s3_bucket=os.environ.get('AWS_S3_BUCKET')
        )


@dataclass
class AzureConfig:
    """Azure Configuration"""
    storage_account_name: str
    storage_account_key: str
    container_name: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> Optional['AzureConfig']:
        """Load Azure config from environment variables"""
        account_name = os.environ.get('AZURE_STORAGE_ACCOUNT_NAME')
        account_key = os.environ.get('AZURE_STORAGE_ACCOUNT_KEY')
        
        if not account_name or not account_key:
            return None
            
        return cls(
            storage_account_name=account_name,
            storage_account_key=account_key,
            container_name=os.environ.get('AZURE_CONTAINER_NAME')
        )


@dataclass
class GCPConfig:
    """Google Cloud Platform Configuration"""
    project_id: str
    private_key_id: str
    private_key: str
    client_email: str
    storage_bucket: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> Optional['GCPConfig']:
        """Load GCP config from environment variables"""
        project_id = os.environ.get('GCP_PROJECT_ID')
        private_key_id = os.environ.get('GCP_PRIVATE_KEY_ID')
        private_key = os.environ.get('GCP_PRIVATE_KEY')
        client_email = os.environ.get('GCP_CLIENT_EMAIL')
        
        if not all([project_id, private_key_id, private_key, client_email]):
            return None
            
        return cls(
            project_id=project_id,
            private_key_id=private_key_id,
            private_key=private_key,
            client_email=client_email,
            storage_bucket=os.environ.get('GCP_STORAGE_BUCKET')
        )


@dataclass
class AliyunConfig:
    """Alibaba Cloud Configuration"""
    access_key_id: str
    access_key_secret: str
    oss_bucket: Optional[str] = None
    region: str = 'cn-hangzhou'
    
    @classmethod
    def from_env(cls) -> Optional['AliyunConfig']:
        """Load Aliyun config from environment variables"""
        access_key = os.environ.get('ALIYUN_ACCESS_KEY_ID')
        access_secret = os.environ.get('ALIYUN_ACCESS_KEY_SECRET')
        
        if not access_key or not access_secret:
            return None
            
        return cls(
            access_key_id=access_key,
            access_key_secret=access_secret,
            oss_bucket=os.environ.get('ALIYUN_OSS_BUCKET'),
            region=os.environ.get('ALIYUN_REGION', 'cn-hangzhou')
        )


@dataclass
class CloudConfig:
    """Unified Cloud Configuration"""
    aws: Optional[AWSConfig] = None
    azure: Optional[AzureConfig] = None
    gcp: Optional[GCPConfig] = None
    aliyun: Optional[AliyunConfig] = None
    
    @classmethod
    def from_env(cls) -> 'CloudConfig':
        """Load all cloud configs from environment variables"""
        return cls(
            aws=AWSConfig.from_env(),
            azure=AzureConfig.from_env(),
            gcp=GCPConfig.from_env(),
            aliyun=AliyunConfig.from_env()
        )
    
    def is_configured(self, provider: str) -> bool:
        """Check if a specific cloud provider is configured"""
        provider_map = {
            'aws': self.aws,
            'azure': self.azure,
            'gcp': self.gcp,
            'aliyun': self.aliyun
        }
        return provider_map.get(provider.lower()) is not None
    
    def get_available_providers(self) -> list:
        """Get list of configured cloud providers"""
        providers = []
        if self.aws:
            providers.append('aws')
        if self.azure:
            providers.append('azure')
        if self.gcp:
            providers.append('gcp')
        if self.aliyun:
            providers.append('aliyun')
        return providers


# Global instance
cloud_config = CloudConfig.from_env()
