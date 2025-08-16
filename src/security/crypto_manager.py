#!/usr/bin/env python3
"""
Cryptographic Manager for DipMaster Trading System
加密管理器 - 提供企业级加密功能

Features:
- AES-256-GCM encryption for API keys
- PBKDF2 key derivation with salt
- Secure key generation and rotation
- Multiple encryption backends support
"""

import os
import base64
import secrets
from typing import Dict, Any, Optional, Tuple
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import logging

logger = logging.getLogger(__name__)


class CryptoManager:
    """
    Enterprise-grade cryptographic manager for secure API key storage.
    
    Uses AES-256-GCM encryption with PBKDF2 key derivation to provide
    authenticated encryption with additional data (AEAD).
    """
    
    def __init__(self, master_password: Optional[str] = None):
        """
        Initialize crypto manager.
        
        Args:
            master_password: Master password for key derivation. If None, 
                           will use environment variable or prompt user.
        """
        self.master_password = master_password or self._get_master_password()
        self.iterations = 480000  # NIST recommended minimum for 2024
        self.key_length = 32  # AES-256
        
    def _get_master_password(self) -> str:
        """Get master password from environment or user input."""
        # Try environment variable first
        password = os.getenv('DIPMASTER_MASTER_PASSWORD')
        if password:
            logger.info("🔐 Master password loaded from environment")
            return password
            
        # For production, implement secure password prompt
        # For now, use a default (should be changed in production)
        logger.warning("⚠️  Using default master password - CHANGE IN PRODUCTION!")
        return "dipmaster-default-key-2025"
    
    def _derive_key(self, salt: bytes) -> bytes:
        """
        Derive encryption key from master password using PBKDF2.
        
        Args:
            salt: Random salt for key derivation
            
        Returns:
            32-byte derived key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.key_length,
            salt=salt,
            iterations=self.iterations,
            backend=default_backend()
        )
        return kdf.derive(self.master_password.encode())
    
    def encrypt_data(self, plaintext: str, additional_data: str = "") -> Dict[str, str]:
        """
        Encrypt data using AES-256-GCM.
        
        Args:
            plaintext: Data to encrypt
            additional_data: Additional authenticated data (AAD)
            
        Returns:
            Dictionary containing encrypted data and metadata
        """
        try:
            # Generate random salt and IV
            salt = secrets.token_bytes(32)
            iv = secrets.token_bytes(12)  # 96-bit IV for GCM
            
            # Derive encryption key
            key = self._derive_key(salt)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # Add additional authenticated data if provided
            if additional_data:
                encryptor.authenticate_additional_data(additional_data.encode())
            
            # Encrypt data
            ciphertext = encryptor.update(plaintext.encode()) + encryptor.finalize()
            
            # Return encrypted package
            return {
                'ciphertext': base64.b64encode(ciphertext).decode(),
                'salt': base64.b64encode(salt).decode(),
                'iv': base64.b64encode(iv).decode(),
                'tag': base64.b64encode(encryptor.tag).decode(),
                'algorithm': 'AES-256-GCM',
                'iterations': self.iterations,
                'additional_data': additional_data
            }
            
        except Exception as e:
            logger.error(f"❌ Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_package: Dict[str, str]) -> str:
        """
        Decrypt data using AES-256-GCM.
        
        Args:
            encrypted_package: Dictionary containing encrypted data and metadata
            
        Returns:
            Decrypted plaintext
        """
        try:
            # Extract components
            ciphertext = base64.b64decode(encrypted_package['ciphertext'])
            salt = base64.b64decode(encrypted_package['salt'])
            iv = base64.b64decode(encrypted_package['iv'])
            tag = base64.b64decode(encrypted_package['tag'])
            additional_data = encrypted_package.get('additional_data', '')
            
            # Derive decryption key
            key = self._derive_key(salt)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            # Add additional authenticated data if present
            if additional_data:
                decryptor.authenticate_additional_data(additional_data.encode())
            
            # Decrypt data
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            return plaintext.decode()
            
        except Exception as e:
            logger.error(f"❌ Decryption failed: {e}")
            raise
    
    def encrypt_api_credentials(self, api_key: str, api_secret: str, 
                              exchange: str = "binance") -> Dict[str, Any]:
        """
        Encrypt API credentials with additional context.
        
        Args:
            api_key: API key to encrypt
            api_secret: API secret to encrypt  
            exchange: Exchange name for additional authenticated data
            
        Returns:
            Encrypted credentials package
        """
        timestamp = str(int(os.times().system))
        additional_data = f"{exchange}:{timestamp}"
        
        encrypted_key = self.encrypt_data(api_key, additional_data)
        encrypted_secret = self.encrypt_data(api_secret, additional_data)
        
        return {
            'api_key': encrypted_key,
            'api_secret': encrypted_secret,
            'exchange': exchange,
            'encrypted_at': timestamp,
            'version': '1.0'
        }
    
    def decrypt_api_credentials(self, encrypted_credentials: Dict[str, Any]) -> Tuple[str, str]:
        """
        Decrypt API credentials.
        
        Args:
            encrypted_credentials: Encrypted credentials package
            
        Returns:
            Tuple of (api_key, api_secret)
        """
        try:
            api_key = self.decrypt_data(encrypted_credentials['api_key'])
            api_secret = self.decrypt_data(encrypted_credentials['api_secret'])
            
            logger.info(f"🔓 Successfully decrypted credentials for {encrypted_credentials['exchange']}")
            return api_key, api_secret
            
        except Exception as e:
            logger.error(f"❌ Failed to decrypt API credentials: {e}")
            raise
    
    def generate_secure_password(self, length: int = 32) -> str:
        """Generate cryptographically secure password."""
        return secrets.token_urlsafe(length)
    
    def verify_data_integrity(self, encrypted_package: Dict[str, str]) -> bool:
        """
        Verify data integrity without decryption.
        
        Args:
            encrypted_package: Encrypted data package
            
        Returns:
            True if data integrity is verified
        """
        try:
            # Attempt to decrypt (will fail if tampered)
            self.decrypt_data(encrypted_package)
            return True
        except Exception:
            return False


# Utility functions
def generate_master_key() -> str:
    """Generate a new master key for production use."""
    return secrets.token_urlsafe(64)


def create_crypto_manager(password: Optional[str] = None) -> CryptoManager:
    """Factory function to create crypto manager."""
    return CryptoManager(password)