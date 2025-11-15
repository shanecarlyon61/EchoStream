"""
Crypto Module - Encryption/decryption utilities

This module provides encryption and decryption functions for audio packets
using AES encryption and base64 encoding/decoding.
"""
import base64
from typing import Optional
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os


# ============================================================================
# Base64 Encoding/Decoding
# ============================================================================

def decode_base64(key_str: str) -> Optional[bytes]:
    """
    Decode base64 string to bytes.
    
    Args:
        key_str: Base64-encoded string
        
    Returns:
        Decoded bytes or None on error
    """
    try:
        decoded = base64.b64decode(key_str)
        return decoded
    except Exception as e:
        print(f"[CRYPTO] ERROR: Failed to decode base64: {e}")
        return None


def encode_base64(data: bytes) -> str:
    """
    Encode bytes to base64 string.
    
    Args:
        data: Bytes to encode
        
    Returns:
        Base64-encoded string
    """
    try:
        encoded = base64.b64encode(data).decode('utf-8')
        return encoded
    except Exception as e:
        print(f"[CRYPTO] ERROR: Failed to encode base64: {e}")
        return ""


# ============================================================================
# AES Encryption/Decryption
# ============================================================================

def encrypt_aes(data: bytes, key: bytes, iv: Optional[bytes] = None) -> Optional[bytes]:
    """
    Encrypt data using AES-256-CBC.
    
    Args:
        data: Data bytes to encrypt
        key: Encryption key (32 bytes for AES-256)
        iv: Initialization vector (16 bytes), if None, generates random IV
        
    Returns:
        Encrypted data with IV prepended, or None on error
    """
    try:
        # Ensure key is 32 bytes
        if len(key) != 32:
            if len(key) < 32:
                key = key + b'\x00' * (32 - len(key))
            else:
                key = key[:32]
        
        # Generate IV if not provided
        if iv is None:
            iv = os.urandom(16)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        
        # Pad data to block size (16 bytes)
        block_size = 16
        padding = block_size - (len(data) % block_size)
        padded_data = data + bytes([padding] * padding)
        
        # Encrypt
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        
        # Prepend IV to encrypted data
        return iv + encrypted
        
    except Exception as e:
        print(f"[CRYPTO] ERROR: AES encryption failed: {e}")
        return None


def decrypt_aes(encrypted_data: bytes, key: bytes) -> Optional[bytes]:
    """
    Decrypt data using AES-256-CBC.
    
    Args:
        encrypted_data: Encrypted data with IV prepended
        key: Decryption key (32 bytes for AES-256)
        
    Returns:
        Decrypted data bytes or None on error
    """
    try:
        # Ensure key is 32 bytes
        if len(key) != 32:
            if len(key) < 32:
                key = key + b'\x00' * (32 - len(key))
            else:
                key = key[:32]
        
        # Extract IV (first 16 bytes)
        if len(encrypted_data) < 16:
            print("[CRYPTO] ERROR: Encrypted data too short (missing IV)")
            return None
        
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        
        # Decrypt
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        padding = decrypted[-1]
        if padding > 16 or padding < 1:
            print(f"[CRYPTO] WARNING: Invalid padding value: {padding}")
            return decrypted
        
        # Verify padding
        if decrypted[-padding:] != bytes([padding] * padding):
            print("[CRYPTO] WARNING: Padding verification failed")
            return decrypted
        
        return decrypted[:-padding]
        
    except Exception as e:
        print(f"[CRYPTO] ERROR: AES decryption failed: {e}")
        return None


# ============================================================================
# Convenience Functions (for compatibility)
# ============================================================================

def encrypt_data(data: bytes, key: bytes) -> Optional[bytes]:
    """
    Encrypt data (alias for encrypt_aes).
    
    Args:
        data: Data bytes to encrypt
        key: Encryption key bytes
        
    Returns:
        Encrypted data or None on error
    """
    return encrypt_aes(data, key)


def decrypt_data(encrypted_data: bytes, key: bytes) -> Optional[bytes]:
    """
    Decrypt data (alias for decrypt_aes).
    
    Args:
        encrypted_data: Encrypted data bytes
        key: Decryption key bytes
        
    Returns:
        Decrypted data or None on error
    """
    return decrypt_aes(encrypted_data, key)


def encrypt_packet(data: bytes, key: bytes) -> Optional[bytes]:
    """
    Encrypt a packet (alias for encrypt_aes).
    
    Args:
        data: Packet data bytes
        key: Encryption key bytes
        
    Returns:
        Encrypted packet or None on error
    """
    return encrypt_aes(data, key)


def decrypt_packet(encrypted_data: bytes, key: bytes) -> Optional[bytes]:
    """
    Decrypt a packet (alias for decrypt_aes).
    
    Args:
        encrypted_data: Encrypted packet bytes
        key: Decryption key bytes
        
    Returns:
        Decrypted packet or None on error
    """
    return decrypt_aes(encrypted_data, key)
