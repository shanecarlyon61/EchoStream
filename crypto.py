"""
Cryptography functions - Base64 encoding/decoding and AES-256-GCM encryption/decryption
"""
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend
import os

def encode_base64(data: bytes) -> str:
    """Encode data to base64 string"""
    return base64.b64encode(data).decode('utf-8')

def decode_base64(input_str: str) -> bytes:
    """Decode base64 string to bytes"""
    try:
        return base64.b64decode(input_str)
    except Exception:
        return b''

def decode_base64_len(input_str: str) -> bytes:
    """Decode base64 string to bytes (alias for decode_base64)"""
    return decode_base64(input_str)

def encrypt_data(data: bytes, key: bytes) -> bytes:
    """Encrypt data using AES-256-GCM"""
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes for AES-256")
    
    # Generate random IV (12 bytes for GCM)
    iv = os.urandom(12)
    
    # Create AESGCM cipher
    aesgcm = AESGCM(key)
    
    # Encrypt data
    ciphertext = aesgcm.encrypt(iv, data, None)
    
    # Return IV + ciphertext (IV is 12 bytes, ciphertext includes tag)
    return iv + ciphertext

def decrypt_data(data: bytes, key: bytes) -> bytes:
    """Decrypt data using AES-256-GCM"""
    if len(data) < 28:  # IV(12) + TAG(16) minimum
        return b''
    
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes for AES-256")
    
    # Extract IV and ciphertext
    iv = data[:12]
    ciphertext = data[12:]
    
    # Create AESGCM cipher
    aesgcm = AESGCM(key)
    
    try:
        # Decrypt data
        plaintext = aesgcm.decrypt(iv, ciphertext, None)
        return plaintext
    except Exception:
        return b''

