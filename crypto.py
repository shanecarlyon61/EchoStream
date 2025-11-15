
import base64
from typing import Optional
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

def decode_base64(key_str: str) -> Optional[bytes]:

    try:
        decoded = base64.b64decode(key_str)
        return decoded
    except Exception as e:
        print(f"[CRYPTO] ERROR: Failed to decode base64: {e}")
        return None

def encode_base64(data: bytes) -> str:

    try:
        encoded = base64.b64encode(data).decode('utf-8')
        return encoded
    except Exception as e:
        print(f"[CRYPTO] ERROR: Failed to encode base64: {e}")
        return ""

def encrypt_aes(data: bytes, key: bytes, iv: Optional[bytes] = None) -> Optional[bytes]:

    try:

        if len(key) != 32:
            if len(key) < 32:
                key = key + b'\x00' * (32 - len(key))
            else:
                key = key[:32]

        if iv is None:
            iv = os.urandom(16)

        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )

        block_size = 16
        padding = block_size - (len(data) % block_size)
        padded_data = data + bytes([padding] * padding)

        encryptor = cipher.encryptor()
        encrypted = encryptor.update(padded_data) + encryptor.finalize()

        return iv + encrypted

    except Exception as e:
        print(f"[CRYPTO] ERROR: AES encryption failed: {e}")
        return None

def decrypt_aes(encrypted_data: bytes, key: bytes) -> Optional[bytes]:

    try:

        if len(key) != 32:
            if len(key) < 32:
                key = key + b'\x00' * (32 - len(key))
            else:
                key = key[:32]

        if len(encrypted_data) < 16:
            print("[CRYPTO] ERROR: Encrypted data too short (missing IV)")
            return None

        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]

        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )

        decryptor = cipher.decryptor()
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()

        padding = decrypted[-1]
        if padding > 16 or padding < 1:
            print(f"[CRYPTO] WARNING: Invalid padding value: {padding}")
            return decrypted

        if decrypted[-padding:] != bytes([padding] * padding):
            print("[CRYPTO] WARNING: Padding verification failed")
            return decrypted

        return decrypted[:-padding]

    except Exception as e:
        print(f"[CRYPTO] ERROR: AES decryption failed: {e}")
        return None

def encrypt_data(data: bytes, key: bytes) -> Optional[bytes]:

    return encrypt_aes(data, key)

def decrypt_data(encrypted_data: bytes, key: bytes) -> Optional[bytes]:

    return decrypt_aes(encrypted_data, key)

def encrypt_packet(data: bytes, key: bytes) -> Optional[bytes]:

    return encrypt_aes(data, key)

def decrypt_packet(encrypted_data: bytes, key: bytes) -> Optional[bytes]:

    return decrypt_aes(encrypted_data, key)
