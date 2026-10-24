import os
import hashlib
import hmac


# ============================================================
# ADMIN CONFIGURATION
# ============================================================

ADMIN_USERNAME = os.environ.get(
    "AI_ATTEND_ADMIN",
    "admin"
)

ADMIN_PASSWORD = os.environ.get(
    "AI_ATTEND_PASSWORD"
)


# ============================================================
# PASSWORD HASH
# ============================================================

def hash_password(password):

    salt = os.urandom(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        310000
    )

    return (
        salt.hex()
        + ":"
        + password_hash.hex()
    )


# ============================================================
# VERIFY PASSWORD
# ============================================================

def verify_password(
    password,
    stored_password
):

    try:

        salt_hex, hash_hex = (
            stored_password.split(":")
        )

        salt = bytes.fromhex(
            salt_hex
        )

        expected_hash = bytes.fromhex(
            hash_hex
        )

        actual_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            310000
        )

        return hmac.compare_digest(
            actual_hash,
            expected_hash
        )

    except Exception:

        return False


# ============================================================
# CREATE ADMIN PASSWORD HASH
# ============================================================

def create_password_hash(password):

    return hash_password(
        password
    )