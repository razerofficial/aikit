import hashlib
import base64
import json
import datetime
from ecdsa import SigningKey, VerifyingKey, SECP256k1, BadSignatureError

def generate_proof(input_text: str, output_text: str, model_name: str) -> dict:
    """
    Generates a zero-persistence cryptographic proof (ATI) for the given input and output.
    An ephemeral ECDSA private key is generated in memory and never saved to disk.
    """
    # Generate an ephemeral ECDSA private key
    # This key only lives in RAM and is discarded after generating the signature
    private_key = SigningKey.generate(curve=SECP256k1)
    public_key = private_key.get_verifying_key()
    
    # Calculate SHA-256 hashes of the input and output
    input_hash = hashlib.sha256(input_text.encode('utf-8')).hexdigest()
    output_hash = hashlib.sha256(output_text.encode('utf-8')).hexdigest()
    
    # Create the payload to sign (combining input and output hashes)
    payload_to_sign = f"{input_hash}:{output_hash}".encode('utf-8')
    
    # Sign the payload
    signature = private_key.sign(payload_to_sign)
    
    # Prepare the JSON-serializable proof dictionary
    proof_dict = {
        "input_hash": input_hash,
        "output_hash": output_hash,
        "signature": base64.b64encode(signature).decode('utf-8'),
        "public_key": base64.b64encode(public_key.to_string()).decode('utf-8'),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_name": model_name,
        "version": "foundlab-ati-v0.1"
    }
    
    return proof_dict

def verify_proof(proof_dict: dict, input_text: str, output_text: str) -> bool:
    """
    Verifies a zero-persistence cryptographic proof (ATI).
    """
    try:
        # Reconstruct hashes
        expected_input_hash = hashlib.sha256(input_text.encode('utf-8')).hexdigest()
        expected_output_hash = hashlib.sha256(output_text.encode('utf-8')).hexdigest()
        
        # Check if hashes match
        if expected_input_hash != proof_dict["input_hash"] or expected_output_hash != proof_dict["output_hash"]:
            return False
            
        # Reconstruct the payload
        payload_to_verify = f"{expected_input_hash}:{expected_output_hash}".encode('utf-8')
        
        # Load the public key and signature
        public_key_bytes = base64.b64decode(proof_dict["public_key"])
        signature_bytes = base64.b64decode(proof_dict["signature"])
        
        # Reconstruct the verifying key
        verifying_key = VerifyingKey.from_string(public_key_bytes, curve=SECP256k1)
        
        # Verify the signature
        return verifying_key.verify(signature_bytes, payload_to_verify)
        
    except (KeyError, ValueError, BadSignatureError, Exception):
        return False
