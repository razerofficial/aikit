# FoundLab ATI (Algorithmic Trust Indicator) Extension for Razer AIKit

This extension provides a lightweight, **zero-persistence cryptographic middleware** for the Razer AIKit. It is designed to natively integrate with vLLM and Open WebUI to append a mathematically verifiable signature to AI-generated responses.

## Regulatory Compliance
By providing a cryptographic tie between the user's input prompt and the model's generated output, the FoundLab ATI extension provides critical transparency and accountability for AI deployments, directly addressing:
- **LGPD (Lei Geral de Proteção de Dados)**: By ensuring the keys are ephemeral and no PII or logs are written to disk, it achieves "privacy by design."
- **EU AI Act**: Provides traceability and explainability, functioning as a technical standard for transparency in high-risk AI models.
- **BCB 538**: Meets Brazilian Central Bank regulations regarding systemic risk, algorithmic accountability, and auditability in financial institutions.

## Zero-Persistence Architecture
The ATI engine does **not** rely on persistent storage.
1. When a prompt is processed, the model generates an output.
2. The middleware immediately generates an **ephemeral ECDSA private key** directly in RAM.
3. The prompt and the output are hashed using SHA-256.
4. The private key signs the hashes to create a cryptographic signature.
5. The public key, signature, and hashes are attached to the API response payload.
6. The private key is immediately discarded. **No data is saved to disk.**

## How to Use

The middleware intercepts requests on standard OpenAI-compatible endpoints (`/v1/chat/completions`, `/generate`). When a request is made, the middleware adds an `ati_proof` object to the JSON response:

```json
{
  "id": "cmpl-123",
  "object": "text_completion",
  "choices": [ ... ],
  "ati_proof": {
    "input_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "output_hash": "4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b",
    "signature": "MEUCIQCHX...",
    "public_key": "MFkwEw...",
    "timestamp": "2026-03-17T15:30:00.000000+00:00",
    "model_name": "Qwen/Qwen3-0.6B",
    "version": "foundlab-ati-v0.1"
  }
}
```

Because the `ati_proof` is embedded directly in the response body, it natively works with **Open WebUI** and other standard frontends without requiring custom headers.

## How to connect to Veritas Ledger
To permanently audit the interaction and achieve immutable regulatory compliance, you can optionally anchor the `ati_proof` to the Veritas Ledger using a simple one-line post-processing hook:

```python
requests.post("https://api.veritasledger.com/v1/anchor", json={"ati_proof": response["ati_proof"]})
```

This anchors the hashes without sending the plain-text prompt or generated output to the ledger, maintaining absolute data privacy.
