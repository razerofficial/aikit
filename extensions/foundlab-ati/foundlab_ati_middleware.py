import json
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .generate_proof import generate_proof

class FoundLabATIMiddleware(BaseHTTPMiddleware):
    """
    FastAPI/ASGI middleware that integrates with the vLLM OpenAI-compatible server.
    It hooks into the generate/completions endpoints and adds an ATI proof to the response.
    """
    
    async def dispatch(self, request: Request, call_next):
        # We only want to process chat completions or generate requests
        if request.url.path not in ["/v1/chat/completions", "/v1/completions", "/generate"]:
            return await call_next(request)
            
        # Extract the request body for the input text
        try:
            body_bytes = await request.body()
            body = json.loads(body_bytes.decode('utf-8'))
            
            # Try to get the prompt from chat format or completion format
            if "messages" in body:
                input_text = json.dumps(body["messages"])
            else:
                input_text = body.get("prompt", str(body))
                
            model_name = body.get("model", "unknown-model")
        except Exception:
            input_text = ""
            model_name = "unknown"
            
        # Call the next middleware / endpoint
        response = await call_next(request)
        
        # Only process successful JSON responses (non-streaming)
        if response.status_code == 200 and getattr(response, 'media_type', None) == "application/json":
            # Buffer the response body
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
                
            try:
                data = json.loads(response_body.decode('utf-8'))
                
                # Extract the generated text
                output_text = ""
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        output_text = choice["message"]["content"]
                    elif "text" in choice:
                        output_text = choice["text"]
                else:
                    output_text = str(data)
                    
                # Generate the ATI Proof
                ati_proof = generate_proof(input_text, output_text, model_name)
                
                # Add the ATI proof to the response payload
                data["ati_proof"] = ati_proof
                
                # Create a new response with the modified JSON data
                new_body = json.dumps(data).encode('utf-8')
                
                # Update headers (especially content-length)
                headers = dict(response.headers)
                headers['content-length'] = str(len(new_body))
                
                return Response(
                    content=new_body,
                    status_code=response.status_code,
                    headers=headers,
                    media_type=response.media_type
                )
                
            except Exception:
                # If anything fails, return the original buffered response
                return Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type
                )
                
        return response
