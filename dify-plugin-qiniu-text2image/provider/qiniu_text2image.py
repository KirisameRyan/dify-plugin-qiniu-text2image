from typing import Any
import requests

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError


class QiniuText2imageProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        """
        Validate Qiniu AI API credentials using a cheap chat completion request
        This avoids the cost of image generation during validation
        """
        try:
            # Check if API key exists
            api_key = credentials.get("api_key")
            if not api_key or not api_key.strip():
                raise ToolProviderCredentialValidationError(
                    "API Key is required and cannot be empty"
                )
            
            # Use chat completion API for validation (much cheaper than image generation)
            url = "https://api.qnaigc.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.strip()}"
            }
            payload = {
                "model": "gemini-2.0-flash-exp",  # Use cheapest/fastest model
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1  # Minimal tokens to reduce cost
            }
            
            # Send validation request with timeout
            response = requests.post(
                url, 
                json=payload, 
                headers=headers, 
                timeout=15
            )
            
            # Check response status
            if response.status_code == 401:
                raise ToolProviderCredentialValidationError(
                    "Invalid API Key. Please check your Qiniu AI API Key."
                )
            elif response.status_code == 403:
                raise ToolProviderCredentialValidationError(
                    "API Key does not have permission to access Qiniu AI services."
                )
            elif 200 <= response.status_code < 300:
                # Success - API key is valid
                pass
            elif response.status_code >= 500:
                # Server error - don't fail validation
                pass
            elif response.status_code == 400:
                # Bad request - check error message
                try:
                    error_data = response.json()
                    error_message = error_data.get("error", {}).get("message", "")
                    if "api" in error_message.lower() or "key" in error_message.lower() or "auth" in error_message.lower():
                        raise ToolProviderCredentialValidationError(
                            f"API Key validation failed: {error_message}"
                        )
                except:
                    # Can't parse error, but if it's 400 and not auth-related, key might be valid
                    pass
                    
        except ToolProviderCredentialValidationError:
            raise
        except requests.exceptions.Timeout:
            # Timeout - don't fail validation
            pass
        except requests.exceptions.RequestException:
            # Network error - don't fail validation
            pass
        except Exception as e:
            raise ToolProviderCredentialValidationError(
                f"Failed to validate credentials: {str(e)}"
            ) from e
