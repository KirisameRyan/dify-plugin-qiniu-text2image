from collections.abc import Generator
from typing import Any
import requests
import base64

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage


class QiniuText2imageTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        # Get credentials
        api_key = self.runtime.credentials.get("api_key")
        if not api_key:
            yield self.create_text_message(
                "Error: Qiniu AI API Key not configured. Please add it in plugin credentials."
            )
            return

        # Get parameters
        prompt = tool_parameters.get("prompt", "").strip()
        if not prompt:
            yield self.create_text_message("Error: Prompt cannot be empty")
            return

        model = tool_parameters.get("model", "gemini-3.0-pro-image-preview")
        n = int(tool_parameters.get("n", 1))
        aspect_ratio = tool_parameters.get("aspect_ratio", "16:9")

        # API configuration
        url = "https://api.qnaigc.com/v1/images/generations"
        payload = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "image_config": {"aspect_ratio": aspect_ratio}
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}"
        }

        try:
            # Send progress message
            yield self.create_text_message(
                f"🎨 Generating {n} image(s) with {model}..."
            )
            
            # Make API request
            response = requests.post(url, json=payload, headers=headers, timeout=90)
            response.raise_for_status()
            data = response.json()

            # Process response
            if "data" in data and data["data"]:
                images_generated = len(data["data"])
                yield self.create_text_message(
                    f"✓ Successfully generated {images_generated} image(s) using {model}"
                )

                # Return all images
                for i, item in enumerate(data["data"]):
                    if "b64_json" in item:
                        try:
                            image_bytes = base64.b64decode(item["b64_json"])
                            yield self.create_blob_message(
                                blob=image_bytes,
                                meta={
                                    "filename": f"qiniu_ai_image_{i+1}.png",
                                    "mime_type": "image/png"
                                }
                            )
                        except Exception as decode_error:
                            yield self.create_text_message(
                                f"⚠ Warning: Failed to decode image {i+1}: {str(decode_error)}"
                            )
                    else:
                        yield self.create_text_message(
                            f"⚠ Warning: Image {i+1} data is missing"
                        )
            else:
                # API returned no image data
                yield self.create_text_message(
                    "Error: API did not return image data"
                )
                yield self.create_json_message({
                    "error": "No image data in response",
                    "response": data
                })

        except requests.exceptions.Timeout:
            yield self.create_text_message(
                "Error: Request timeout (90s). The model might be overloaded, please try again later."
            )
        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code if http_err.response else "Unknown"
            error_detail = ""
            try:
                error_detail = http_err.response.json() if http_err.response else {}
            except:
                error_detail = http_err.response.text if http_err.response else str(http_err)
            
            yield self.create_text_message(
                f"Error: HTTP {status_code} - {error_detail}"
            )
        except requests.exceptions.RequestException as req_err:
            yield self.create_text_message(
                f"Error: Request failed - {str(req_err)}"
            )
        except Exception as e:
            yield self.create_text_message(
                f"Error: Unexpected error - {str(e)}"
            )
