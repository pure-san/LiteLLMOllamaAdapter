import sys
import logging
import json
from datetime import datetime, timezone
from flask import Flask, request, jsonify, Response
from litellm import completion

# --debug option: log detailed messages to stdout
if "--debug" in sys.argv:
    logging.basicConfig(
        level=logging.DEBUG,
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

app = Flask(__name__)
SERVER_PORT = 3030

# Allowed models (normalize incoming model names)
ALLOWED_MODELS = [
    "openrouter/qwen/qwen-2.5-coder-32b-instruct",
    "openrouter/qwen/qwen2.5-vl-72b-instruct:free",
    "openrouter/google/gemini-2.0-pro-exp-02-05:free",
    "openrouter/meta-llama/llama-3.2-11b-vision-instruct",
]

# Allowed models information (for /api/tags endpoint)
# If no ':' is present in the model name, add ":latest" to the name for
# consistency.
ALLOWED_MODELS_INFO = [
    {
        "name": "openrouter/qwen/qwen-2.5-coder-32b-instruct:latest",
        "model": "openrouter/qwen/qwen-2.5-coder-32b-instruct:latest",
        "modified_at": "2023-12-12T00:00:00Z",
        "size": 1234567890,
        "digest": "dummyhash_qwen",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "qwen",
            "families": ["qwen"],
            "parameter_size": "32B",
            "quantization_level": "Q4_0"
        }
    },
    {
        "name": "openrouter/qwen/qwen2.5-vl-72b-instruct:free",
        "model": "openrouter/qwen/qwen2.5-vl-72b-instruct:free",
        "modified_at": "2023-12-12T00:00:00Z",
        "size": 1234567890,
        "digest": "dummyhash_qwen",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "qwen",
            "families": ["qwen"],
            "parameter_size": "72B",
            "quantization_level": "Q4_0"
        }
    },
    {
        "name": "openrouter/google/gemini-2.0-pro-exp-02-05:free",
        "model": "openrouter/google/gemini-2.0-pro-exp-02-05:free",
        "modified_at": "2023-12-12T00:00:00Z",
        "size": 2345678901,
        "digest": "dummyhash_gemini",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "gemini",
            "families": ["gemini"],
            "parameter_size": "2.0",
            "quantization_level": "none"
        }
    },
    {
        "name": "openrouter/meta-llama/llama-3.2-11b-vision-instruct:latest",
        "model": "openrouter/meta-llama/llama-3.2-11b-vision-instruct:latest",
        "modified_at": "2023-12-12T00:00:00Z",
        "size": 2345678901,
        "digest": "dummyhash_llama",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "2.0",
            "quantization_level": "none"
        }
    }
]


def normalize_model_name(model_name):
    """
    Remove a trailing ":latest" if present.
    """
    if model_name.endswith(":latest"):
        model_name = model_name[:-7]
    return model_name


def format_response(model, content, image_data=None):
    """Format the final non-streaming response."""
    created_at = datetime.now(timezone.utc).isoformat()
    return {
        "model": model,
        "created_at": created_at,
        "message": {
            "role": "assistant",
            "content": content.rstrip("null"),
            "images": image_data,
        },
        "done": True,
        "total_duration": 5191566416,
        "load_duration": 2154458,
        "prompt_eval_count": 26,
        "prompt_eval_duration": 383809000,
        "eval_count": 298,
        "eval_duration": 4799921000,
    }


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat endpoint:
      - Reads the JSON payload and optionally an uploaded image.
      - If an image file is uploaded (via multipart/form-data)
        or an "image_data" field exists in the JSON,
        the image is base64-encoded and added to the parameters.
      - Calls litellm.completion() with messages (and image_data if provided).
      - If stream is False, extracts the response text from the ModelResponse.
    """
    try:
        # Try to get JSON data
        data = request.get_json(silent=True)
        if data is None:
            data = {}
        logging.debug(f"Received chat request data: {data}")

        # Check for image data from file upload (multipart/form-data)
        image_data = None
        messages = data.get("messages", [])
        if messages and "images" in messages[0]:
            img = messages[0]["images"]
            ctype = messages[0].get("content_type", "image/png")
            if img:
                base64_data = img[0]
                image_data = f"data:{ctype};base64,{base64_data}"
            msg = "Image data received in JSON payload (from messages). "
            logging.debug(f"{msg}Content type: {ctype}")

        if not data or not messages or "model" not in data:
            error_msg = "Invalid request format"
            logging.error(error_msg)
            return jsonify({"error": error_msg}), 400

        raw_model = data["model"]
        model = normalize_model_name(raw_model)
        logging.debug(
            f"Received model: '{raw_model}' normalized to: '{model}'"
        )

        if model not in ALLOWED_MODELS:
            msg = (
                f"Model '{raw_model}' (normalized: '{model}') is not allowed."
                f"Allowed models: {ALLOWED_MODELS}."
            )
            logging.error(msg)
            return jsonify({"error": msg}), 400

        is_streaming = data.get("stream", True)
        logging.debug(f"Messages received: {messages}")

        if image_data:
            # Find last user message
            role = "user"
            user_msgs = [m for m in messages if m.get("role") == role]
            last_msg = user_msgs[-1] if user_msgs else None
            if last_msg:
                text = last_msg.get("content", "")
                new_message = {
                    "role": role,
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data},
                        },
                    ],
                }
                # Replace with the newly formatted message
                messages = [new_message]
            else:
                logging.warning("No user message found to attach image to.")

        logging.debug(f"Messages after processing: {messages}")

        temperature = data.get("temperature", 0.6)
        max_tokens = data.get("max_tokens", 512)
        top_p = data.get("top_p", 0.9)
        transforms = data.get("transforms", [])
        route = data.get("route", "")
        logging.debug(
            f"Parameters: temperature={temperature}, "
            f"max_tokens={max_tokens}, top_p={top_p}, "
            f"stream={is_streaming}, transforms={transforms}, "
            f"route={route}"
        )

        completion_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": is_streaming,
            "transforms": transforms,
        }
        if route:
            completion_params["route"] = route

        if is_streaming:
            logging.debug("Processing streaming response.")
            generator = completion(**completion_params)

            def generate():
                content = ""
                created_at = datetime.now(timezone.utc).isoformat()
                for chunk in generator:
                    if hasattr(chunk.choices[0], "delta") and hasattr(
                        chunk.choices[0].delta, "content"
                    ):
                        token_content = chunk.choices[0].delta.content
                        logging.debug(f"Streaming token: {token_content}")
                        if token_content is not None:
                            content += str(token_content)
                            message_content = {
                                "role": "assistant",
                                # Send only the new token
                                "content": token_content,
                                "images": image_data,
                            }
                            yield (
                                json.dumps({
                                    "model": model,
                                    "created_at": created_at,
                                    "message": message_content,
                                    "done": False,
                                }).encode("utf-8") + b"\n"
                            )
                # Send final done message with complete content
                yield (
                    json.dumps({
                        "model": model,
                        "created_at": created_at,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "images": image_data,
                        },
                        "done": True,
                    }).encode("utf-8") + b"\n"
                )

            return Response(generate(), mimetype="application/json")
        else:
            result = completion(**completion_params)
            log_msg = "completion() call successful. "
            logging.debug(f"{log_msg}Result type: {type(result)}")
            # Non-streaming: extract response text from ModelResponse
            logging.debug(f"Full result object: {result}")  # Added logging
            if hasattr(result, "choices") and result.choices:
                try:
                    response_text = result.choices[0].message.content
                    logging.debug(f"Response text: {response_text}")
                except Exception as e:
                    msg = f"Failed to extract message from result: {e}"
                    logging.error(msg)
                    response_text = str(result)
            else:
                response_text = str(result)
                logging.debug(f"Non-streaming fallback: {response_text}")

            if response_text is None:  # Handle None case
                logging.warning("response_text is None. Setting to empty str.")
                response_text = ""

            final_response = format_response(
                model, response_text, image_data=image_data
            )
            logging.debug(f"Sending final response: {final_response}")
            return jsonify(final_response)

    except Exception as e:
        logging.error(f"Error in /api/chat: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/version', methods=['GET'])
def get_version():
    """Return API version information."""
    version_info = {
        "version": "0.3.35",
        "description": "A fake API server for OpenWebUI"
    }
    logging.debug("Returning API version information.")
    return jsonify(version_info)


@app.route('/api/tags', methods=['GET'])
def get_tags():
    """
    Return detailed information about available models.
    """
    logging.debug("Returning allowed models information.")
    return jsonify({"models": ALLOWED_MODELS_INFO})


if __name__ == '__main__':
    logging.info(f"Starting server on port {SERVER_PORT}")
    app.run(host='0.0.0.0', port=SERVER_PORT)
