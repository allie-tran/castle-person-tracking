import json
import os
from collections.abc import Sequence
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from google import genai  # type: ignore
from google.genai.types import Content, GenerateContentConfig, Part  # type: ignore
from partialjson.json_parser import JSONParser
from pydantic import BaseModel
from pyrate_limiter import Duration, Limiter, Rate
from rich import print

load_dotenv()

DEBUG = False
JSON_START_FLAG = "```json"
JSON_END_FLAG = "```"

parser = JSONParser()
parser.on_extra_token = lambda *_, **__: None

rate = Rate(3, Duration.SECOND)
limiter = Limiter(rate)

# Set up ChatGPT generation model
GEMINI = os.environ.get("GEMINI_API", "")
MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "")


class MixedContent(BaseModel):
    type: Literal["text", "image_url"]
    content: str | bytes


class LLM:
    # Set up the template messages to use for the completion
    system_instruction: str = "You are a helpful assistant."

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI)
        self.model_name = MODEL_NAME

    def generate(self, contents: Content, parse_json=False):
        """
        Generate completions from a list of messages
        """
        request = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=GenerateContentConfig(system_instruction=self.system_instruction),
        )
        response = request.text
        if DEBUG:
            print("GPT", response)
        if parse_json and response:
            return self.__parse(response)
        else:
            return response

    def __parse(self, response: str) -> Optional[Dict]:
        while JSON_START_FLAG in response:
            start = response.find(JSON_START_FLAG)
            response = response[start + len(JSON_START_FLAG) :]
            json_object = response
            if JSON_END_FLAG in response:
                end = response.find(JSON_END_FLAG)
                json_object = response[: end + len(JSON_END_FLAG)]
                response = response[end + len(JSON_END_FLAG) :]
            try:
                json_object = parser.parse(json_object)
                return json_object
            except json.JSONDecodeError:
                pass

    def generate_from_text(self, text: str, parse_json=False) -> Optional[Dict | str]:
        """
        Generate completions from text
        Then parse the JSON object from the completion
        If the completion is not a JSON object, return the text
        """
        contents = Content(role="user", parts=[Part.from_text(text=text)])
        return self.generate(contents, parse_json)

    def generate_from_mixed_media(
        self, data: Sequence[MixedContent], parse_json=False
    ) -> Optional[Dict | str]:
        parts: List[Part] = []
        for part in data:
            if part.type == "text":
                parts.append(Part.from_text(text=part.content))
            elif part.type == "image_url":
                parts.append(Part.from_bytes(data=part.content, mime_type="image/jpeg"))
        return self.generate(Content(role="user", parts=parts), parse_json=parse_json)


def get_visual_content(image_paths: List[str] | List[bytes]) -> List[MixedContent]:
    """
    Get a visual message for OpenAI from a list of image paths.
    """
    if not image_paths:
        return []

    messages = []
    for image_path in image_paths:
        try:
            image_bytes = (
                open(image_path, "rb").read()
                if isinstance(image_path, str)
                else image_path
            )
            messages.append(MixedContent(type="image_url", content=image_bytes))
        except OSError as e:
            print(f"Error reading image {image_path}: {e}")
            continue
    return messages
