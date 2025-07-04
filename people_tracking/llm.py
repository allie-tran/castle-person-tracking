import asyncio
import base64
import json
import os
from collections.abc import Sequence
from typing import AsyncGenerator, Dict, Generator, List, Literal, Optional

from openai import AsyncOpenAI, BaseModel, OpenAI
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from partialjson.json_parser import JSONParser
from pyrate_limiter import BucketFullException, Duration, Limiter, Rate
from rich import print

from dotenv import load_dotenv

load_dotenv()

DEBUG = True
JSON_START_FLAG = "```json"
JSON_END_FLAG = "```"

parser = JSONParser()
parser.on_extra_token = lambda *_, **__: None

rate = Rate(3, Duration.SECOND)
limiter = Limiter(rate)

# Set up ChatGPT generation model
OPENAI_API = os.environ.get("OPENAI_API", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")


class MixedContent(BaseModel):
    type: Literal["text", "image_url"]
    content: str


class LLM:
    # Set up the template messages to use for the completion
    template_message: ChatCompletionMessageParam = ChatCompletionSystemMessageParam(
        role="system", content="You are a useful assistant."
    )

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API)
        self.model_name = MODEL_NAME

    def generate(self, messages: List[ChatCompletionMessageParam], parse_json=False):
        """
        Generate completions from a list of messages
        """
        request = self.client.chat.completions.create(
            model=self.model_name, messages=messages
        )
        response = request.choices[0].message.content
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
        messages = [self.template_message]
        messages.append(ChatCompletionUserMessageParam(role="user", content=text))
        return self.generate(messages, parse_json)

    def generate_from_mixed_media(
        self, data: Sequence[MixedContent], parse_json=False
    ) -> Optional[Dict | str]:
        messages = [self.template_message]
        content: List[ChatCompletionContentPartParam] = []
        for part in data:
            if part.type == "text":
                content.append(
                    ChatCompletionContentPartTextParam(text=part.content, type="text")
                )
            elif part.type == "image_url":
                content.append(
                    ChatCompletionContentPartImageParam(
                        image_url=ImageURL(url=part.content), type="image_url"
                    )
                )
        messages.append(ChatCompletionUserMessageParam(role="user", content=content))
        return self.generate(messages, parse_json=parse_json)


def get_openai_visual_messages(image_paths: List[str | bytes]) -> List[MixedContent]:
    """
    Get a visual message for OpenAI from a list of image paths.
    """
    if not image_paths:
        return []

    messages = []
    for image_path in image_paths:
        try:
            base64_code = base64.b64encode(
                open(image_path, "rb").read() if isinstance(image_path, str) else image_path
            ).decode("utf-8")
            bs64_image = f"data:image/jpeg;base64,{base64_code}"
            messages.append(MixedContent(type="image_url", content=bs64_image))
        except OSError as e:
            print(f"Error reading image {image_path}: {e}")
            continue

    return messages
