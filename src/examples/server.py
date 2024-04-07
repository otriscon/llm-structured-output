# pylint: disable=missing-function-docstring,missing-class-docstring
"""
Example model server with OpenAI-like API, including function calls / tools.
"""
import json
import os
from enum import Enum
from typing import Literal, List, Optional, Union

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from examples.llm_schema import Model
from llm_structured_output.util.output import info, warning, debug


app = FastAPI()

model = Model()
info("Loading model...")
try:
    model.load(os.environ["MODEL_PATH"])
except KeyError:
    warning("Need to specify MODEL_PATH environment variable")


@app.exception_handler(RequestValidationError)
# pylint: disable-next=unused-argument
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}"
    warning(f"RequestValidationError: {exc_str}")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


@app.get("/status")
def get_status():
    return {"status": "OK"}


class V1ChatMessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class V1ChatMessage(BaseModel):
    role: V1ChatMessageRole
    content: str


class V1Function(BaseModel):
    name: str
    description: str = ""
    parameters: dict = {}


class V1ToolFunction(BaseModel):
    type: Literal["function"]
    function: V1Function


class V1ToolChoiceKeyword(str, Enum):
    AUTO = "auto"
    NONE = "none"


class V1ToolChoiceFunction(BaseModel):
    type: Optional[Literal["function"]] = None
    name: str


class V1ResponseFormatType(str, Enum):
    JSON_OBJECT = "json_object"


class V1ResponseFormat(BaseModel):
    type: V1ResponseFormatType


class V1ChatCompletionsRequest(
    BaseModel
):  # pylint: disable=too-many-instance-attributes
    model: str
    max_tokens: int = 1000
    temperature: float = 0.0
    messages: List[V1ChatMessage]
    # The 'functions' and 'function_call' fields have been dreprecated and
    # replaced with 'tools' and 'tool_choice', that work similarly but allow
    # for multiple functions to be invoked.
    functions: List[V1Function] = None
    function_call: Union[V1ToolChoiceKeyword, V1ToolChoiceFunction] = None
    tools: List[V1ToolFunction] = None
    tool_choice: Union[V1ToolChoiceKeyword, V1ToolChoiceFunction] = None
    response_format: V1ResponseFormat = None
    stream: bool = False


@app.post("/v1/chat/completions")
async def post_v1_chat_completions(request: V1ChatCompletionsRequest):
    debug("REQUEST", request)
    response = post_v1_chat_completions_impl(request)
    debug("RESPONSE", response)
    return response


def post_v1_chat_completions_impl(request: V1ChatCompletionsRequest):
    messages = request.messages[:]

    functions = []
    if request.tool_choice == "none":
        pass
    elif request.tool_choice == "auto":
        functions = [tool.function for tool in request.tools if tool.type == "function"]
    elif request.tool_choice is not None:
        functions = [
            next(
                tool.function
                for tool in request.tools
                if tool.type == "function"
                and tool.function.name == request.function_call.name
            )
        ]
    elif request.function_call == "none":
        pass
    elif request.function_call == "auto":
        functions = request.functions
    elif request.function_call is not None:
        functions = [
            next(
                fn for fn in request.functions if fn.name == request.function_call.name
            )
        ]

    schema = None
    if len(functions) > 0:
        function_schemas = [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "const", "const": fn.name},
                    "arguments": fn.parameters,
                },
                "required": ["name", "arguments"],
            }
            for fn in functions
        ]
        separator = "\n\n"
        if len(functions) == 1:
            schema = function_schemas[0]
            tool_prompt = f"""
You are a helpful assistant with access to a tool that you must invoke to answer the user's request.
The tool is:
Function {functions[0].name}: {functions[0].description}
Invocation schema: {json.dumps(function_schemas[0])}
Your answer is a JSON object following the invocation schema in order to answer the user request below.
"""
        elif request.tool_choice:
            schema = {"type": "array", "items": {"anyOf": function_schemas}}
            tool_prompt = f"""
You are a helpful assistant with access to tools that you must invoke to answer the user's request.
The following tools are available:
{separator.join([ f'''
Function {fn.name}: {fn.description}
Invocation schema: {json.dumps(fn_schema)}
''' for fn, fn_schema in zip(functions, function_schemas) ])}
Your answer is a JSON array with one or more tool invocations according to the appropriate schema(s)
in order to answer the user request below.
"""
        else:  # Legacy function calling only allowed one function to be called.
            schema = {"oneOf": function_schemas}
            tool_prompt = f"""
You are a helpful assistant with access to tools that you must invoke to answer the user's request.
The following tools are available:
{"separator".join([ f'''
Function {fn.name}: {fn.description}
Invocation schema: {json.dumps(fn_schema)}
''' for fn, fn_schema in zip(functions, function_schemas) ])}
Your answer is a JSON object following the invocation schema of the most appropriate tool to use
to answer the user request below.
"""
        # Insert a pre-prompt instructing the LLM with the tool schemas.
        messages.append(
            V1ChatMessage(
                role="system",
                content=tool_prompt,
            )
        )

    prompt = "\n".join(
        [
            (
                message.content
                if message.role == V1ChatMessageRole.ASSISTANT
                else f"[INST] {message.content} [/INST]"
            )
            for message in messages
        ]
    )

    if schema is not None:
        debug("Using schema:", schema)

    if request.response_format is not None:
        if request.response_format.type != "json_object":
            warning(
                f"Unsupported response format type '{request.response_format.type}'"
            )
        elif schema is None:
            schema = {"type": "object"}

    info("Starting generation...")

    content = ""
    prompt_tokens = 0
    completion_tokens = 0

    for result in model.completion(
        prompt,
        schema=schema,
        max_tokens=request.max_tokens,
        temp=request.temperature,
    ):
        if result["op"] == "evaluatedPrompt":
            prompt_tokens += result["token_count"]
        elif result["op"] == "generatedTokens":
            completion_tokens += result["token_count"]
            content += result["text"]
        elif result["op"] == "stop":
            finish_reason = translate_reason(request, result["reason"])
        else:
            assert False
    if finish_reason == "tool_calls":
        tool_calls = json.loads(content)
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                        "arguments": json.dumps(function_call["arguments"]),
                    },
                }
                for function_call in tool_calls
            ],
        }
    elif finish_reason == "function_call":
        function_call = json.loads(content)
        message = {
            "role": "assistant",
            "function_call": {
                "name": function_call["name"],
                "arguments": json.dumps(function_call["arguments"]),
            },
        }
    else:
        message = {"role": "assistant", "content": content}
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": completion_tokens + prompt_tokens,
        },
    }


def translate_reason(request, reason):
    """
    Translate our reason codes to OpenAI ones.
    """
    if reason == "end":
        if request.tool_choice:
            return "tool_calls"
        if request.function_call:
            return "function_call"
        else:
            return "stop"
    elif reason == "max_tokens":
        return "length"
    else:
        return f"error: {reason}"  # Not a standard OpenAI API reason
