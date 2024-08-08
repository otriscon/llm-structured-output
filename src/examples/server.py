# pylint: disable=missing-function-docstring,missing-class-docstring
"""
Example model server with OpenAI-like API, including function calls / tools.
"""
import json
import time
import os
from enum import Enum
from traceback import format_exc
from typing import Literal, List, Optional, Union

from fastapi import FastAPI, Request, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from examples.llm_schema import Model
from llm_structured_output.util.output import info, warning, debug


app = FastAPI()

model = Model()
info("Loading model...")
try:
    model_path = os.environ["MODEL_PATH"]
    model.load(model_path)
except KeyError:
    warning("Need to specify MODEL_PATH environment variable")


@app.exception_handler(RequestValidationError)
# pylint: disable-next=unused-argument
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}"
    warning(f"RequestValidationError: {exc_str}")
    content = {"error": exc_str}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


@app.get("/status")
def get_status():
    return {"status": "OK"}


@app.get("/")
def get_root():
    return FileResponse(f"{os.path.dirname(os.path.realpath(__file__))}/static/ui.html")


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


class V1ToolOptions(BaseModel):  # Non-standard, our addition.
    # We automatically add instructions with the JSON schema
    # for the tool calls to the prompt. This option disables
    # it and is useful when the user prompt already includes
    # the schema and relevant instructions.
    no_prompt_steering: bool = False


class V1ResponseFormatType(str, Enum):
    JSON_OBJECT = "json_object"


class V1ResponseFormat(BaseModel):
    type: V1ResponseFormatType
    # schema is our addition, not an OpenAI API parameter
    schema: str = None


class V1StreamOptions(BaseModel):
    include_usage: bool = False


class V1ChatCompletionsRequest(
    BaseModel
):  # pylint: disable=too-many-instance-attributes
    model: str = "default"
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
    tool_options: V1ToolOptions = None
    response_format: V1ResponseFormat = None
    stream: bool = False
    stream_options: V1StreamOptions = None


@app.post("/v1/chat/completions")
async def post_v1_chat_completions(request: V1ChatCompletionsRequest):
    debug("REQUEST", request)
    if request.stream:
        async def get_content():
            try:
                async for message in post_v1_chat_completions_impl(request):
                    yield message
            # pylint: disable-next=broad-exception-caught
            except Exception as e:
                warning(format_exc())
                yield 'data: {"choices": [{"index": 0, "finish_reason": "error: ' + str(e) + '"}]}'
        return StreamingResponse(
            content=get_content(),
            media_type="text/event-stream",
        )
    else:
        # FUTURE: Python 3.10 can use `await anext(x))` instead of `await x.__anext__()`.
        try:
            response = await post_v1_chat_completions_impl(request).__anext__()
        # pylint: disable-next=broad-exception-caught
        except Exception as e:
            warning(format_exc())
            content = {"error": str(e)}
            response = JSONResponse(
                content=content, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        debug("RESPONSE", response)
        return response


async def post_v1_chat_completions_impl(request: V1ChatCompletionsRequest):
    messages = request.messages[:]

    # Extract valid functions from the request.
    functions = []
    is_legacy_function_call = False
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
        is_legacy_function_call = True
    elif request.function_call is not None:
        functions = [
            next(
                fn for fn in request.functions if fn.name == request.function_call.name
            )
        ]
        is_legacy_function_call = True

    model_name = model_path
    schema = None
    if functions:
        # If the request includes functions, create a system prompt to instruct the LLM
        # to use tools, and assemble a JSON schema to steer the LLM output.
        if request.stream:
            responder = ToolCallStreamingResponder(
                model_name,
                functions,
                is_legacy_function_call,
                model,
            )
        else:
            responder = ToolCallResponder(
                model_name, functions, is_legacy_function_call
            )
        if not (request.tool_options and request.tool_options.no_prompt_steering):
            messages.insert(
                0,
                V1ChatMessage(
                    role="system",
                    content=responder.tool_prompt,
                ),
            )
        schema = responder.schema
    else:
        if request.response_format:
            assert request.response_format.type == V1ResponseFormatType.JSON_OBJECT
            # The request may specify a JSON schema (this option is not in the OpenAI API)
            if request.response_format.schema:
                schema = json.loads(request.response_format.schema)
            else:
                schema = {"type": "object"}
        if request.stream:
            responder = ChatCompletionStreamingResponder(model_name, schema, model)
        else:
            responder = ChatCompletionResponder(model_name)

    if schema is not None:
        debug("Using schema:", schema)

    info("Starting generation...")

    prompt_tokens = None

    for result in model.completion(
        messages,
        schema=schema,
        max_tokens=request.max_tokens,
        temp=request.temperature,
        cache_prompt=True,
    ):
        if result["op"] == "evaluatedPrompt":
            prompt_tokens = result["token_count"]
        elif result["op"] == "generatedTokens":
            message = responder.generated_tokens(result["text"])
            if message:
                yield message
        elif result["op"] == "stop":
            completion_tokens = result["token_count"]
            yield responder.generation_stopped(
                result["reason"], prompt_tokens, completion_tokens
            )
        else:
            assert False


class ChatCompletionResponder:
    def __init__(self, model_name: str):
        self.object_type = "chat.completion"
        self.model_name = model_name
        self.created = int(time.time())
        self.id = f"{id(self)}_{self.created}"
        self.content = ""

    def message_properties(self):
        return {
            "object": self.object_type,
            "id": f"chatcmpl-{self.id}",
            "created": self.created,
            "model": self.model_name,
        }

    def translate_reason(self, reason):
        """
        Translate our reason codes to OpenAI ones.
        """
        if reason == "end":
            return "stop"
        if reason == "max_tokens":
            return "length"
        return f"error: {reason}"  # Not a standard OpenAI API reason

    def format_usage(self, prompt_tokens: int, completion_tokens: int):
        return {
            "usage": {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": completion_tokens + prompt_tokens,
            },
        }

    def generated_tokens(
        self,
        text: str,
    ):
        self.content += text
        return None

    def generation_stopped(
        self,
        stop_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
    ):
        finish_reason = self.translate_reason(stop_reason)
        message = {"role": "assistant", "content": self.content}
        return {
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish_reason}
            ],
            **self.format_usage(prompt_tokens, completion_tokens),
            **self.message_properties(),
        }


class ChatCompletionStreamingResponder(ChatCompletionResponder):
    def __init__(self, model_name: str, schema: dict = None, _model = None):
        super().__init__(model_name)
        self.object_type = "chat.completion.chunk"
        if schema:
            assert _model
            self.schema_parser = _model.get_driver_for_json_schema(schema)
        else:
            self.schema_parser = None

    def generated_tokens(
        self,
        text: str,
    ):
        delta = {"role": "assistant", "content": text}
        if self.schema_parser:
            values = {}
            for char in text:
                self.schema_parser.advance_char(char)
                for path in self.schema_parser.get_current_value_paths():
                    values[path] = values.get(path, "") + char
            delta["values"] = values
        message = {
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            **self.message_properties(),
        }
        return f"data: {json.dumps(message)}\n"

    def generation_stopped(
        self,
        stop_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
    ):
        finish_reason = self.translate_reason(stop_reason)
        delta = {"role": "assistant", "content": ""}
        message = {
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            # Usage field notes:
            # - OpenAI only sends usage in streaming if the option
            #   stream_options.include_usage is true, but we send it always.
            **self.format_usage(prompt_tokens, completion_tokens),
            **self.message_properties(),
        }
        return f"data: {json.dumps(message)}\ndata: [DONE]\n"


class ToolCallResponder(ChatCompletionResponder):
    def __init__(
        self, model_name: str, functions: list[dict], is_legacy_function_call: bool
    ):
        super().__init__(model_name)

        self.is_legacy_function_call = is_legacy_function_call

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
        if len(function_schemas) == 1:
            self.schema = function_schemas[0]
            self.tool_prompt = self._one_tool_prompt(functions[0], function_schemas[0])
        elif is_legacy_function_call:  # Only allows one function to be called.
            self.schema = {"oneOf": function_schemas}
            self.tool_prompt = self._select_tool_prompt(functions, function_schemas)
        else:
            self.schema = {"type": "array", "items": {"anyOf": function_schemas}}
            self.tool_prompt = self._multiple_tool_prompt(functions, function_schemas)

    def translate_reason(self, reason):
        if reason == "end":
            if self.is_legacy_function_call:
                return "function_call"
            return "tool_calls"
        return super().translate_reason(reason)

    def generation_stopped(
        self,
        stop_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
    ):
        finish_reason = self.translate_reason(stop_reason)
        if finish_reason == "tool_calls":
            tool_calls = json.loads(self.content)
            if not isinstance(tool_calls, list):
                # len(functions) == 1 was special cased
                tool_calls = [tool_calls]
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{self.id}_{i}",
                        "type": "function",
                        "function": {
                            "name": function_call["name"],
                            "arguments": json.dumps(function_call["arguments"]),
                        },
                    }
                    for i, function_call in enumerate(tool_calls)
                ],
            }
        elif finish_reason == "function_call":
            function_call = json.loads(self.content)
            message = {
                "role": "assistant",
                "function_call": {
                    "name": function_call["name"],
                    "arguments": json.dumps(function_call["arguments"]),
                },
            }
        else:
            message = None
        return {
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish_reason}
            ],
            **self.format_usage(prompt_tokens, completion_tokens),
            **self.message_properties(),
        }

    def _one_tool_prompt(self, tool, tool_schema):
        return f"""
You are a helpful assistant with access to a tool that you must invoke to answer the user's request.
The tool is:
Tool {tool.name}: {tool.description}
Invocation schema: {json.dumps(tool_schema)}
Your answer is a JSON object according to the invocation schema in order to answer the user request below.
"""

    def _multiple_tool_prompt(self, tools, tool_schemas, separator="\n"):
        return f"""
You are a helpful assistant with access to tools that you must invoke to answer the user's request.
The following tools are available:
{separator.join([ f'''
Tool {tool.name}: {tool.description}
Invocation schema: {json.dumps(tool_schema)}
''' for tool, tool_schema in zip(tools, tool_schemas) ])}
Your answer is a JSON array with one or more tool invocations according to the appropriate schema(s)
in order to answer the user request below.
"""

    def _select_tool_prompt(self, tools, tool_schemas, separator="\n"):
        return f"""
You are a helpful assistant with access to tools that you must invoke to answer the user's request.
The following tools are available:
{separator.join([ f'''
Function {tool.name}: {tool.description}
Tool schema: {json.dumps(tool_schema)}
''' for tool, tool_schema in zip(tools, tool_schemas) ])}
Your answer is a JSON object according to the invocation schema of the most appropriate tool to use
to answer the user request below.
"""


class ToolCallStreamingResponder(ToolCallResponder):
    def __init__(
        self,
        model_name: str,
        functions: list[dict],
        is_legacy_function_call: bool,
        _model,
    ):
        super().__init__(model_name, functions, is_legacy_function_call)
        self.object_type = "chat.completion.chunk"

        # We need to parse the output as it's being generated in order to send
        # streaming messages that contain the name and arguments of the function
        # being called.

        self.current_function_index = -1
        self.current_function_name = None
        self.in_function_arguments = False

        def set_function_name(_prop_name: str, prop_value):
            self.current_function_index += 1
            self.current_function_name = prop_value

        def start_function_arguments(_prop_name: str):
            self.in_function_arguments = True

        def end_function_arguments(_prop_name: str, _prop_value: str):
            self.in_function_arguments = False

        hooked_function_schemas = [
            {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "const",
                        "const": fn.name,
                        "__hooks": {
                            "value_end": set_function_name,
                        },
                    },
                    "arguments": {
                        **fn.parameters,
                        "__hooks": {
                            "value_start": start_function_arguments,
                            "value_end": end_function_arguments,
                        },
                    },
                },
                "required": ["name", "arguments"],
            }
            for fn in functions
        ]
        if len(hooked_function_schemas) == 1:
            hooked_schema = hooked_function_schemas[0]
        elif is_legacy_function_call:
            hooked_schema = {"oneOf": hooked_function_schemas}
        else:
            hooked_schema = {
                "type": "array",
                "items": {"anyOf": hooked_function_schemas},
            }
        self.tool_call_parser = _model.get_driver_for_json_schema(hooked_schema)

    def generated_tokens(
        self,
        text: str,
    ):
        argument_text = ""
        for char in text:
            if self.in_function_arguments:
                argument_text += char
            # Update state. This is certain to parse, no need to check for rejections.
            self.tool_call_parser.advance_char(char)
        if not argument_text:
            return None
        assert self.current_function_name
        if self.is_legacy_function_call:
            delta = {
                "function_call": {
                    "name": self.current_function_name,
                    "arguments": argument_text,
                }
            }
        else:
            delta = {
                "tool_calls": [
                    {
                        "index": self.current_function_index,
                        "id": f"call_{self.id}_{self.current_function_index}",
                        "type": "function",
                        "function": {
                            # We send the name on every update, but OpenAI only sends it on
                            # the first one for each call, with empty arguments (""). Further
                            # updates only have the arguments field. This is something we may
                            # want to emulate if client code depends on this behavior.
                            "name": self.current_function_name,
                            "arguments": argument_text,
                        },
                    }
                ]
            }
        message = {
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            **self.message_properties(),
        }
        return f"data: {json.dumps(message)}\n"

    def generation_stopped(
        self,
        stop_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
    ):
        finish_reason = self.translate_reason(stop_reason)
        message = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            # Usage field notes:
            # - OpenAI only sends usage in streaming if the option
            #   stream_options.include_usage is true, but we send it always.
            # - OpenAI sends two separate messages: one with the finish_reason and no
            #   usage field, and one with an empty choices array and the usage field.
            **self.format_usage(prompt_tokens, completion_tokens),
            **self.message_properties(),
        }
        return f"data: {json.dumps(message)}\ndata: [DONE]\n"
