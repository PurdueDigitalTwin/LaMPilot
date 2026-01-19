import ast
import re

from langchain_openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage, ChatMessage, BaseMessage
from projects.lampilot.utils.io import load_prompt, load_apis


class CodeGenerationAgent:
    llms = {
        "gpt-3.5-turbo": (ChatOpenAI, "gpt-3.5-turbo"),
        "gpt-4": (ChatOpenAI, "gpt-4"),
        "gpt-4-1106-preview": (ChatOpenAI, "gpt-4-1106-preview"),
        "gpt-5.2": (ChatOpenAI, "gpt-5.2"),
    }

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.0,
                 request_timeout: int = 120,
                 zero_shot: bool = False,
                 ):
        if model_name not in self.llms:
            raise RuntimeError(f"Unknown model name: {model_name}")

        llm, model = self.llms[model_name]

        if llm is ChatOpenAI:
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                request_timeout=request_timeout,
            )
        else:
            raise RuntimeError("Unknown LLM")

        self.zero_shot = zero_shot

    def render_system_message(self):
        system_template = load_prompt(f"cg_template_{'zs' if self.zero_shot else 'fs'}")
        apis = load_apis()
        response_format = load_prompt("cg_response_format")
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        system_message = system_message_prompt.format(
            apis=apis,
            response_format=response_format,
        )
        assert isinstance(system_message, SystemMessage)
        return system_message

    @staticmethod
    def render_human_message(
            command: str = "",
            context_info: str = "",
    ):
        message = ""
        if command == "" or command is None:
            raise RuntimeError("Command is empty.")
        message += f"Command: {command}\n"
        message += f"Context Info: {context_info}\n\n"
        return HumanMessage(content=message)

    @staticmethod
    def process_ai_message(message):
        if isinstance(message, BaseMessage):
            message = message.content
        elif isinstance(message, str):
            pass
        else:
            raise RuntimeError("Unknown message type")

        code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)  # Extract the code from the message
        code = "\n".join(code_pattern.findall(message))

        # Parse the code into an AST
        try:
            parsed_code = ast.parse(code)
        except SyntaxError as e:
            print(f"SyntaxError: {e}")
            return {
                "program_code": "",
                "program_name": "",
                "exec_code": "",
            }

        functions = CodeGenerationAgent.analyze_ast(parsed_code)
        if len(functions) == 0:
            print(f"Warning: No function in the code")
            return {
                "program_code": "",
                "program_name": "",
                "exec_code": "",
            }
        main_function = functions[-1]
        exec_code = f"policy = {main_function['name']}()"

        parsed_message = {
            "program_code": main_function["code"],
            "program_name": main_function["name"],
            "exec_code": exec_code
        }

        return parsed_message

    def reset(self, command: str, context_info: str = ""):
        self.command = command
        self.code = {}
        system_message = self.render_system_message()

        # print(f"\033[32m**** CG Agent system message****\n{system_message.content}\033[0m")
        human_message = self.render_human_message(
            command=command,
            context_info=context_info,
        )
        self.messages = [system_message, human_message]
        print(f"\033[32m****CG Agent human message****\n{human_message.content}\033[0m")
        assert len(self.messages) == 2
        self.conversations = []
        return self.messages

    def step(self):
        if isinstance(self.llm, ChatOpenAI):
            ai_message = self.llm(self.messages)
        else:
            raise RuntimeError("Unknown LLM")

        if isinstance(ai_message, BaseMessage):
            ai_message = ai_message.content
        print(f"\033[34m****CG Agent ai message****\n{ai_message}\033[0m")
        self.conversations.append((self.messages[0].content, self.messages[1].content, ai_message))
        parsed_result = self.process_ai_message(ai_message)
        assert isinstance(parsed_result, dict)
        self.code.update({
            "program_code": parsed_result["program_code"],
            "program_name": parsed_result["program_name"],
        })
        ret_code = {
            'reused_code': "",
            'new_code': parsed_result["program_code"] + "\n" + parsed_result["exec_code"]
        }

        return ret_code

    @staticmethod
    def analyze_ast(node):
        # List to hold information about functions
        functions_info = []

        for subnode in ast.walk(node):
            if isinstance(subnode, ast.FunctionDef):
                func_name = subnode.name
                # noinspection PyTypeChecker
                func_code = ast.unparse(subnode)

                functions_info.append(
                    {
                        "name": func_name,
                        "code": func_code,
                    }
                )

        return functions_info
