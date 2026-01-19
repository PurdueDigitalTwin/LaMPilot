from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from projects.lampilot.utils.io import load_prompt, load_apis, load_primitives
from .cg_agent import CodeGenerationAgent
from .policy_repo import PolicyRepository


class HumanFeedbackCGAgent(CodeGenerationAgent):
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.0,
                 request_timeout: int = 120,
                 policy_repo_model_name: str = "gpt-3.5-turbo",
                 policy_repo_temperature: float = 0.0,
                 policy_repo_retrieval_top_k: int = 3,
                 ckpt_dir: str = "ckpt",
                 resume: bool = False,
                 ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
        )

        self.policy_repo = PolicyRepository(
            model_name=policy_repo_model_name,
            temperature=policy_repo_temperature,
            retrieval_top_k=policy_repo_retrieval_top_k,
            request_timeout=request_timeout,
            ckpt_dir=ckpt_dir,
            resume=resume,
        )

        # init variables
        self.command: str = ""
        self.context_info: str = ""
        self.messages = None
        self.conversations = []
        self.code: dict = {}

    def render_system_message(self, polices=[]):
        system_template = load_prompt(f"cg_template_hf")
        base_policies = [
            "imports",
            "car_following",
            "intersection",
            "overtaking",
        ]
        apis = load_apis()
        programs = "\n\n".join(load_primitives(base_policies) + polices)
        response_format = load_prompt("cg_response_format")
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        system_message = system_message_prompt.format(
            apis=apis,
            programs=programs,
            response_format=response_format,
        )
        assert isinstance(system_message, SystemMessage)
        return system_message

    @staticmethod
    def render_human_message(command: str = "", context_info: str = "", code: str = "", critique: str = ""):
        message = ""
        if code:
            message += f"Code from last round:\n{code}\n\n"
        else:
            message += f"Code from last round: No code in the first round\n\n"
        if command == "" or command is None:
            raise RuntimeError("Command is empty.")
        message += f"Command: {command}\n\n"
        message += f"Context Info: {context_info}\n\n"
        if critique:
            message += f"Feedback: {critique}\n\n"
        else:
            message += f"Feedback: None\n\n"
        return HumanMessage(content=message)

    def reset(self, command: str, context_info: str = ""):
        self.command = command
        self.code = {}

        policies = self.policy_repo.retrieve_policies(query=command)
        print(f"\033[33mRender HF Agent system message with {len(policies)} policies\033[0m")
        system_message = self.render_system_message(polices=policies)
        human_message = self.render_human_message(
            command=command, context_info=context_info, code="", critique=""
        )
        self.messages = [system_message, human_message]
        print(f"\033[32m**** CG Agent human message****\n{human_message.content}\033[0m")
        assert len(self.messages) == 2
        self.conversations = []
        return self.messages

    def step(self):
        ret_code = super().step()
        ret_code.update({'reused_code': self.policy_repo.programs})
        return ret_code

    def receive_feedback(self, success: bool, critique: str, commit: bool = True):
        if success:
            assert (
                    "program_code" in self.code and "program_name" in self.code
            ), "program_code and program_name must be in self.code when success"
            if len(self.code['program_code'].split('\n')) > 2:  # If the program is not a one-liner
                if commit:
                    print(f"\033[32mAdding policy {self.code['program_name']} to policy repo\033[0m")
                    self.policy_repo.add_new_policy(self.code)
            else:
                print(f"\033[33mSkip adding policy {self.code['program_name']} because it is a one-liner\033[0m")
        else:
            new_policies = self.policy_repo.retrieve_policies(query=self.command + "\n\n" + critique)
            system_message = self.render_system_message(polices=new_policies)
            human_message = self.render_human_message(
                command=self.command,
                context_info=self.context_info,
                code=self.code['program_code'],
                critique=critique,
            )
            self.messages = [system_message, human_message]
