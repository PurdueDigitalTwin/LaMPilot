# Adapted from: https://github.com/MineDojo/Voyager/blob/main/voyager/agents/skill.py

import json
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma

from projects.lampilot.utils.io import load_primitives, load_json, load_prompt


class PolicyRepository:
    def __init__(self,
                 model_name="gpt-3.5-turbo",
                 temperature=0,
                 retrieval_top_k=5,
                 request_timeout=120,
                 ckpt_dir="ckpt",
                 resume=False,
                 ):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
        )
        os.makedirs(f"{ckpt_dir}/policy/code", exist_ok=True)
        os.makedirs(f"{ckpt_dir}/policy/description", exist_ok=True)
        os.makedirs(f"{ckpt_dir}/policy/vectordb", exist_ok=True)

        self.primitives = load_primitives()
        if resume:
            print(f"\033[33mLoading Policy Repository from {ckpt_dir}/policy\033[0m")
            self.policies = load_json(f"{ckpt_dir}/policy/policies.json")
        else:
            self.policies = {}
        self.retrieve_top_k = retrieval_top_k
        self.ckpt_dir = ckpt_dir
        self.vectordb = Chroma(
            collection_name="policy_vectordb",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=f"{ckpt_dir}/policy/vectordb",
        )
        assert self.vectordb._collection.count() == len(self.policies), (
            f"Policy Repository's vectordb is not synced with policies.json.\n"
            f"There are {self.vectordb._collection.count()} policies in vectordb but {len(self.policies)} policies in policies.json.\n"
            f"Did you set resume=False when initializing the repository?\n"
            f"You may need to manually delete the vectordb directory for running from scratch."
        )

    @property
    def programs(self):
        programs = ""
        for policy_name, entry in self.policies.items():
            programs += f"{entry['code']}\n\n"
        for primitives in self.primitives:
            programs += f"{primitives}\n\n"
        return programs

    def add_new_policy(self, info):
        program_name = info["program_name"]
        program_code = info["program_code"]
        policy_description = self.generate_policy_description(program_name, program_code)
        print(f"\033[33mPolicy Repository generated description for {program_name}:\n{policy_description}\033[0m")
        if program_name in self.policies:
            print(f"\033[33mPolicy {program_name} already exists. Rewriting!\033[0m")
            self.vectordb._collection.delete(ids=[program_name])
            version = 2
            while f"{program_name}V{version}.py" in os.listdir(f"{self.ckpt_dir}/policy/code"):
                version += 1
            dumped_program_name = f"{program_name}V{version}"
        else:
            dumped_program_name = program_name

        self.vectordb.add_texts(
            texts=[policy_description],
            ids=[program_name],
            metadatas=[{"name": program_name}],
        )

        self.policies[program_name] = {
            "code": program_code,
            "description": policy_description,
        }
        assert self.vectordb._collection.count() == len(
            self.policies), "Policy Repository and VectorDB are not synchronized!"

        with open(f"{self.ckpt_dir}/policy/code/{dumped_program_name}.py", "w") as f:
            f.write(program_code)

        with open(f"{self.ckpt_dir}/policy/description/{dumped_program_name}.txt", "w") as f:
            f.write(policy_description)

        with open(f"{self.ckpt_dir}/policy/policies.json", "w") as f:
            json.dump(self.policies, f)

    def generate_policy_description(self, program_name, program_code):
        messages = [
            SystemMessage(content=load_prompt("policy")),
            HumanMessage(content=program_code + "\n\n" + f"The main function is `{program_name}`."),
        ]
        policy_description = f"{self.llm(messages).content}"
        return f"def {program_name}(ego):\n    {policy_description}"

    def retrieve_policies(self, query):
        k = min(self.vectordb._collection.count(), self.retrieve_top_k)
        if k == 0:
            return []
        print(f"\033[33mPolicy Repository retrieving for {k} policies\033[0m")
        docs_and_scores = self.vectordb.similarity_search_with_score(query, k=k)
        print(
            f"\033[33mPolicy Repository retrieved policies: "
            f"{', '.join([doc.metadata['name'] for doc, _ in docs_and_scores])}\033[0m"
        )
        policies = []
        for doc, _ in docs_and_scores:
            policies.append(self.policies[doc.metadata["name"]]["code"])
        return policies
