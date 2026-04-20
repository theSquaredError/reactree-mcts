# Copyright 2024 THUDM and the LlamaFactory team.
#
# This code is inspired by the THUDM's ChatGLM implementation.
# https://github.com/THUDM/ChatGLM-6B/blob/main/cli_demo.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import openai
import re
import numpy as np
from threading import Thread
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence

from llamafactory.extras.misc import torch_gc
from llamafactory.hparams import get_infer_args
from llamafactory.chat.hf_engine import HuggingfaceEngine
from llamafactory.chat.vllm_engine import VllmEngine


if TYPE_CHECKING:
    from PIL.Image import Image

    from llamafactory.chat.base_engine import BaseEngine, Response


def _start_background_loop(loop: "asyncio.AbstractEventLoop") -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()

def process_topk_tokens(topk_tokens_list):
    # 最后一个token是EOS，不考虑
    action_candidates = {}
    cum_prob = 1
    action_str = ''
    for tokens in topk_tokens_list[:-2]:
        cum_prob *= tokens[0][1]
        action_str += tokens[0][0]
    last_tokens = topk_tokens_list[-2]

    action_candidates = {}
    for token,prob in last_tokens:
        action_candidate = (action_str+token).replace('Ġ', ' ')


        pattern = r'\baction\s+\d+\s+is\s+' # 去掉action num is
        action_candidate = re.sub(pattern, '', action_candidate)
        pattern = r'\bAction\s+\d+\s+is\s+' # 去掉Action num is
        action_candidate = re.sub(pattern, '', action_candidate)
        pattern = r'^action\s+\d+:\s+'
        action_candidate = re.sub(pattern, '', action_candidate)
        pattern = r'^Action\s+\d+:\s+'
        action_candidate = re.sub(pattern, '', action_candidate)
        pattern = r'\baction\s+'
        action_candidate = re.sub(pattern, '', action_candidate)


        if "action:" in action_candidate or ">" in action_candidate or '.' in action_candidate:
            action_candidate = action_candidate.replace("action:", "").replace(">", "").replace('.','').strip()

        action_candidates[action_candidate] = cum_prob * prob
    return action_candidates
                

class LlaMaChatModel:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        import sys
        if "--model_name_or_path" not in sys.argv:  # 如果没有提供model_name_or_path参数
            new_args = {
                "model_name_or_path": "/mnt/tangyehui/model/llama_31_8b_instruct/",
                "adapter_name_or_path": "/mnt/tangyehui/code/LLaMA-Factory/saves/llama3-8b/lora/sft_20240914_132522/",
                # "adapter_name_or_path": "/mnt/public/tangyehui/code/LLaMA-Factory/saves/llama3-8b/lora/dpo_20241014_165346/",
                "finetuning_type": "lora",
                "template": "llama3",
            }
        else:
            new_args = args
        model_args, data_args, finetuning_args, generating_args = get_infer_args(new_args)
        if model_args.infer_backend == "huggingface":
            self.engine: "BaseEngine" = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)
        elif model_args.infer_backend == "vllm":
            self.engine: "BaseEngine" = VllmEngine(model_args, data_args, finetuning_args, generating_args)
        else:
            raise NotImplementedError("Unknown backend: {}".format(model_args.infer_backend))

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["Image"] = None,
        **input_kwargs,
    ) -> Dict[str, float]:
        task = asyncio.run_coroutine_threadsafe(self.achat(messages, system, tools, image, **input_kwargs), self._loop)
        response = task.result()
        topk_tokens_list = response[-1].topk_tokens_list
        action_candidates = process_topk_tokens(topk_tokens_list)
        return action_candidates

    async def achat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["Image"] = None,
        **input_kwargs,
    ) -> List["Response"]:
        if messages[0]["role"] == "system":
            messages[1]["content"] = "System prompt:" + messages[0]["content"] + "\n" + messages[1]["content"]
            messages = messages[1:]
        return await self.engine.chat(messages, system, tools, image, **input_kwargs)

    def stream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["Image"] = None,
        **input_kwargs,
    ) -> Generator[str, None, None]:
        generator = self.astream_chat(messages, system, tools, image, **input_kwargs)
        while True:
            try:
                task = asyncio.run_coroutine_threadsafe(generator.__anext__(), self._loop)
                yield task.result()
            except StopAsyncIteration:
                break

    async def astream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["Image"] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        if messages[0]["role"] == "system":
            messages[1]["content"] = "system prompt:" + messages[0]["content"] + "\n" + messages[1]["content"]
            messages = messages[1:]
        async for new_token in self.engine.stream_chat(messages, system, tools, image, **input_kwargs):
            yield new_token

    def get_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        task = asyncio.run_coroutine_threadsafe(self.aget_scores(batch_input, **input_kwargs), self._loop)
        return task.result()

    async def aget_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        return await self.engine.get_scores(batch_input, **input_kwargs)

class GPT4Engine:
    def __init__(self):
        super().__init__()
        openai.api_key = "sk-xxx" # change if you need
        openai.base_url = "https://api.chatanywhere.tech"
        self.model = "gpt-4-turbo-preview"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                logprobs=True,
                top_logprobs=5,
            )
            action_str = ''
            prob_chain = response.choices[0].logprobs.content
            prob = 0
            for prob_item in prob_chain[:-1]:
                action_str += prob_item.token
                prob += prob_item.logprob
            
            # 基于最后一个token的概率预测分布
            final_prob = prob_chain[-1].top_logprobs
            action_candidates = {}
            for final_item in final_prob:
                action_candidate = action_str+final_item.token

                pattern = r'\baction\s+\d+\s+is\s+' # 去掉action num is
                action_candidate = re.sub(pattern, '', action_candidate)
                pattern = r'\bAction\s+\d+\s+is\s+' # 去掉Action num is
                action_candidate = re.sub(pattern, '', action_candidate)
                pattern = r'^action\s+\d+:\s+'
                action_candidate = re.sub(pattern, '', action_candidate)
                pattern = r'^Action\s+\d+:\s+'
                action_candidate = re.sub(pattern, '', action_candidate)
                pattern = r'\baction\s+'
                action_candidate = re.sub(pattern, '', action_candidate)

                if "action:" in action_candidate or ">" in action_candidate:
                    action_candidate = action_candidate.replace("action:", "").replace(">", "").strip()
                action_candidates[action_candidate] = np.exp(prob+final_item.logprob)
            action = response.choices[0].message.content.strip().strip("'")
            return action_candidates
        except Exception as e:
            print(f"Error in GPT4Engine chat: {e}")
            return ""