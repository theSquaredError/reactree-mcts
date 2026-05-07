
import torch
# from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import os
from huggingface_hub import login, whoami
# import cfg

class LlmAgent:
    def __init__(self, hf_token=None, model_name='meta-llama/Llama-3.1-8B-Instruct'):
        self.model_name = model_name
        if hf_token:
            # print(f'hf_token: {hf_token}')
            login(hf_token)
            info = whoami()
            # print(f'logged in as: {info}')

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                          torch_dtype=torch.float16,
                                                          device_map='auto')
        self.messages = []   # conversation history
        self._seed = 42
        

    
    def chat(self, messages, max_new_tokens=30, temperature=0, seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
        ).to(self.model.device)
        attention_mask = torch.ones_like(inputs, device=self.model.device)

        generate_kwargs = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["do_sample"] = True
        outputs = self.model.generate(
            inputs,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs,
        )
        new_tokens = outputs[0][inputs.shape[-1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if response.lower().startswith("assistant"):
            response = response[len("assistant"):].lstrip(" \n\r\t:")
        return response


    def build_messages(self, task_desc, init_obs, admissible_commands=None):
        from src.rm.rm_prompt import REACTREE_PROMPT
        system = REACTREE_PROMPT
        user_content = f"Goal: {task_desc}\nObservation: {init_obs}"
        if admissible_commands:
            cmds = ', '.join(admissible_commands) if isinstance(admissible_commands, list) else admissible_commands
            user_content += f"\nAvailable actions: {cmds}"
        user_content += "\n\nExpand the current goal into subgoals."
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def first_expand(self, task_desc, init_obs, admissible_commands=None, seed=42):
        messages = self.build_messages(task_desc, init_obs, admissible_commands)
        return self.chat(messages, max_new_tokens=200, temperature=0, seed=seed)

    # ------------------------------------------------------------------ #
    # Chat-style solving: maintains self.messages across multiple steps   #
    # ------------------------------------------------------------------ #

    def reset_chat(self, goal, init_obs, admissible_commands=None, context=None, seed=42):
        """Start a fresh conversation for a root/initial goal."""
        from src.rm.rm_prompt import REACTREE_PROMPT
        parts = []
        parts.append(f"Your current task is to: {goal}")
        parts.append(f"Current observation: {init_obs}")
        if admissible_commands:
            cmds = ', '.join(admissible_commands) if isinstance(admissible_commands, list) else admissible_commands
            parts.append(f"Available actions: {cmds}")
        if context:
            parts.append(context)
        self.messages = [
            {"role": "system", "content": REACTREE_PROMPT},
            {"role": "user",   "content": "\n".join(parts)},
        ]
        self._seed = seed

    def append_subgoal(self, goal, init_obs, admissible_commands=None, context=None, prior_messages=None):
        """Continue from parent's history — no reset. Restores prior conversation
        and appends a new user message with the current subgoal and observation."""
        parts = []
        if context:
            parts.append(context)
        parts.append(f"Current observation: {init_obs}")
        parts.append(f"Your current subgoal is to: {goal}")
        if admissible_commands:
            cmds = ', '.join(admissible_commands) if isinstance(admissible_commands, list) else admissible_commands
            parts.append(f"Available actions: {cmds}")
        self.messages = list(prior_messages) + [{"role": "user", "content": "\n".join(parts)}]

    def add_observation(self, obs_text):
        """Append environment feedback to conversation history."""
        self.messages.append({"role": "user", "content": f"Observation: {obs_text}"})

    def plan_step(self):
        """Ask LLM for the next decision using full conversation history."""
        import logging
        log = logging.getLogger(__name__)

        log.info("---- LLM INPUT (%d messages) ----", len(self.messages))
        for i, msg in enumerate(self.messages):
            log.info("[%d] role=%s | %s", i, msg['role'], msg['content'])
        log.info("---------------------------------")

        response = self.chat(self.messages, max_new_tokens=100, temperature=0, seed=self._seed)
        self.messages.append({"role": "assistant", "content": response})

        log.info("---- LLM OUTPUT ----")
        log.info("%s", response)
        log.info("--------------------")

        return self._parse_step_response(response)

    @staticmethod
    def _parse_step_response(response):
        import re
        r = response.strip()
        # Match "Think:" or "Think\n" (colon optional — LLMs sometimes drop it)
        if re.match(r'Think[:\s]', r, re.IGNORECASE):
            content = re.sub(r'^Think[:\s]*', '', r, flags=re.IGNORECASE).strip()
            return {'type': 'Think', 'content': content}
        if re.match(r'Act[:\s]', r, re.IGNORECASE):
            content = re.sub(r'^Act[:\s]*', '', r, flags=re.IGNORECASE).strip()
            return {'type': 'Act', 'content': content}
        if re.match(r'Expand[:\s]', r, re.IGNORECASE):
            return {'type': 'Expand', 'content': LlmAgent.parse_expand_response(r)}
        return {'type': 'Unknown', 'content': r}

    @staticmethod
    def parse_expand_response(response):
        import re

        # Extract "- control flow: sequence"
        cf_match = re.search(r'-\s*control flow:\s*(sequence|fallback|parallel)', response, re.IGNORECASE)
        control_flow = cf_match.group(1).lower() if cf_match else 'sequence'

        # Primary: "- subgoals: a, b, c" → split on commas
        sg_match = re.search(r'-\s*subgoals:\s*(.+)', response)
        if sg_match:
            subgoals = [s.strip() for s in sg_match.group(1).split(',') if s.strip()]
        else:
            # Fallback: "Subgoal N: text" lines
            subgoals = re.findall(r'Subgoal\s+\d+[:\.\s]+(.+)', response)
            # Fallback: numbered list "1. Title"
            if not subgoals:
                subgoals = re.findall(r'^\d+\.\s+\*{0,2}([^*:\n]+)\*{0,2}', response, re.MULTILINE)
            subgoals = [s.strip().rstrip('.') for s in subgoals if s.strip()]

        return {'control_flow': control_flow, 'conditions': subgoals}

    # def reactexpand_plan_next_step(self, skill_set):
    #     self.llm += guidance.select(['Act: ', 'Think: ', 'Expand:\n'], name='choice')
    #     if self.llm['choice'] == 'Think: ':
    #         self.llm += guidance.gen(stop='\n', name='reasoning', max_tokens=200, temperature=0) + '\nOK.\n'
    #         next_step_info = {'next_step_class': 'Think', 'next_step': self.llm['reasoning']}
    #     elif self.llm['choice'] == 'Act: ':
    #         self.llm += guidance.select(skill_set, name='nl_skill') + '\n'
    #         next_step_info = {'next_step_class': 'Act', 'next_step': self.llm['nl_skill']}
    #     elif self.llm['choice'] == 'Expand:\n':
    #         self.llm += '- control flow: ' + guidance.select(['sequence', 'fallback', 'parallel'], name='control_flow') + '\n- subgoals: ' + guidance.gen(stop='\n', name='conditions', max_tokens=200, temperature=0) + '\nOK.\n'
    #         next_step_info = {'next_step_class': 'Expand', 'next_step': {'control_flow': self.llm['control_flow'], 'conditions': self.llm['conditions']}}

    #     return next_step_info
    

    def load_prompt(self, nl_inst, init_obs):
        raise NotImplementedError()
    
    # def answer_question(self, question):
    #     self.llm += f'{question}\nAnswer: ' + guidance.gen(name='answer', max_tokens=200)
    #     return self.llm['answer']
