import os
import re
import numpy as np
from typing import Dict, Any, List, Sequence, Optional, TYPE_CHECKING
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from copy import deepcopy

if TYPE_CHECKING:
    from PIL.Image import Image

def process_topk_tokens(topk_tokens_list):
    """Extract action candidates from top-k token probabilities"""
    action_candidates = {}
    cum_prob = 1
    action_str = ''
    
    for tokens in topk_tokens_list[:-2]:
        cum_prob *= tokens[0][1]
        action_str += tokens[0][0]
    
    last_tokens = topk_tokens_list[-2]
    
    for token, prob in last_tokens:
        action_candidate = (action_str + token).replace('Ġ', ' ')
        
        # Clean up action formatting
        pattern = r'\baction\s+\d+\s+is\s+'
        action_candidate = re.sub(pattern, '', action_candidate)
        pattern = r'\bAction\s+\d+\s+is\s+'
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


class DirectHuggingfaceModel:
    """Direct HuggingFace transformer-based LLM wrapper"""

    def __init__(self, model_name: Optional[str] = None) -> None:
        if model_name is None:
            # model_name = os.path.expanduser(
            #     "~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
            # )
            model_name = 'meta-llama/Llama-3.1-8B-Instruct'
        self.token = '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'
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token = self.token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token = self.token
        )
        self.model.eval()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def _extract_action_from_text(self, text: str) -> str:
        t = text.strip()

        # Prefer explicit final answer forms
        patterns = [
            r"final answer is:\s*([^\n\r\.]+)",
            r"answer:\s*([^\n\r\.]+)",
            r"action:\s*([^\n\r\.]+)",
        ]
        for p in patterns:
            m = re.search(p, t, flags=re.IGNORECASE)
            if m:
                return self._clean_action_candidate(m.group(1))

        # fallback: first non-empty line
        for line in t.splitlines():
            line = line.strip()
            if line:
                return self._clean_action_candidate(line)

        return ""

    def _match_to_candidates(self, pred: str, candidates: List[str]) -> Optional[str]:
        if not pred:
            return None

        pred_l = pred.lower().strip()

        # exact
        for c in candidates:
            if pred_l == c.lower().strip():
                return c

        # prefix
        starts = [c for c in candidates if c.lower().startswith(pred_l)]
        if len(starts) == 1:
            return starts[0]

        # contains
        contain = [c for c in candidates if pred_l in c.lower()]
        if len(contain) == 1:
            return contain[0]

        return None

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        **kwargs
    ) -> Dict[str, float]:
        prompt = self._build_prompt(messages, system)
        self.logger.info(f'[LLM] prompt after build_prompt: {prompt}')
        # prompt = list(messages) # your [{'role':..., 'content':...}, ...]
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 150),
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )

        generated_ids = outputs.sequences[0][input_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        # self.logger.info(f"[LLM] Prompt (last 300 chars): ...{prompt[-300:]}")
        self.logger.info(f"[LLM] Generated token IDs: {generated_ids.tolist()}")
        self.logger.info(f"[LLM] Generated text: {repr(generated_text)}")
        self.logger.info(f"[LLM] Number of score steps: {len(outputs.scores)}")

        candidates = kwargs.get("candidates", None)
        parsed = self._extract_action_from_text(generated_text)

        if candidates:
            matched = self._match_to_candidates(parsed, candidates)
            if matched:
                return {matched: 1.0}

        # fallback to old token-prob extraction
        action_candidates = self._extract_action_probs(
            outputs=outputs,
            input_len=input_len,
            top_k=kwargs.get("top_k", 5),
        )
        self.logger.info(f"[LLM] action_candidates(full): {action_candidates}")
        return action_candidates

    def _build_prompt(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None
    ) -> str:
        msgs = deepcopy(list(messages))

        if msgs and msgs[0].get("role") == "system" and len(msgs) > 1:
            msgs[1]["content"] = "System prompt:" + msgs[0]["content"] + "\n" + msgs[1]["content"]
            msgs = msgs[1:]

        # Optional explicit `system` arg support.
        if system:
            if msgs and msgs[0].get("role") == "user":
                msgs[0]["content"] = "System prompt:" + system + "\n" + msgs[0]["content"]
            else:
                msgs = [{"role": "user", "content": "System prompt:" + system}] + msgs

        prompt = ""
        for msg in msgs:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            prompt += f"{role}: {content}\n"

        prompt += "Assistant:"
        return prompt

    def _clean_action_candidate(self, text: str) -> str:
        text = re.sub(r"\baction\s+\d+\s+is\s+", "", text)
        text = re.sub(r"\bAction\s+\d+\s+is\s+", "", text)
        text = re.sub(r"^action\s+\d+:\s+", "", text)
        text = re.sub(r"^Action\s+\d+:\s+", "", text)
        text = re.sub(r"\baction\s+", "", text)
        text = text.replace("action:", "").replace(">", "").replace(".", "").strip()
        return text

    def _extract_action_probs(
        self,
        outputs,
        input_len: int,
        top_k: int = 5
    ) -> Dict[str, float]:
        generated_ids = outputs.sequences[0][input_len:]
        scores = outputs.scores

        if generated_ids.numel() == 0 or not scores:
            return {}

        # Prefix is all generated tokens except final token
        prefix_ids = generated_ids[:-1]
        prefix_text = self.tokenizer.decode(prefix_ids, skip_special_tokens=True)

        # Sum log-probs for generated prefix tokens
        prefix_logprob = 0.0
        if len(generated_ids) > 1:
            for step in range(len(generated_ids) - 1):
                step_logits = scores[step][0]
                step_logprobs = torch.log_softmax(step_logits, dim=-1)
                chosen_token_id = int(generated_ids[step].item())
                prefix_logprob += float(step_logprobs[chosen_token_id].item())

        # Top-k alternatives for the final token
        final_step = len(generated_ids) - 1
        final_logits = scores[final_step][0]
        final_logprobs = torch.log_softmax(final_logits, dim=-1)

        k = min(int(top_k), final_logits.shape[-1])
        top_vals, top_ids = torch.topk(final_logprobs, k=k)

        raw_candidates: Dict[str, float] = {}
        for logp, tok_id in zip(top_vals.tolist(), top_ids.tolist()):
            token_text = self.tokenizer.decode([tok_id], skip_special_tokens=True)
            action_text = self._clean_action_candidate(prefix_text + token_text)
            if not action_text:
                continue

            prob = float(np.exp(prefix_logprob + logp))
            raw_candidates[action_text] = prob

        # Normalize
        total = sum(raw_candidates.values())
        if total <= 0:
            return {}

        action_candidates = {k: v / total for k, v in raw_candidates.items()}
        return dict(sorted(action_candidates.items(), key=lambda x: x[1], reverse=True))


class HuggingfaceChatModel:
    """Alias for backward compatibility"""
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        model_name = None
        if args and "model_name_or_path" in args:
            model_name = os.path.expanduser(args["model_name_or_path"])
        
        self.engine = DirectHuggingfaceModel(model_name)
    
    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        **kwargs
    ) -> Dict[str, float]:
        return self.engine.chat(messages, system, **kwargs)


