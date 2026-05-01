
import torch
# from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import os
# import cfg

class LlmAgent:
    def __init__(self, hf_token):
        # self.cfg = cfg
        self.model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
        # self.agent_type = cfg.llm_agent.agent_type
        # self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        # if cfg.llm_agent.ic_ex_select_type == 'rerank':
        #     self.rerank_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
        #     self.rerank_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token = hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                          torch_dtype=torch.float16,
                                                          device_map='auto', token= hf_token)
        

    
    def chat(self,messages, max_new_tokens=256, temperature=0):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            return_tensors ="pt",
        ).to(self.model.device)

        outputs = self.model.generate(inputs,
                                      max_new_tokens = max_new_tokens,
                                      temperature=temperature,
                                      do_sample=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokense=True)
    
    def plan_next_step(self, skill_set):
        next_step_info = self.reactexpand_plan_next_step(skill_set)
        return next_step_info

    def load_reactree_prompt(self, path=None):
        try:
            from rm.rm_prompt import REACTREE_PROMPT
        except:
            print("cannot load prompt")
        return REACTREE_PROMPT
    
    def build_messages():
        pass

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