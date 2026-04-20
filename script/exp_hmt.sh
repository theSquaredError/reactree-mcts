# python src/hmt.py --config-name=alfred_reactree exp_type=collect_llm llm_agent.model_name=meta-llama/Meta-Llama-3.1-8B llm_agent.working_memory=True dataset.collect_ex_root_dir=resource/alfred prompt.sys_prompt_root_dir=resource/alfred/sys_prompt prompt.ic_ex_root_dir=resource/alfred/em_human llm_agent.ic_ex_select_type=rag llm_agent.max_decisions=100 alfred.diverse_task=True

### Evaluate HMT on full valid_seen
python src/hmt2.py --config-name=alfred_reactree exp_type=evaluate dataset.eval_set=valid_seen llm_agent.model_name=meta-llama/Meta-Llama-3.1-8B llm_agent.working_memory=True prompt.sys_prompt_root_dir=resource/alfred/sys_prompt prompt.ic_ex_root_dir=resource/alfred/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=100 llm_agent.max_decisions=100

### Evaluate HMT on full valid_unseen
# python src/hmt.py --config-name=alfred_hmt exp_type=evaluate dataset.eval_set=valid_unseen llm_agent.model_name=meta-llama/Meta-Llama-3.1-8B llm_agent.working_memory=True prompt.sys_prompt_root_dir=resource/alfred/sys_prompt prompt.ic_ex_root_dir=resource/alfred/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=100 llm_agent.max_decisions=100
