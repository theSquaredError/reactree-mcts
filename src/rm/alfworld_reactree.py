import logging

logger = logging.getLogger(__name__)


class ControlFlowNode:
    def __init__(self, content, depth, max_depth):
        self.content = content
        self.depth = depth
        self.max_depth = max_depth
        self.children = []
        self.parent = None

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def run(self, cur_step_id, cur_decision_id, log):
        if self.depth > self.max_depth:
            log.info('Max depth')
            return {'success': False, 'terminate': 'max_depth', 'step_id': cur_step_id, 'decision_id': cur_decision_id}

        if self.content == 'sequence':
            for child in self.children:
                terminate_info = child.run(cur_step_id, cur_decision_id, log)
                cur_step_id = terminate_info['step_id']
                cur_decision_id = terminate_info['decision_id']
                if not terminate_info['success']:
                    return {'success': False, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return {'success': True, 'step_id': cur_step_id, 'decision_id': cur_decision_id}

        if self.content == 'fallback':
            for child in self.children:
                terminate_info = child.run(cur_step_id, cur_decision_id, log)
                cur_step_id = terminate_info['step_id']
                cur_decision_id = terminate_info['decision_id']
                if terminate_info['success']:
                    return {'success': True, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return {'success': False, 'step_id': cur_step_id, 'decision_id': cur_decision_id}

        if self.content == 'parallel':
            is_success = True
            for child in self.children:
                terminate_info = child.run(cur_step_id, cur_decision_id, log)
                cur_step_id = terminate_info['step_id']
                cur_decision_id = terminate_info['decision_id']
                if not terminate_info['success']:
                    is_success = False
            return {'success': is_success, 'step_id': cur_step_id, 'decision_id': cur_decision_id}

        raise NotImplementedError()


class AlfWorldAgentNode:
    """
    ALFWorld tree node.

    run() is a unified chat loop:
      - Calls reset_chat() to start a fresh conversation for this goal.
      - Each iteration calls plan_step() → LLM sees full action history.
      - Think  : log and continue (decision counter increments).
      - Act    : execute in env, append new observation to history.
      - Expand : build ControlFlowNode + child ALFWorld nodes, then delegate
                 to ControlFlowNode.run() which applies sequence/fallback/parallel.
    """

    def __init__(
        self,
        content,
        depth=0,
        llm_agent=None,
        env=None,
        max_steps=1,
        max_decisions=1,
        max_depth=2,
    ):
        self.content = content
        self.depth = depth
        self.llm_agent = llm_agent
        self.env = env
        self.max_steps = max_steps
        self.max_decisions = max_decisions
        self.max_depth = max_depth
        self.children = []
        self.parent = None

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def run(self, cur_step_id=1, cur_decision_id=1, log=None):
        import re
        task_desc = self.content['nl_inst']
        task_type = self.content.get('task_type', 'unknown')
        # Use last_obs so subgoal nodes see current env state, not the stale initial obs
        raw_obs = self.env.last_obs if self.env.last_obs is not None else self.env.init_obs
        # ALFWorld embeds "Your task is to: ..." in the obs text — strip it to avoid duplication
        current_obs = re.sub(r'\nYour task is to:[^\n]*', '', raw_obs).strip()
        try:
            admissible_commands = self.env.last_info['admissible_commands'][0]
        except (KeyError, TypeError, IndexError):
            try:
                admissible_commands = self.env.init_info['admissible_commands'][0]
            except (KeyError, TypeError, IndexError):
                admissible_commands = []
        logger.info(f'cur_step_id:{cur_step_id} | cur_decision_id: {cur_decision_id}')
        logger.info(f'task_desc: {task_desc}| task_type: {task_type} | current_obs: {current_obs}')

        context = self._make_context()
        self.llm_agent.reset_chat(task_desc, current_obs, admissible_commands, context=context)
        logger.info("LLM reset | subgoal=%s | is_child=%s", task_desc, context is not None)

        log = log or logger
        while True:
            # if cur_step_id > self.max_steps:
            #     log.info('Max steps')
            #     return {'success': False, 'terminate': 'max_step', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            if cur_decision_id > self.max_decisions:
                log.info('Max decisions')
                return {'success': False, 'terminate': 'max_decision', 'step_id': cur_step_id, 'decision_id': cur_decision_id}

            try:
                next_step_info = self.llm_agent.plan_step()
            except Exception as error_message:
                log.info(f"Plan Next Step Error: {error_message}")
                return {'success': False, 'terminate': 'plan_next_step_error', 'step_id': cur_step_id, 'decision_id': cur_decision_id}

            next_step_class = next_step_info['type']
            next_step = next_step_info['content']
            log.info(f'{next_step_class}: {next_step}')

            if next_step_class == 'Think':
                cur_decision_id += 1
            elif next_step_class == 'Act':
                action = next_step.strip()
                if action == 'done':
                    return {'success': True, 'terminate': 'done', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                if action == 'failure':
                    return {'success': False, 'terminate': 'failure', 'step_id': cur_step_id, 'decision_id': cur_decision_id}

                obs, _, dones, info = self.env.step(action)
                obs_text = obs[0] if isinstance(obs, list) else obs
                done = dones[0] if isinstance(dones, list) else dones
                try:
                    admissible_commands = info['admissible_commands'][0]
                except (KeyError, TypeError, IndexError):
                    admissible_commands = []

                log.info(obs_text)
                self.llm_agent.add_observation(
                    f"{obs_text}\nAvailable actions: {', '.join(admissible_commands)}"
                    if admissible_commands else obs_text
                )
                cur_step_id += 1
                cur_decision_id += 1
                if done:
                    return {'success': True, 'terminate': 'env_done', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            elif next_step_class == 'Expand':
                cur_decision_id += 1
                control_flow = next_step['control_flow']
                subgoals = next_step['conditions']
                if isinstance(subgoals, str):
                    subgoals = [subgoal.strip() for subgoal in subgoals.split(',') if subgoal.strip()]

                control_flow_node = ControlFlowNode(control_flow, self.depth + 1, self.max_depth)
                self.add_child(control_flow_node)

                for idx, subgoal in enumerate(subgoals):
                    subgoal_info = {'nl_inst': subgoal, 'task_type': task_type}
                    agent_node = AlfWorldAgentNode(
                        subgoal_info,
                        depth=self.depth + 2,
                        llm_agent=self.llm_agent,
                        env=self.env,
                        max_steps=self.max_steps,
                        max_decisions=self.max_decisions,
                        max_depth=self.max_depth,
                    )
                    control_flow_node.add_child(agent_node)
                    logger.info(
                        "Created AlfWorldAgentNode [%d/%d] | depth=%d | subgoal='%s'",
                        idx + 1, len(subgoals), self.depth + 2, subgoal,
                    )

                return control_flow_node.run(cur_step_id, cur_decision_id, log)
            elif next_step_class == 'Unknown':
                cur_decision_id += 1
            else:
                raise NotImplementedError()

    def _make_context(self):
        """Build root goal + immediate control-flow + sibling context for subgoal nodes.

        Tree shape: AlfWorldAgentNode (self) → parent=ControlFlowNode → parent=AlfWorldAgentNode (supergoal).
        Returns None for root nodes (no parent ControlFlowNode).
        """
        cf_node = self.parent  # ControlFlowNode, or None for root
        if cf_node is None:
            return None

        supergoal_node = cf_node.parent  # AlfWorldAgentNode one level up
        if supergoal_node is None:
            return None

        supergoal = supergoal_node.content['nl_inst']
        control_flow = cf_node.content
        sibling_goals = [child.content['nl_inst'] for child in cf_node.children]

        if control_flow == 'sequence':
            control_flow_phrase = 'in sequence'
        elif control_flow == 'fallback':
            control_flow_phrase = 'using a fallback strategy'
        elif control_flow == 'parallel':
            control_flow_phrase = 'in parallel'
        else:
            control_flow_phrase = control_flow

        if len(sibling_goals) > 1:
            sibling_goals_phrase = ', '.join(sibling_goals[:-1]) + ', and ' + sibling_goals[-1]
        else:
            sibling_goals_phrase = sibling_goals[0] if sibling_goals else ''

        return (
            f"[Context] Your supergoal is to: {supergoal}\n"
            f"To achieve the supergoal, sibling tasks run {control_flow_phrase}. "
            f"Sibling tasks at this level: {sibling_goals_phrase}."
        )

    def _log_tree(self):
        lines = [f"[AlfWorldNode   d={self.depth}] {self.content['nl_inst']}"]
        for cf in self.children:
            lines.append(f"  [ControlFlowNode d={cf.depth}] {cf.content}")
            for child in cf.children:
                lines.append(f"    [AlfWorldNode d={child.depth}] {child.content['nl_inst']}")
        logger.info("Tree:\n%s", "\n".join(lines))
