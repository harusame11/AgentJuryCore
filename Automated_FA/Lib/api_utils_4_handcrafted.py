import os
import re
import json
import concurrent.futures
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Hand-Crafted Dataset Adapter Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_agent_name(raw_role: str) -> str:
    """
    Hand-Crafted 数据集里的 role 字段是复合字符串，例如：
      "Orchestrator (thought)"        -> "Orchestrator"
      "Orchestrator (-> WebSurfer)"   -> "Orchestrator"
      "WebSurfer"                     -> "WebSurfer"
      "Orchestrator (termination condition)" -> "Orchestrator"
    提取括号之前的纯 Agent 名称并去除首尾空格。
    """
    return raw_role.split("(")[0].strip()


def _parse_team_from_initial_plan(history: list) -> dict:
    """
    Hand-Crafted 数据集没有独立的 system_prompt 字段。
    从 Orchestrator 的初始计划消息（通常是 history[1]）中解析团队成员描述。
    在文本中寻找 'assembled the following team:' 标识符，提取后续的 Agent 描述段落。

    返回格式: {"AgentName": "role description ...", ...}
    """
    roles_dict = {}

    for entry in history[:4]:  # 初始计划通常在前几步
        raw_role = entry.get("role", "")
        content = entry.get("content", "")
        if "Orchestrator" in raw_role and "assembled the following team" in content:
            # 找到标识符之后的文本
            marker = "assembled the following team:"
            idx = content.find(marker)
            if idx == -1:
                continue
            team_text = content[idx + len(marker):]

            # 截取到下一个主要章节标识符（如 "Here is an initial fact sheet"）
            stop_markers = ["Here is an initial fact sheet", "Here is the plan", "\n\n\n"]
            for sm in stop_markers:
                stop_idx = team_text.find(sm)
                if stop_idx != -1:
                    team_text = team_text[:stop_idx]
                    break

            # 按行解析 "AgentName: description"
            for line in team_text.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                # 格式通常为 "AgentName: description text"
                if ":" in line:
                    parts = line.split(":", 1)
                    agent_id = parts[0].strip()
                    agent_desc = parts[1].strip() if len(parts) > 1 else ""
                    if agent_id and len(agent_id) < 50:  # 防止把整句话当 key
                        roles_dict[agent_id] = agent_desc
            break

    return roles_dict


def _get_effective_agents_hc(history: list) -> list:
    """
    从 history 中提取所有出现过的、归一化后的 Agent 名称（去重、去掉 human）。
    """
    seen = set()
    agents = []
    skip_roles = {"human", "computerTerminal", "computer_terminal"}
    for entry in history:
        raw = entry.get("role", "")
        normalized = _normalize_agent_name(raw)
        if normalized.lower() in skip_roles:
            continue
        if normalized not in seen:
            seen.add(normalized)
            agents.append(normalized)
    return agents


def _is_passive_executor_hc(normalized_name: str) -> bool:
    """
    判断当前 Agent 是否是纯物理执行终端（无推理能力），予以豁免。
    """
    passive_keywords = ["computer_terminal", "computerterminal"]
    return normalized_name.lower().replace(" ", "_") in passive_keywords


# ─────────────────────────────────────────────────────────────────────────────
# Main Hand-Crafted Evaluation Function
# ─────────────────────────────────────────────────────────────────────────────

def step_by_step_dao_4_handcrafted(client, directory_path: str, model: str, max_tokens: int):
    """
    针对 Hand-Crafted 数据集格式的 AgentJury (Probe + DAO) 归因评估函数。

    Hand-Crafted 数据集结构与算法生成数据集的主要差异：
      1. 聊天历史字段为 "history"（相同），但 Agent 身份在 "role" 字段的复合字符串中
         例如 "Orchestrator (-> WebSurfer)" 需要归一化为 "Orchestrator"
      2. 没有独立的 "system_prompt" 字段，团队成员的角色描述内嵌在初始计划消息中
      3. 有 "role": "human" 的初始消息需要跳过
      4. GT 的 "mistake_agent" 是 Agent 的简称（如 "WebSurfer"），而非带括号的复合 role
    """
    from Lib.api_utils import _load_json_data, _get_sorted_json_files, _make_api_call, \
        _make_api_DAO_1_call, _make_api_DAO_2_call, _make_api_DAO_3_call

    print("\n--- [Hand-Crafted] Starting AgentJury Forward Probe + DAO Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    CONVICTION_THRESHOLD = 0.5

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        history_raw = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        # ── 过滤掉 "human" 消息，构建工作用的 chat_history ──────────────────
        # 同时对每条消息的 role 做归一化，存回 normalized_name 字段
        # 关键：保留 history_index（在原始 history_raw 中的真实下标）
        # 人类标注的 mistake_step 以 history_raw 的 0-based 索引为准
        # （每个 JSON body 算一步，role="human" 是第 0 步）
        chat_history = []
        for h_idx, entry in enumerate(history_raw):
            raw_role = entry.get("role", "")
            normalized = _normalize_agent_name(raw_role)
            if normalized.lower() == "human":
                continue  # human 消息跳过探针，但其 h_idx=0 保留在编号系统中
            chat_history.append({
                "content": entry.get("content", ""),
                "normalized_name": normalized,
                "raw_role": raw_role,
                "history_index": h_idx,   # ← 原始下标，与 GT mistake_step 对齐
            })

        if not chat_history:
            continue

        # ── 提取团队成员与角色描述（替代 system_prompt）───────────────────────
        roles_dict = _parse_team_from_initial_plan(history_raw)

        # 关键修复：取 roles_dict 与实际出场 Agent 的交集
        # Orchestrator 初始计划声明了全团队（如 Assistant/FileSurfer），
        # 但很多 case 只用到其中部分 Agent，挂名但未出场的成员会稀释 DAO 软投票得分
        actual_participants = set(
            e["normalized_name"] for e in []
        )  # placeholder，下面 chat_history 构建完才能用，先在后面做

        valid_agents_from_plan = list(roles_dict.keys()) if roles_dict else []

        # roles_desc 延迟到 valid_agents 确定后构建（见下方），此处仅占位
        roles_desc = ""

        print(f"--- Analyzing File: {json_file} ---")
        print(f"    Planned team (from initial plan): {valid_agents_from_plan}")

        # ── 关键修复：用实际参与者过滤计划团队，去除挂名未出场成员 ─────────────
        actual_participants = set(e["normalized_name"] for e in chat_history)
        if valid_agents_from_plan:
            valid_agents = [a for a in valid_agents_from_plan if a in actual_participants]
            # 如果交集为空（全都没出场），退化为从历史中直接提取
            if not valid_agents:
                valid_agents = _get_effective_agents_hc(history_raw)
        else:
            valid_agents = _get_effective_agents_hc(history_raw)

        # Orchestrator 是隐式协调者，不会出现在 "assembled the following team" 里，
        # 但在对话里始终参与，需要手动补回，以便 DAO 能对其进行评分
        for p in sorted(actual_participants):
            if "orchestrator" in p.lower() and p not in valid_agents:
                valid_agents.append(p)

        print(f"    Active participants (intersected): {valid_agents}")

        # roles_desc：只保留实际出场 agent 的角色描述，过滤挂名未出场成员
        if roles_dict:
            for k, v in roles_dict.items():
                if k in valid_agents:
                    roles_desc += f"--- Role: {k} ---\n{v}\n"
        if not roles_desc:
            roles_desc = "No explicit role descriptions available. Infer from agent names and behavior.\n"

        # ── Phase 1: ECHO 分层上下文摘要预计算（支持缓存复用）────────────────────
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        file_idx = os.path.splitext(json_file)[0]
        abstract_filename = f"structured_abstract_hc_{file_idx}.json"
        abstract_path = os.path.join(output_dir, abstract_filename)

        step_summaries = [""] * len(chat_history)

        if os.path.exists(abstract_path):
            # 缓存命中：直接加载已有摘要，跳过 LLM 调用
            print(f"--- [HC] Loading cached abstract from {abstract_path} (skipping Phase 1) ---")
            with open(abstract_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # 按 history_index 对齐回 step_summaries（防止顺序不一致）
            hist_idx_to_chat_idx = {e["history_index"]: i for i, e in enumerate(chat_history)}
            for item in cached:
                chat_pos = hist_idx_to_chat_idx.get(item["step_index"])
                if chat_pos is not None:
                    step_summaries[chat_pos] = item.get("summary", "")
        else:
            print(f"--- [HC] Pre-computing Hierarchical Summaries for {json_file} ---")
            summary_sys = (
            "You are a structured data extractor for a multi-agent conversation log. "
            "Your job is to summarize each agent's step into a compact JSON object.\n\n"
            "ROLE FORMAT NOTES (Hand-Crafted dataset):\n"
            "- 'Orchestrator (thought)': An internal planning/reasoning log of the Orchestrator. Summarize its decision.\n"
            "- 'Orchestrator (-> AgentName)': An instruction dispatched by the Orchestrator to a sub-agent.\n"
            "- 'WebSurfer', 'FileSurfer', 'Assistant': Action-taking agents. Summarize their action and output.\n\n"
            "Output ONLY a valid JSON object (no markdown, no ```json). "
            "The JSON object MUST have EXACTLY these three keys:\n"
            "- 'Upstream_Instruction_or_Context': Briefly describe the instruction, goal, or context passed to this agent (max 1 sentence).\n"
            "- 'Node_Action': Briefly describe what this agent decided or did (max 1 sentence).\n"
            "- 'Result_and_Feedback': Briefly describe the final output, resolution, or observation result (max 1 sentence).\n\n"
            "### FEW-SHOT EXAMPLES ###\n"
            "Example 1 — WebSurfer:\n"
            "Input:\n"
            "Agent: WebSurfer\n"
            "Content: I searched for martial arts schools near NYSE and found a list of dojos.\n\n"
            "Output:\n"
            "{\n"
            "  \"Upstream_Instruction_or_Context\": \"Instructed to search for martial arts schools near the NYSE.\",\n"
            "  \"Node_Action\": \"Performed a web search and returned a list of martial arts schools near NYSE.\",\n"
            "  \"Result_and_Feedback\": \"Retrieved a list of dojos near NYSE.\"\n"
            "}\n\n"
            "Example 2 — Orchestrator dispatching an instruction:\n"
            "Input:\n"
            "Agent: Orchestrator (-> WebSurfer)\n"
            "Content: Please go to TripAdvisor and find trails in Yellowstone with at least 50 reviews and a 4.5+ rating.\n\n"
            "Output:\n"
            "{\n"
            "  \"Upstream_Instruction_or_Context\": \"Task requires finding highly-rated Yellowstone trails on TripAdvisor.\",\n"
            "  \"Node_Action\": \"Dispatched instruction to WebSurfer to search TripAdvisor for trails meeting specific criteria.\",\n"
            "  \"Result_and_Feedback\": \"Instruction sent; awaiting WebSurfer's search results.\"\n"
            "}"
        )

            for s_idx in range(len(chat_history)):
                s_agent = chat_history[s_idx]["normalized_name"]
                s_content = chat_history[s_idx]["content"]

                past_summaries_context = ""
                for i in range(max(0, s_idx - 2), s_idx):
                    if step_summaries[i] and "bundled" not in step_summaries[i]:
                        past_summaries_context += f"[Step {chat_history[i]['history_index']}] {chat_history[i]['normalized_name']}: {step_summaries[i]}\n"

                summary_user = ""
                if past_summaries_context:
                    summary_user += f"--- Previous Steps Context ---\n{past_summaries_context}\n\n"
                s_content_truncated = s_content[:3000] + ("...[TRUNCATED]" if len(s_content) > 3000 else "")
                summary_user += f"--- Current Step to Extract ---\nAgent: {s_agent}\nContent: {s_content_truncated}"

                msg = [{"role": "system", "content": summary_sys}, {"role": "user", "content": summary_user}]
                res = _make_api_DAO_3_call(client, model, msg, 512)
                step_summaries[s_idx] = res if res else "(Failed)"

            # 保存摘要供后续复用
            abstract_list = []
            for s_idx, summary in enumerate(step_summaries):
                abstract_list.append({
                    "step_index": chat_history[s_idx]["history_index"],
                    "agent": chat_history[s_idx]["normalized_name"],
                    "summary": summary
                })
            with open(abstract_path, "w", encoding="utf-8") as f:
                json.dump(abstract_list, f, ensure_ascii=False, indent=2)
            print(f"--- [HC] Structured abstract saved to {abstract_path} ---")

        # ── Phase 2: 滑动探针前向扫描 ────────────────────────────────────────
        error_found = False

        for idx, entry in enumerate(chat_history):
            agent_name = entry["normalized_name"]

            # 跳过纯物理沙盒输出（Hand-Crafted 里没有 Computer_terminal，
            # 但有些 Orchestrator (thought) 等内部消息也可以跳过探针直接扫描）
            # 这里只跳过被明确标记为沙盒的条目
            if _is_passive_executor_hc(agent_name):
                continue

            # ── 构建分层上下文 ──────────────────────────────────────────────
            # L3: 遥远过去 (JSON 摘要)
            dist_past = ""
            for i in range(0, idx):
                name_i = chat_history[i]["normalized_name"]
                if _is_passive_executor_hc(name_i):
                    continue
                hist_i = chat_history[i]["history_index"]
                dist_past += f"[Step {hist_i}] {name_i}: (JSON SUMMARY) {step_summaries[i]}\n"

            # 高分辨率窗口：以当前步骤为中心，前后各 5 步（chat_history 内坐标）
            # 扩展后向覆盖：即使 Probe 在早期触发，DAO 也能看到后续 5 步的全文，
            # 避免真实错误步骤仅出现在 DISTANT FUTURE 摘要中导致 Step 精度损失。
            WIN_HALF = 5
            win_start = max(0, idx - WIN_HALF)
            win_end   = min(len(chat_history), idx + WIN_HALF + 1)  # exclusive

            forward_window = ""
            for i in range(win_start, win_end):
                name_i = chat_history[i]["normalized_name"]
                if _is_passive_executor_hc(name_i):
                    continue
                fw_text = chat_history[i]["content"]
                hist_i  = chat_history[i]["history_index"]
                forward_window += f"[Step {hist_i}] {name_i}: (FULL TEXT)\n{fw_text}\n\n"

            # 窗口之前的历史：改为仅保留窗口外（win_start 之前）的 JSON 摘要
            dist_past = ""
            for i in range(0, win_start):
                name_i = chat_history[i]["normalized_name"]
                if _is_passive_executor_hc(name_i):
                    continue
                hist_i = chat_history[i]["history_index"]
                dist_past += f"[Step {hist_i}] {name_i}: (JSON SUMMARY) {step_summaries[i]}\n"

            # L4: 遥远未来 (JSON 摘要)
            future_down = ""
            for i in range(win_end, len(chat_history)):
                name_i = chat_history[i]["normalized_name"]
                if _is_passive_executor_hc(name_i):
                    continue
                hist_i = chat_history[i]["history_index"]
                future_down += f"[Step {hist_i}] {name_i}: (JSON SUMMARY) {step_summaries[i]}\n"

            hierarchical_context = ""
            if dist_past:
                hierarchical_context += f"--- SCANNED PAST (JSON Summaries) ---\n{dist_past}\n"
            _win_start_hist = chat_history[win_start]["history_index"]
            _win_end_hist   = chat_history[win_end - 1]["history_index"]
            hierarchical_context += f"--- HIGH-RESOLUTION WINDOW (Steps {_win_start_hist} to {_win_end_hist}) ---\n{forward_window}\n"
            if future_down:
                hierarchical_context += f"--- DISTANT FUTURE (JSON Summaries) ---\n{future_down}\n"

            # 当前步骤的原始 history 下标（与 GT 对齐）
            hist_idx = entry["history_index"]

            # ── 探针提示词（与算法数据集完全一致的三大法典）──────────────────
            prompt = (
                "You are an AI Accuser evaluating a localized multi-agent conversation window to identify the Final Proximal Cause of failure.\n"
                "Note: 'Orchestrator (thought)' entries are internal planning logs. 'Orchestrator (-> AgentName)' entries are dispatched instructions.\n"
                f"Problem: {problem}\nGround Truth Answer: {ground_truth}\n\n"
                f"Hierarchical view:\n{hierarchical_context}\n"
                f"--- ROLE-BASED EVALUATION TASK ---\n"
                f"Evaluate Step {hist_idx} by '{agent_name}'. You MUST employ the following Strict Liability logic before deciding:\n"
                f"- OBSERVABLE ERROR THRESHOLD [CRITICAL]: A step should ONLY be marked as 'Root Error: Yes' if there is a concrete, observable manifestation of error in THAT SPECIFIC STEP's output — such as hallucinated data introduced, wrong resource navigated to, incorrect data returned, or an explicitly wrong/incomplete instruction issued. Merely existing in the conversation without producing a demonstrable error in this exact step's output is INSUFFICIENT for conviction.\n"
                f"- STRICT EXECUTOR EXCEPTION: If '{agent_name}' is a strictly passive actuator with no reasoning capacity and it merely returned an execution output without modification, mark 'Root Error: No'.\n"
                f"- TOOL-AGENT DIRECTION CHECK [applies when agent_name is WebSurfer or FileSurfer]: "
                f"Before marking 'Root Error: Yes' for a tool agent, check the INSTRUCTION it received from the directing agent (Orchestrator). "
                f"If '{agent_name}' correctly executed the instruction (navigated to the URL it was given, ran the query it was told to run) "
                f"but the instruction itself was wrong (wrong URL, wrong search target, wrong resource), "
                f"then '{agent_name}' is NOT the root cause — the directing agent that issued the wrong instruction is. "
                f"Mark 'Root Error: No' for '{agent_name}' and continue scanning toward the step where the wrong instruction was issued.\n"
                f"- LOCK-IN TEST [GATE — apply after OBSERVABLE ERROR THRESHOLD is satisfied]: Check the DISTANT FUTURE section before finalizing 'Root Error: Yes'. Ask: was the error at this step LOCKED IN as the decisive cause, or did subsequent agents continue executing, retrying, or replanning after this step?\n"
                f"  * If DISTANT FUTURE shows continued task execution after this step (more retrieval attempts, replanning, or additional agent actions), the error here has NOT yet been locked in — a downstream agent will make the actual decisive decision. Mark 'Root Error: No' and continue scanning for the step where the wrong output was definitively accepted or the task was finalized incorrectly.\n"
                f"  * If DISTANT FUTURE shows NO further correction attempts — the task ended or the wrong answer was directly propagated from this step — the error IS locked in. Mark 'Root Error: Yes'.\n"
                f"  * EXCEPTION (narrow): Only apply if ALL of the following are true: (1) this step returned NOTHING (empty result, complete retrieval failure), AND (2) the DISTANT FUTURE shows ZERO subsequent attempts by ANY agent to obtain the same information through the same or alternative means, AND (3) the task terminated with a wrong answer directly attributable to this missing data. A failed navigation that was followed by ANY retry attempt — even an unsuccessful one — does NOT meet this exception. When in doubt, mark 'Root Error: No' and continue scanning.\n"
                f"- RETRY SEQUENCE RULE [SPECIAL — for WebSurfer/FileSurfer navigation failures]: Web navigation tasks commonly involve multiple sequential attempts before succeeding or permanently failing. If the current step is a failed navigation attempt (wrong page, irrelevant content, redirect error), you MUST check the HIGH-RESOLUTION WINDOW and DISTANT FUTURE for retry attempts before deciding:\n"
                f"  * Count how many times WebSurfer or FileSurfer appears after this step in the HIGH-RESOLUTION WINDOW or DISTANT FUTURE with a similar navigation action. If count >= 1, this step is part of a retry sequence and is NOT the lock-in point. Mark 'Root Error: No'.\n"
                f"  * The lock-in point in a retry sequence is the LAST attempt whose output was accepted by the Orchestrator as the final answer — not any intermediate failed attempt.\n\n"
                "Respond ONLY in the exact format:\n"
                "1. Root Error: [Yes/No]\n"
                "2. Reason: [Cite the specific observable error in this step's output, or state why none exists]"
            )

            messages = [{"role": "system", "content": "You are a precise step-by-step evaluator scanning a multi-agent task that has FAILED. You evaluate steps chronologically to find the LOCK-IN point: the step where a wrong output was produced AND accepted as the basis for the final answer, with no subsequent correction. Information retrieval failures that were followed by further attempts are NOT the lock-in point — even if those further attempts also failed. The lock-in point is where the wrong information or wrong decision became IRREVERSIBLE: typically where an agent accepted bad data and finalized a conclusion, or where a required action was permanently skipped with no retry. For web navigation tasks, WebSurfer routinely retries multiple times. A single failed navigation step is almost never the lock-in point unless it was the FINAL attempt before the Orchestrator accepted an incorrect answer. When evaluating a WebSurfer step, always look for subsequent WebSurfer appearances in the context before marking Root Error: Yes."}, {"role": "user", "content": prompt}]
            answer = _make_api_call(client, model, messages, max_tokens)

            if not answer:
                continue
            print(f"Probe Evaluation for Step {hist_idx} ({agent_name}): {answer[:120]}...")

            ans_lower = answer.lower()
            if "yes" in ans_lower[:20]:
                reason = answer.split('Reason:', 1)[-1].strip() if 'Reason:' in answer else answer
                print(f"Probe flagged error at Step {hist_idx}. Invoking DAO Arbitration...")

                # ── Phase 3: DAO 仲裁（三权分立）────────────────────────────
                base_context = (
                    "[BACKGROUND] This is a post-mortem investigation of a FAILED multi-agent task. "
                    "The agents below did NOT produce the correct final answer. "
                    "Your job as an arbitrator is to determine WHICH agent and WHICH step was the root cause of this failure.\n\n"
                    f"General Task Origin: {problem}\nGround Truth Expected Outcome: {ground_truth}\n"
                    f"Active Agents in This Case (ONLY evaluate these):\n{roles_desc}\n"
                    f"SCOPE CONSTRAINT: Only the agents listed above actually participated in this task. "
                    f"Do NOT generate evaluations for any other agents not present in the log.\n\n"
                    "Note: 'Orchestrator (thought)' are internal planning logs; 'Orchestrator (-> X)' are dispatched instructions.\n\n"
                    "You have access to hierarchical context showing:\n"
                    "Immediate agents: Full details\n"
                    "Nearby agents: Key decisions\n"
                    "Distant agents: Brief summaries\n\n"
                    "Step numbers are original history indices. Step 0 is the human task input (not shown in the log). Agent steps start from Step 1. Internal thought entries ('Orchestrator (thought)') are included in the count and have their own step numbers. IMPORTANT: Output the EXACT [Step N] number you see in the log — do NOT re-index or offset.\n\n"
                    "--- CASE DOSSIER (Hierarchical Log) ---\n"
                    f"{hierarchical_context}\n"
                    "--- FLAG FOR REVIEW & ATTRIBUTION CONSTRAINTS ---\n"
                    f"A frontline scanner has flagged Step {hist_idx} (Agent: '{agent_name}') in the context window.\n"
                    "You are evaluating a LOCALIZED window of action. You must strictly adhere to the following liability doctrines:\n"
                    "[Proximal Cause Attribution]: Identify the final proximal cause — the precise step where the error was first *committed* (i.e., where wrong data was first introduced or wrong action was first taken), not just where it was observed downstream.\n"
                    "[ROOT CAUSE vs MANIFESTATION POINT]: Distinguish between where the error ORIGINATED and where it became OBSERVABLE. "
                    "A downstream agent that produced a wrong output because it faithfully executed a wrong instruction is a MANIFESTATION POINT — "
                    "the agent that ISSUED the wrong instruction is the ROOT CAUSE.\n"
                    "[TOOL-AGENT DIRECTION DOCTRINE — CRITICAL FOR WebSurfer/FileSurfer]: "
                    "WebSurfer and FileSurfer are instruction-directed tool agents. They navigate to URLs, perform searches, "
                    "and access resources as commanded by the directing agent (typically Orchestrator). "
                    "Apply the following rule BEFORE assigning liability to WebSurfer or FileSurfer:\n"
                    "  (a) EXONERATE if: the tool agent correctly executed the instruction it received "
                    "(went to the URL it was told to go to, performed the search query it was given), "
                    "but the instruction itself was wrong. In this case, the ROOT CAUSE is the DIRECTING AGENT that issued the wrong URL, "
                    "wrong search query, or wrong navigation target — NOT WebSurfer/FileSurfer.\n"
                    "  (b) CONVICT if: the tool agent demonstrably went to a DIFFERENT destination than instructed, "
                    "silently returned clearly wrong data from the correct destination without flagging discrepancies "
                    "its role required it to detect, or completely failed to execute the action.\n"
                    "  KEY QUESTION: 'Did WebSurfer/FileSurfer do what it was told?' If YES → blame the instructor. If NO → blame the tool agent.\n"
                    "[RETRY SEQUENCE PROTECTION — CRITICAL]: Web navigation tasks involve iterative attempts. Before assigning high error_likelihood to any WebSurfer or FileSurfer step, count how many times that agent appears in the log AFTER the flagged step. "
                    "If the SAME agent made >= 1 subsequent navigation attempt after the flagged step, the flagged step is an intermediate retry failure, NOT the final proximal cause. "
                    "Assign it LOW error_likelihood (≤ 0.2). The correct attribution target is either: "
                    "(a) the LAST navigation attempt whose output the Orchestrator accepted as final without seeking further verification, OR "
                    "(b) the Orchestrator step where it terminated the task based on unverified or incorrect retrieved information.\n"
                )

                task_and_format_instruction = (
                    "Output your response as valid JSON wrapper in <json></json> tags (or just raw JSON):\n"
                    "{\n"
                    '  "analysis_summary": "Brief overview of your investigation approach and findings",\n'
                    '  "agent_evaluations": [\n'
                    '    {\n'
                    '      "agent_name": "agent_name",\n'
                    '      "step_index": 1,\n'
                    '      "error_likelihood": 1.0,\n'
                    '      "reasoning": "Why this agent may or may not have caused the error"\n'
                    '    }\n'
                    '  ],\n'
                    '  "primary_conclusion": {\n'
                    '    "attribution": ["agent_name"],\n'
                    '    "mistake_step": 1,\n'
                    '    "confidence": 1.0,\n'
                    '    "reasoning": "Explanation of your primary conclusion"\n'
                    '  },\n'
                    '  "secondary_conclusion": {\n'
                    '    "attribution": ["agent_name"],\n'
                    '    "mistake_step": 1,\n'
                    '    "confidence": 0.5,\n'
                    '    "reasoning": "An alternative plausible attribution if the evidence supports multiple candidates. Omit if only one candidate is credible."\n'
                    '  }\n'
                    "}\n"
                    "Be thorough and objective. Provide calibrated confidence scores. "
                    "If the evidence supports multiple plausible candidates, rank them — the goal is to narrow the attribution space, not force a single point. "
                    "Include secondary_conclusion only when a second candidate has meaningful supporting evidence.\n"
                    "STEP SELECTION RULE: Select the EARLIEST step where the responsible agent first produced "
                    "a concretely wrong output — wrong data returned, wrong destination accessed, wrong instruction issued, "
                    "or required output completely absent. Do NOT select a step where the error was merely observed or "
                    "propagated downstream from an earlier root cause."
                )


                # --- Expert 1 (HC Cluster 0, n=14): Action Execution & Resource Access Expert ---
                # Empirical cluster theme: physical execution failures — agents failing to perform
                # required actions, access external resources, or produce valid actionable outputs.
                # Agent-agnostic: any role (planner or executor) can commit execution-layer failures.
                sys_1 = (
                    "You are an Action Execution and Resource Access Attribution Expert. "
                    "Your focus is exclusively on failures where an agent performed the WRONG physical action, "
                    "or failed to perform a required physical action at all — regardless of its role.\n\n"
                    "FAILURE PATTERNS YOU COVER (physical action layer only):\n"
                    "- Agent navigated to the wrong resource or URL (wrong destination, not wrong data quality)\n"
                    "- Agent failed to execute a required navigation, click, or access action entirely\n"
                    "- Agent failed to interact with required interface elements (dropdowns, tabs, filters, buttons)\n"
                    "- Agent produced no output at all — action was silently skipped or abandoned\n"
                    "- Agent lacked the capability to process a required input modality (audio, image, structured document) and took no compensating action\n"
                    "- Agent issued a wrong concrete instruction that directed a downstream agent to the wrong target\n\n"
                   
                    "YOUR TASK:\n"
                    "1. Analyze ALL agents. Do NOT pre-assign blame based on agent role or name.\n"
                    "2. Attribute guilt only to the agent whose concrete physical action (or its complete absence) directly caused the failure.\n"
                    "3. Identify the exact step where the physical execution failure first occurred.\n"
                    "4. Prioritize execution-layer failures over reasoning or information quality failures, "
                    "but do not ignore an agent's execution if its concrete action was itself wrong or missing.\n"
                    "5. Provide confidence scores and specific evidence from the conversation log."
                )
                dao_1 = (
                    base_context +
                    "\n[YOUR EXPERT DIMENSION] You are the ACTION EXECUTION & RESOURCE ACCESS Expert. "
                    "Analyze ALL agents. Prioritize attribution to the agent whose physical execution action (or inaction) "
                    "is the direct cause — regardless of whether that agent is a planner, coordinator, or executor. "
                    "Prioritize execution-layer failures, but do not dismiss an agent's concrete wrong action "
                    "simply because it also has reasoning or information dimensions.\n\n" +
                    task_and_format_instruction
                )

                # --- Expert 2 (HC Cluster 1, n=17): Reasoning, Planning & Process Control Expert ---
                # Empirical cluster theme: decision-making and logic failures — wrong instructions,
                # premature conclusions, flawed algorithms, invalid delegations, insufficient validation.
                # Agent-agnostic: both planners and executors can produce flawed reasoning.
                sys_2 = (
                    "You are a Reasoning, Planning, and Process Control Attribution Expert. "
                    "that share a common root: an agent applying flawed internal reasoning, issuing incorrect "
                    "decisions, or failing to control task flow correctly, regardless of its role.\n\n"
                    "FAILURE PATTERNS YOU PRIORITIZE (derived from empirical cluster analysis):\n"
                    "- Agent issued incorrect, ambiguous, or hallucinated procedural guidance\n"
                    "- Agent prematurely concluded the task without sufficient information\n"
                    "- Agent failed to verify prior results before initiating new planning\n"
                    "- Agent applied an incorrect algorithm, formula, or logical rule\n"
                    "- Agent made decisions based on incomplete data from a subordinate agent\n"
                    "- Agent triggered premature replanning that interrupted sequential execution\n"
                    "- Agent misinterpreted or misapplied a linguistic or procedural rule\n"
                    "- Agent delegated a task to an incorrect or incapable resource\n"
                    "- Agent omitted a required final step, leaving the task incomplete\n"
                    "- Agent issued repetitive commands instead of adapting its strategy\n"
                    "- Agent failed to validate intermediate results before generating final output\n"
                    "- Agent produced an incorrect calculation due to missing or approximated parameters\n\n"
                    "YOUR TASK:\n"
                    "1. Analyze ALL agents. Do NOT pre-assign blame based on agent role or name.\n"
                    "2. Attribute guilt only to the agent whose reasoning or decision was the root cause.\n"
                    "3. Identify the exact step where the logical or planning failure first occurred.\n"
                    "4. Prioritize reasoning and decision-layer failures over physical execution or information quality failures, "
                    "but do not ignore an agent's flawed decision if it concretely drove the failure even when intertwined with execution.\n"
                    "5. Provide confidence scores and specific evidence from the conversation log."
                )
                dao_2 = (
                    base_context +
                    "\n[YOUR EXPERT DIMENSION] You are the REASONING, PLANNING & PROCESS CONTROL Expert. "
                    "Analyze ALL agents. Prioritize attribution to the agent whose internal reasoning, planning decision, "
                    "or process control action is the direct cause — this may be a planner, coordinator, or executor. "
                    "Prioritize reasoning-layer failures, but do not dismiss a clearly flawed decision "
                    "simply because it also has execution or information quality dimensions.\n\n" +
                    task_and_format_instruction
                )

                # --- Expert 3 (HC Cluster 2, n=27): Information Retrieval Completeness & Quality Expert ---
                # Empirical cluster theme: information scope and quality failures — insufficient retrieval,
                # wrong sources, incomplete extraction, temporal mismatch, query formulation failures.
                # Agent-agnostic: any agent that handles or relies on information can commit these failures.
                sys_3 = (
                    "You are an Information Retrieval Completeness and Quality Attribution Expert. "
                    "Your focus is exclusively on failures where an agent reached the RIGHT destination "
                    "but obtained WRONG, INCOMPLETE, or INSUFFICIENT information from it — or failed to seek "
                    "a better source when the current one was inadequate.\n\n"
                    "FAILURE PATTERNS YOU COVER (information quality and scope layer only):\n"
                    "- Agent accessed the correct resource but extracted incomplete or wrong data from it\n"
                    "- Agent over-relied on a secondary or unverified source instead of the primary authoritative one\n"
                    "- Agent used imprecise or insufficient search/query parameters leading to retrieval failure\n"
                    "- Agent failed to apply available filters or perform iterative search refinement\n"
                    "- Agent retrieved information from the wrong time period or an outdated version of a source\n"
                    "- Agent performed partial page processing, missing critical information within the page\n"
                    "- Agent prematurely concluded from incomplete data without seeking additional sources\n"
                    "- OCR or parsing failure within a document leading to missed or incorrect extracted content\n"
                    "- Agent's query formulation was semantically insufficient for the task's retrieval constraints\n\n"
                    "WHAT YOU DO NOT COVER (defer to other experts):\n"
                    "- Whether the agent navigated to the correct destination at all → Expert 1\n"
                    "- Whether the agent's reasoning, planning, or decision logic was flawed → Expert 2\n\n"
                    "YOUR TASK:\n"
                    "1. Analyze ALL agents. Do NOT pre-assign blame based on agent role or name.\n"
                    "2. Attribute guilt to the agent whose information retrieval scope or quality failure is the root cause.\n"
                    "3. Identify the exact step where incorrect or incomplete information was first introduced into the pipeline.\n"
                    "4. Do NOT attribute errors rooted in wrong physical navigation or reasoning/logic flaws.\n"
                    "5. Provide confidence scores and specific evidence from the conversation log."
                )
                dao_3 = (
                    base_context +
                    "\n[YOUR EXPERT DIMENSION] You are the INFORMATION RETRIEVAL COMPLETENESS & QUALITY Expert. "
                    "Analyze ALL agents. Attribute failure to the agent that reached the right destination "
                    "but obtained wrong, incomplete, or insufficient information — or failed to seek a better source. "
                    "Do NOT blame wrong-navigation failures (agent went to wrong destination → Expert 1) or "
                    "reasoning/planning failures (agent made a flawed decision → Expert 2). "
                    "Focus solely on information scope, source selection quality, and extraction completeness.\n\n" +
                    task_and_format_instruction
                )

                def _call_dao(args):
                    i, sys_p, usr_p = args
                    msg = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
                    if i == 0: return _make_api_DAO_1_call(client, model, msg, max_tokens)
                    if i == 1: return _make_api_DAO_2_call(client, model, msg, max_tokens)
                    return _make_api_DAO_3_call(client, model, msg, max_tokens)

                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futs = {executor.submit(_call_dao, (i, s, u)): i
                            for i, (s, u) in enumerate([(sys_1, dao_1), (sys_2, dao_2), (sys_3, dao_3)])}
                    results = [None] * 3
                    for f in concurrent.futures.as_completed(futs):
                        results[futs[f]] = f.result()

                # ─── Phase 4: Dense Soft Voting (对齐 Alg-Gen api_utils.py) ──────────
                # 累加每位仲裁者 agent_evaluations[].error_likelihood（而非
                # primary_conclusion.confidence），与日志解析字段保持一致。
                agent_scores = {a: 0.0 for a in valid_agents}
                step_votes   = {}          # {step_str: accumulated_conf}
                judge_votes  = []          # [(culprit_agent, step_str, conf)]
                valid_resp   = 0

                for i, r in enumerate(results):
                    if not r:
                        continue
                    print(f"--- Arbitrator {i+1} response ---\n{r}\n--- End Arbitrator {i+1} ---\n")
                    try:
                        json_match = re.search(r'<json>(.*?)</json>', r, re.DOTALL)
                        if json_match:
                            jstr = json_match.group(1).strip()
                        else:
                            # 兼容 ```json ... ``` Markdown 代码块
                            md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', r, re.DOTALL)
                            jstr = md_match.group(1).strip() if md_match else \
                                   re.search(r'\{.*\}', r, re.DOTALL).group(0)
                        jdata = json.loads(jstr)

                        # Dense Soft Voting: 累加 agent_evaluations 中的 error_likelihood
                        local_ev_scores = {}
                        for ev in jdata.get('agent_evaluations', []):
                            ev_agent      = ev.get('agent_name', '')
                            ev_likelihood = float(ev.get('error_likelihood', 0.0))
                            if ev_agent:
                                if ev_agent not in agent_scores:
                                    agent_scores[ev_agent] = 0.0
                                agent_scores[ev_agent] += ev_likelihood
                                local_ev_scores[ev_agent] = local_ev_scores.get(ev_agent, 0.0) + ev_likelihood

                        # Step 票：来自 primary_conclusion（加权用 confidence）
                        pc   = jdata.get('primary_conclusion', {})
                        conf = float(pc.get('confidence', 0.0))
                        step_val = str(pc.get('mistake_step', hist_idx))
                        step_votes[step_val] = step_votes.get(step_val, 0.0) + conf

                        # judge_agent: 以该仲裁者 agent_evaluations 中 error_likelihood 最高的 agent
                        # 作为 step 票归属（修复 attr[0] bug：attribution 列表首位不一定是主罪人）
                        if local_ev_scores:
                            judge_agent = max(local_ev_scores, key=local_ev_scores.get)
                        else:
                            attr = pc.get('attribution', [])
                            if isinstance(attr, str):
                                attr = [attr]
                            judge_agent = attr[0] if attr else ''
                        judge_votes.append((judge_agent, step_val, conf))
                        valid_resp += 1
                    except Exception:
                        pass

                if valid_resp > 0:
                    # 定罪门槛：top agent 的平均 error_likelihood >= CONVICTION_THRESHOLD
                    top_score = max(agent_scores.values()) if agent_scores else 0.0
                    avg_top   = top_score / valid_resp

                    if avg_top >= CONVICTION_THRESHOLD:
                        final_culprit = max(agent_scores, key=agent_scores.get)

                        # Step 选择：优先取投给 final_culprit 的法官的 step 票
                        culprit_step_votes = {}
                        for jg_agent, jg_step, jg_conf in judge_votes:
                            if jg_agent == final_culprit:
                                culprit_step_votes[jg_step] = \
                                    culprit_step_votes.get(jg_step, 0.0) + jg_conf

                        if culprit_step_votes:
                            best_step_str = max(culprit_step_votes, key=culprit_step_votes.get)
                        elif step_votes:
                            # 退而求其次：在 final_culprit 的真实步骤中找最高票
                            valid_steps = [
                                str(e["history_index"]) for e in chat_history
                                if e["normalized_name"] == final_culprit
                            ]
                            f_votes = {k: v for k, v in step_votes.items() if k in valid_steps}
                            if f_votes:
                                max_v = max(f_votes.values())
                                candidates = [k for k, v in f_votes.items() if v == max_v]
                                best_step_str = min(candidates,
                                                    key=lambda x: int(x) if x.isdigit() else 0)
                            else:
                                best_step_str = max(step_votes, key=step_votes.get)
                        else:
                            best_step_str = str(hist_idx)

                        try:
                            culprit_step = int(best_step_str)
                        except ValueError:
                            culprit_step = hist_idx

                        # Ghost-step 防护：step 必须是 final_culprit 的真实出场步骤
                        valid_hist_steps = [
                            str(e["history_index"]) for e in chat_history
                            if e["normalized_name"] == final_culprit
                        ]
                        if valid_hist_steps and str(culprit_step) not in valid_hist_steps:
                            culprit_step = min(int(s) for s in valid_hist_steps)
                            print(f"  [Ghost-step guard] Step corrected to earliest real appearance: {culprit_step}")

                        print(f"Error conclusively found at step {culprit_step}. "
                              f"(AvgLikelihood: {avg_top:.2f}) Halting forward scan.")
                        print(f"\nPrediction for {json_file}: Error found.")
                        print(f"Agent Name: {final_culprit}")
                        print(f"Step Number: {culprit_step}")
                        print(f"Reason provided by DAO: Convicted by DAO Tribunal. (Dense Soft Vote)")
                        error_found = True
                        break
                    else:
                        print(f"DAO absolved step {hist_idx} "
                              f"(AvgLikelihood: {avg_top:.2f} < {CONVICTION_THRESHOLD}). Continuing scan...")
                else:
                    print("All DAO calls failed. Assuming error at probed step.")
                    print(f"\nPrediction for {json_file}: Error found.")
                    print(f"Agent Name: {agent_name}")
                    print(f"Step Number: {hist_idx}")
                    error_found = True
                    break





        # ── 探针未命中时的规划失职兜底（对齐 Alg-Gen Step-0 Failsafe）──────────
        # Alg-Gen 版归因 chat_history 中的第一个出场 Agent（通常为 Planner）。
        # HC 版同样取 chat_history[0]（已过滤 human），即 Orchestrator 的首次出现，
        # 语义上等价于"规划阶段失责"，与 Alg-Gen 策略保持一致。
        if not error_found:
            first_entry      = chat_history[0]
            first_exec_agent = first_entry["normalized_name"]
            first_exec_step  = first_entry["history_index"]

            print(f"Probe missed the error. Activating Step-0 Failsafe for {json_file}.")
            print(f"\nPrediction for {json_file}: Error found.")
            print(f"Agent Name: {first_exec_agent}")
            print(f"Step Number: {first_exec_step}")
            print(f"Reason provided by DAO: (HC Failsafe) Initial planning or first execution failure.")

# ─────────────────────────────────────────────────────────────────────────────
# AgentJury_hc_enhance — Half-Split Probe + Multi-Window DAO
# ─────────────────────────────────────────────────────────────────────────────

def _build_probe_half_text(chat_half: list, step_summaries_half: list) -> str:
    """
    将半段日志构建为全文字符串（供探针窗口使用）。
    Orchestrator (thought) 也保留全文（用户要求）。
    WebSurfer 内容截断至 2000 chars 防止 prompt 过长。
    """
    parts = []
    for entry, summary in zip(chat_half, step_summaries_half):
        hist_idx   = entry["history_index"]
        agent_name = entry["normalized_name"]
        raw_role   = entry["raw_role"]
        content    = entry["content"]
        # WebSurfer / FileSurfer 内容通常包含页面截图描述，截断控制 token
        if agent_name.lower() in ("websurfer", "filesurfer"):
            content = content[:2000] + ("...[TRUNCATED]" if len(content) > 2000 else "")
        parts.append(f"[Step {hist_idx}] {raw_role}:\n{content}\n")
    return "\n".join(parts)


def _build_summary_text(chat_half: list, step_summaries_half: list) -> str:
    """
    将半段日志构建为 JSON 摘要字符串（供远端/近端上下文使用）。
    """
    parts = []
    for entry, summary in zip(chat_half, step_summaries_half):
        hist_idx   = entry["history_index"]
        agent_name = entry["normalized_name"]
        parts.append(f"[Step {hist_idx}] {agent_name}: (JSON SUMMARY) {summary}")
    return "\n".join(parts)


def _parse_probe_output(raw: str) -> list:
    """
    从探针的原始输出中解析 suspicious_steps 列表。
    返回: [{"step": int, "agent": str, "confidence": float, "reason": str}, ...]
    """
    if not raw:
        return []
    try:
        # 优先匹配 JSON 代码块
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if not m:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return []
        data = json.loads(m.group(0) if not hasattr(m, 'group') or m.lastindex is None
                          else m.group(1) if m.lastindex >= 1 else m.group(0))
        steps = data.get("suspicious_steps", [])
        result = []
        for s in steps:
            try:
                result.append({
                    "step":       int(s.get("step", -1)),
                    "agent":      str(s.get("agent", "")),
                    "confidence": float(s.get("confidence", 0.0)),
                    "reason":     str(s.get("reason", "")),
                })
            except (ValueError, TypeError):
                continue
        return result
    except Exception:
        return []


def _build_merged_dao_context(chat_history: list, step_summaries: list,
                               top_flagged: list, win_half: int = 2) -> str:
    """
    构建多窗口合并 DAO 上下文。

    top_flagged: [(chat_idx, hist_idx, agent_name, confidence, reason), ...]
    高分辨率区：每个 flagged step 的 ±win_half 步 → 全文
    其余区域   → JSON 摘要
    """
    # 确定高分辨率位置集合
    high_res_pos = set()
    for (chat_idx, _, _, _, _) in top_flagged:
        for offset in range(-win_half, win_half + 1):
            pos = chat_idx + offset
            if 0 <= pos < len(chat_history):
                high_res_pos.add(pos)

    # 构建分层上下文正文（不暴露探针标记信息，防止 DAO 盲从偏见）
    lines = []
    in_high_res = None
    for i, entry in enumerate(chat_history):
        hist_idx   = entry["history_index"]
        agent_name = entry["normalized_name"]
        raw_role   = entry["raw_role"]
        content    = entry["content"]

        if i in high_res_pos:
            if in_high_res is not True:
                lines.append("--- [HIGH-RESOLUTION ZONE — Full Text] ---")
            content_disp = content[:2500] + ("...[TRUNCATED]" if len(content) > 2500 else "")
            lines.append(f"[Step {hist_idx}] {raw_role}: (FULL TEXT)\n{content_disp}\n")
            in_high_res = True
        else:
            if in_high_res is not False:
                lines.append("--- [COMPRESSED ZONE — JSON Summaries] ---")
            lines.append(f"[Step {hist_idx}] {agent_name}: (JSON SUMMARY) {step_summaries[i]}")
            in_high_res = False

    return "\n".join(lines)


def AgentJury_hc_enhance(client, directory_path: str, model: str,
                          max_tokens: int,
                          target_seg_size: int = 15, max_probe_k: int = 8):
    """
    Enhanced AgentJury for Hand-Crafted dataset.

    Architecture
    ────────────
    Phase 1: Sequential step summarization（支持缓存复用）
    Phase 2: Adaptive K-Split Probe
             K 由 target_seg_size 和 max_probe_k 自动推算：
               K = max(2, min(max_probe_k, ceil(N / target_seg_size)))
             每段构造：当前段全文 + 过去段摘要 + 未来段摘要
             跨段去重：同一步骤保留最高置信度
    Phase 3: 单次 DAO 调用（三专家并行）
             - top-N 高置信步骤 ±win_half 步 → 高分辨率区（全文）
             - 其余 → 压缩区（JSON 摘要）
             - Dense Soft Voting 定罪逻辑

    参数
    ────
    target_seg_size    : 每个探针段的目标步骤数，默认 15
    max_probe_k        : 探针段数上限，默认 8
    win_half           : 高分辨率窗口半径，固定 +-3
    PROBE_CONF_THRESH  : 进入 DAO 的最低置信度，默认 0.5
    TOP_K_FLAGGED      : 进入 DAO 的最多步骤数，默认 3
    CONVICTION_THRESHOLD: DAO 定罪门槛，默认 0.5
    """
    from Lib.api_utils import (_load_json_data, _get_sorted_json_files,
                                _make_api_call, _make_api_call_with_retry,
                                _make_api_DAO_1_call, _make_api_DAO_2_call,
                                _make_api_DAO_3_call)

    WIN_HALF            = 3
    PROBE_CONF_THRESH   = 0.5
    TOP_K_FLAGGED       = 3
    CONVICTION_THRESHOLD = 0.5

    print("\n--- [HC-Enhance] Starting AgentJury Half-Split Probe + Multi-Window DAO ---\n")
    json_files = _get_sorted_json_files(directory_path)

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data      = _load_json_data(file_path)
        if not data:
            continue

        history_raw  = data.get("history", [])
        problem      = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        # ── 构建 chat_history（跳过 human，保留 history_index）────────────────
        chat_history = []
        for h_idx, entry in enumerate(history_raw):
            raw_role   = entry.get("role", "")
            normalized = _normalize_agent_name(raw_role)
            if normalized.lower() == "human":
                continue
            chat_history.append({
                "content":         entry.get("content", ""),
                "normalized_name": normalized,
                "raw_role":        raw_role,
                "history_index":   h_idx,
            })

        if not chat_history:
            continue

        # ── 团队成员与角色描述 ────────────────────────────────────────────────
        roles_dict             = _parse_team_from_initial_plan(history_raw)
        valid_agents_from_plan = list(roles_dict.keys()) if roles_dict else []
        roles_desc             = ""

        print(f"--- [HC-Enhance] Analyzing: {json_file} ---")
        print(f"    Planned team (from initial plan): {valid_agents_from_plan}")

        # 关键修复：取 roles_dict 与实际出场 Agent 的交集
        # Orchestrator 初始计划可能声明了全团队，但很多 case 只用到部分 Agent，
        # 挂名但未出场的成员会导致 DAO 产生幻觉评分并稀释软投票结果
        actual_participants = set(e["normalized_name"] for e in chat_history)
        if valid_agents_from_plan:
            valid_agents = [a for a in valid_agents_from_plan if a in actual_participants]
            if not valid_agents:
                valid_agents = _get_effective_agents_hc(history_raw)
        else:
            valid_agents = _get_effective_agents_hc(history_raw)

        # Orchestrator 是隐式协调者，不会出现在 "assembled the following team" 里，
        # 但在对话里始终参与，需要手动补回
        for p in sorted(actual_participants):
            if "orchestrator" in p.lower() and p not in valid_agents:
                valid_agents.append(p)

        print(f"    Active participants (intersected): {valid_agents}")

        # roles_desc：只保留实际出场 agent 的角色描述，过滤挂名未出场成员
        if roles_dict:
            for k, v in roles_dict.items():
                if k in valid_agents:
                    roles_desc += f"--- Role: {k} ---\n{v}\n"
        if not roles_desc:
            roles_desc = "No explicit role descriptions available. Infer from agent names and behavior.\n"

        # ════════════════════════════════════════════════════════════════════
        # Phase 1: 分层上下文摘要预计算（支持缓存复用）
        # ════════════════════════════════════════════════════════════════════
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        file_idx          = os.path.splitext(json_file)[0]
        abstract_filename = f"structured_abstract_hc_{file_idx}.json"
        abstract_path     = os.path.join(output_dir, abstract_filename)

        step_summaries = [""] * len(chat_history)

        if os.path.exists(abstract_path):
            print(f"  [Phase1 Cache HIT] Loading {abstract_path}")
            with open(abstract_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            hist_idx_to_chat_idx = {e["history_index"]: i
                                     for i, e in enumerate(chat_history)}
            for item in cached:
                chat_pos = hist_idx_to_chat_idx.get(item["step_index"])
                if chat_pos is not None:
                    step_summaries[chat_pos] = item.get("summary", "")
        else:
            print(f"  [Phase1] Generating summaries for {json_file} ...")
            summary_sys = (
                "You are a structured data extractor for a multi-agent conversation log. "
                "Output ONLY a valid JSON object (no markdown, no ```json) with EXACTLY these three keys:\n"
                "- 'Upstream_Instruction_or_Context': The instruction or context this agent received (max 1 sentence).\n"
                "- 'Node_Action': What this agent decided, navigated, or retrieved (max 1 sentence).\n"
                "- 'Result_and_Feedback': The direct outcome — page content, sub-task result, or conclusion (max 1 sentence).\n\n"
                "ROLE FORMAT NOTES:\n"
                "- 'Orchestrator (thought)': internal planning/reasoning log.\n"
                "- 'Orchestrator (-> X)': instruction dispatched to sub-agent X.\n"
                "- 'WebSurfer'/'FileSurfer': action-taking agents; summarize their action and result.\n\n"
                "### FEW-SHOT EXAMPLES ###\n"
                "Example — WebSurfer (failed navigation):\n"
                "Input: Agent: WebSurfer\nContent: I clicked 'NY Jidokwan Taekwondo' but landed on a KEYENCE microscopy product page.\n"
                "Output: {\"Upstream_Instruction_or_Context\": \"Instructed to click martial arts school links.\","
                " \"Node_Action\": \"Clicked 'NY Jidokwan Taekwondo' link, redirected to irrelevant KEYENCE product page.\","
                " \"Result_and_Feedback\": \"Navigation failed; no martial arts information retrieved.\"}\n\n"
                "Example — Orchestrator (thought) ledger update:\n"
                "Input: Agent: Orchestrator (thought)\nContent: Updated Ledger: {\"is_in_loop\": {\"answer\": true, \"reason\": \"WebSurfer repeated same off-target action.\"}}\n"
                "Output: {\"Upstream_Instruction_or_Context\": \"Reviewing WebSurfer's last failed navigation attempt.\","
                " \"Node_Action\": \"Updated ledger: loop detected, WebSurfer navigated off-target twice.\","
                " \"Result_and_Feedback\": \"Issuing corrective instruction to redirect WebSurfer.\"}"
            )

            for s_idx in range(len(chat_history)):
                s_agent   = chat_history[s_idx]["normalized_name"]
                s_content = chat_history[s_idx]["content"]

                past_ctx = ""
                for i in range(max(0, s_idx - 2), s_idx):
                    if step_summaries[i]:
                        past_ctx += (f"[Step {chat_history[i]['history_index']}] "
                                     f"{chat_history[i]['normalized_name']}: {step_summaries[i]}\n")

                s_content_trunc = s_content[:3000] + ("...[TRUNCATED]" if len(s_content) > 3000 else "")
                summary_user = ""
                if past_ctx:
                    summary_user += f"--- Previous Steps Context ---\n{past_ctx}\n\n"
                summary_user += f"--- Current Step ---\nAgent: {s_agent}\nContent: {s_content_trunc}"

                msg = [{"role": "system", "content": summary_sys},
                       {"role": "user",   "content": summary_user}]
                res = _make_api_DAO_3_call(client, model, msg, 512)
                step_summaries[s_idx] = res if res else "(Failed)"

            abstract_list = [
                {"step_index": chat_history[i]["history_index"],
                 "agent":      chat_history[i]["normalized_name"],
                 "summary":    step_summaries[i]}
                for i in range(len(chat_history))
            ]
            with open(abstract_path, "w", encoding="utf-8") as f:
                json.dump(abstract_list, f, ensure_ascii=False, indent=2)
            print(f"  [Phase1] Abstract saved → {abstract_path}")

        # ════════════════════════════════════════════════════════════════════
        # Phase 2: Adaptive K-Split Probe（自适应 K 段并行探针扫描）
        # ════════════════════════════════════════════════════════════════════
        import math as _math

        N = len(chat_history)
        # 自适应推算：每段目标 target_seg_size 步，K 上限 max_probe_k，下限 2
        K = max(2, min(max_probe_k, _math.ceil(N / target_seg_size)))
        K = min(K, N)  # 不超过实际步骤数
        seg_size = max(1, N // K)

        # 划分 K 段（最后一段吸收余数）
        segments_chat = []
        segments_sums = []
        for _k in range(K):
            _s = _k * seg_size
            _e = (_k + 1) * seg_size if _k < K - 1 else N
            segments_chat.append(chat_history[_s:_e])
            segments_sums.append(step_summaries[_s:_e])

        # ── 探针公共提示词指令 ────────────────────────────────────────────────
        probe_rules = (
            "Apply these evaluation rules BEFORE assigning any confidence score:\n"
            "0. [ORCHESTRATOR-FIRST CHECK — MANDATORY] Before flagging ANY WebSurfer or FileSurfer step, "
            "look at the Orchestrator dispatch step that issued the instruction. "
            "Ask: 'Was the instruction itself wrong or misleading?' "
            "If YES → flag the Orchestrator dispatch step (confidence ≥ 0.6) and assign the tool agent confidence ≤ 0.15. "
            "Only flag WebSurfer/FileSurfer directly if the instruction was correct but the agent deviated from it.\n"
            "1. [RETRY CORRECTION] If a step shows a failed navigation/retrieval, "
            "but the context reveals the SAME agent subsequently retried the SAME task "
            "and succeeded, it is a transient failure, not a lock-in.\n"
            "If multiple retries were performed after an error was reported in the log, select the step that ultimately led to a decisive failure.\n"
            "2. [LOCK-IN CRITERION] Assign confidence ≥ 0.6 when the step's wrong output "
            "was ACCEPTED by a downstream agent with NO subsequent correction visible in context.\n"
            "Before listing suspicious steps, reason through these checks in order:\n"
            "[CHECK 1 — Task Failure] What did the task require? Why did the final output fail to meet the ground truth?\n"
            "[CHECK 2 — Orchestrator Instruction Audit] For each orchestrator (->X) dispatch step: Is the instruction correct? If incorrect with no subsequent retries, flag this step.\n"
            "[CHECK 3 — Per-Step Review] For each tool agent step: did the agent correctly follow its received instruction? Did it produce a wrong or misleading output?\n"
            "[CHECK 4 — Retry Scan] For each candidate step: does any later step show the same agent retrying the same subtask? If yes, downgrade confidence — it is a transient failure.\n"
            "[CHECK 5 — Lock-in Test] Which step's wrong output was accepted by downstream agents with no subsequent correction? That is the true lock-in point.\n\n"
            "Output your reasoning for the checks above, then output the JSON.\n"
            "List ALL suspicious steps found in the FOCAL WINDOW (up to 5), sorted by confidence descending:\n"
            "{\n"
            "  \"suspicious_steps\": [\n"
            "    {\"step\": <history_index_int>, \"agent\": \"<normalized_name>\", "
            "\"confidence\": <0.0-1.0>, \"reason\": \"<one sentence>\"},\n"
            "    ...\n"
            "  ]\n"
            "}\n"
            "Do NOT collapse multiple candidates — list all suspicious steps found. "
            "If no steps are suspicious, output: {\"suspicious_steps\": []}"
        )

        valid_agents_str = ", ".join(f'"{a}"' for a in valid_agents)
        probe_sys = (
            "You are a fault attribution scanner for a FAILED multi-agent task. "
            "Your goal is to identify ALL suspicious steps in the assigned FOCAL WINDOW that may have "
            "contributed to the final failure — ranked by confidence. "
            "Candidates include: LOCK-IN steps (wrong output produced and accepted with no subsequent "
            "correction), wrong delegation steps (Orchestrator issued incorrect instruction), "
            "and premature termination steps. "
            "Use RECENT PAST and DISTANT FUTURE summaries only to judge whether a candidate "
            "was later corrected or originated upstream.\n\n"
            f"AGENT SCOPE RESTRICTION: The ONLY agents that participated are: [{valid_agents_str}]. "
            "You MUST NOT flag or name any agent outside this list. "
            "Orchestrator internal plans may mention other agents by name — ignore those names entirely."
        )

        # ── K 段并行探针调用 ──────────────────────────────────────────────────
        def _run_probe_seg(seg_k: int) -> str | None:
            focal_chat = segments_chat[seg_k]
            focal_sums = segments_sums[seg_k]

            past_chat = [e for s in segments_chat[:seg_k] for e in s]
            past_sums = [s for ss in segments_sums[:seg_k] for s in ss]
            future_chat = [e for s in segments_chat[seg_k + 1:] for e in s]
            future_sums = [s for ss in segments_sums[seg_k + 1:] for s in ss]

            focal_start = focal_chat[0]["history_index"]
            focal_end   = focal_chat[-1]["history_index"]

            focal_text  = _build_probe_half_text(focal_chat,  focal_sums)
            past_text   = _build_summary_text(past_chat,   past_sums)   if past_chat   else ""
            future_text = _build_summary_text(future_chat, future_sums) if future_chat else ""

            probe_user = f"Task: {problem}\n\n"
            probe_user += (f"You are scanning SEGMENT {seg_k + 1}/{K} of the log "
                           f"(Steps {focal_start}–{focal_end}).\n")
            if past_text:
                past_end_label = past_chat[-1]["history_index"]
                probe_user += (
                    f"The RECENT PAST section shows compressed summaries of Steps 0–{past_end_label} "
                    f"— use them to detect if errors in this segment originated upstream.\n\n"
                    f"--- RECENT PAST (JSON Summaries, Steps 0–{past_end_label}) ---\n"
                    f"{past_text}\n\n"
                )
            probe_user += (
                f"--- FOCAL WINDOW (FULL TEXT, Steps {focal_start}–{focal_end}) ---\n"
                f"{focal_text}\n\n"
            )
            if future_text:
                future_start_label = future_chat[0]["history_index"]
                future_end_label   = future_chat[-1]["history_index"]
                probe_user += (
                    f"--- DISTANT FUTURE  (JSON Summaries, Steps {future_start_label}–{future_end_label}) ---\n"
                    f"— use them to detect if this-segment errors were corrected by later retries.\n"
                    f"{future_text}\n\n"
                )
            probe_user += probe_rules

            msg = [{"role": "system", "content": probe_sys},
                   {"role": "user",   "content": probe_user}]
            return _make_api_call_with_retry(client, "ds-v3.2", msg,
                                             max_tokens=2048, thinking=True)

        print(f"  [Phase2] K-Split Probe: K={K}, N={N} steps, seg_size≈{seg_size}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=K) as _exe:
            _futs      = {_exe.submit(_run_probe_seg, k): k for k in range(K)}
            probe_raws = [None] * K
            for _f in concurrent.futures.as_completed(_futs):
                probe_raws[_futs[_f]] = _f.result()

        # ── 跨调用去重合并：同一步骤取最高置信度 ─────────────────────────────
        all_flagged: dict = {}   # hist_idx → flag_dict
        for k, raw in enumerate(probe_raws):
            flags = _parse_probe_output(raw)
            print(f"    Segment {k + 1} flags: {[(f['step'], f['confidence']) for f in flags]}")
            for flag in flags:
                s = flag["step"]
                if s < 0:
                    continue
                if s not in all_flagged or flag["confidence"] > all_flagged[s]["confidence"]:
                    all_flagged[s] = flag

        # 过滤低于阈值 + 取 top-K
        qualified = [f for f in all_flagged.values()
                     if f["confidence"] >= PROBE_CONF_THRESH]
        qualified.sort(key=lambda x: x["confidence"], reverse=True)
        top_flags = qualified[:TOP_K_FLAGGED]

        print(f"  [Phase2] Merged flagged steps (conf ≥ {PROBE_CONF_THRESH}): "
              f"{[(f['step'], f['confidence']) for f in top_flags]}")

        # ── 无可疑步骤兜底 ────────────────────────────────────────────────────
        if not top_flags:
            first_entry      = chat_history[0]
            first_exec_agent = first_entry["normalized_name"]
            first_exec_step  = first_entry["history_index"]
            print(f"  [Phase2] No suspicious steps found. Activating Step-0 Failsafe.")
            print(f"\nPrediction for {json_file}: Error found.")
            print(f"Agent Name: {first_exec_agent}")
            print(f"Step Number: {first_exec_step}")
            print(f"Reason: (HC-Enhance Failsafe) Probe found no suspicious steps.")
            continue

        # hist_idx → chat_idx 映射（供 _build_merged_dao_context 使用）
        hist_to_chat = {e["history_index"]: i for i, e in enumerate(chat_history)}

        top_flagged_tuples = []
        for f in top_flags:
            hist_idx   = f["step"]
            chat_idx   = hist_to_chat.get(hist_idx, -1)
            if chat_idx < 0:
                continue
            agent_name = f.get("agent", chat_history[chat_idx]["normalized_name"])
            top_flagged_tuples.append(
                (chat_idx, hist_idx, agent_name, f["confidence"], f["reason"])
            )

        # ════════════════════════════════════════════════════════════════════
        # Phase 3: 多窗口合并上下文 → 单次 DAO 三专家并行调用
        # ════════════════════════════════════════════════════════════════════
        merged_context = _build_merged_dao_context(
            chat_history, step_summaries, top_flagged_tuples, win_half=WIN_HALF
        )

        # 取置信度最高的步骤作为"探针锚点"（供 base_context 中的描述使用）
        anchor        = top_flagged_tuples[0]
        anchor_hist   = anchor[1]
        anchor_agent  = anchor[2]
        anchor_reason = anchor[4]

        base_context = (
            "[BACKGROUND] Post-mortem investigation of a FAILED multi-agent task. "
            "The agents below did NOT produce the correct final answer.\n\n"
            f"General Task Origin: {problem}\n"
            f"Active Agents in This Case:\n{roles_desc}\n"
            "SCOPE CONSTRAINT: Only evaluate agents listed above.\n\n"
            "Note: 'Orchestrator (thought)' are internal planning logs; "
            "'Orchestrator (-> X)' are dispatched instructions.\n\n"
            "Step numbers are original history indices (0-based). "
            "Output the EXACT [Step N] number from the log — do NOT re-index.\n\n"
            "--- CASE DOSSIER (Multi-Window Hierarchical Log) ---\n"
            f"{merged_context}\n"
            "--- ATTRIBUTION CONSTRAINTS ---\n"
            "Liability doctrines:\n"
            "[ORCHESTRATOR-FIRST PRINCIPLE — CHECK THIS FIRST]: Before attributing fault to any "
            "WebSurfer or FileSurfer step, ALWAYS examine the preceding Orchestrator (-> X) dispatch. "
            "If the Orchestrator issued a wrong URL, wrong search query, wrong target, or wrong instruction, "
            "the Orchestrator is the root cause and the tool agent is merely a manifestation point. "
            "Assign the tool agent error_likelihood ≤ 0.2 in this case.\n"
            "[Proximal Cause Attribution]: Identify the step where the error was FIRST COMMITTED "
            "(wrong data introduced / wrong action first taken), not where it was observed downstream.\n"
            "[ROOT CAUSE vs MANIFESTATION]: A downstream agent that faithfully executed a wrong "
            "instruction is a manifestation point. The agent that ISSUED the wrong instruction is the root cause.\n"
            "[TOOL-AGENT DIRECTION DOCTRINE]: WebSurfer/FileSurfer are instruction-directed. "
            "If they correctly executed a wrong instruction → blame the directing Orchestrator. "
            "If they went to a different destination than instructed → blame the tool agent.\n"
            "  KEY QUESTION: 'Did WebSurfer/FileSurfer do what it was told?' YES → blame instructor. NO → blame tool.\n"
            "[RETRY SEQUENCE PROTECTION]: If a flagged step is an intermediate retry and the log "
            "shows subsequent attempts by the same agent, assign it LOW error_likelihood (≤ 0.2). "
            "The true root cause is the LAST attempt whose output was accepted as final, "
            "or the Orchestrator step that terminated the task with unverified information.\n"
        )

        task_and_format_instruction = (
            "NOTE ON LOG FORMAT: The case dossier uses a two-tier format. "
            "Steps marked '(FULL TEXT)' are shown in full detail. "
            "Steps marked '(JSON SUMMARY)' are compressed to one-sentence abstracts.\n\n"
            "Output your response as valid JSON in <json></json> tags:\n"
            "{\n"
            '  "agent_evaluations": [\n'
            '    {"agent_name": "Orchestrator", "step_index": 5, "error_likelihood": 0.9, "reasoning": "issued wrong URL"},\n'
            '    {"agent_name": "Orchestrator", "step_index": 12, "error_likelihood": 0.6, "reasoning": "terminated prematurely"},\n'
            '    {"agent_name": "WebSurfer", "step_index": 7, "error_likelihood": 0.3, "reasoning": "followed wrong instruction"}\n'
            '  ]\n'
            "}\n"
            "For each agent, output ONE ENTRY PER SUSPICIOUS STEP — an agent may appear multiple times "
            "if it made errors at multiple steps. Sort all entries by error_likelihood descending. "
            "step_index must be the EXACT [Step N] number from the log — do NOT re-index. "
            "Flag every step where an agent produced a wrong output, issued a wrong instruction, "
            "or accepted unverified information — do NOT collapse multiple suspicious steps into one entry."
        )


        # Expert 1: Action Execution & Resource Access
        sys_1 = (
            "You are an Action Execution and Resource Access Attribution Expert. "
            "Focus on failures where an agent performed the WRONG physical action or "
            "failed to perform a required physical action entirely.\n"
            "FAILURE PATTERNS: wrong destination navigated, required action skipped, "
            "wrong resource accessed, wrong concrete instruction issued to downstream agent.\n"
            "Analyze ALL agents. Attribute fault to the agent whose physical action (or absence) "
            "is the direct cause. Provide confidence scores with evidence.\n\n"
            "Before outputting JSON, reason through:\n"
            "[B — Action Lens] Among all agents, which took a WRONG PHYSICAL ACTION "
            "(navigated to wrong URL, clicked wrong element, accessed wrong resource, "
            "issued wrong concrete instruction to a downstream agent)? "
            "Then determine: was this agent misexecuting a correct instruction it received, "
            "or faithfully executing a wrong instruction issued by an upstream agent? "
            "If the latter, the upstream instructor is the root cause, not this agent."
        )
        dao_1 = (base_context
                 + "\n[YOUR DIMENSION] ACTION EXECUTION & RESOURCE ACCESS Expert.\n\n"
                 + task_and_format_instruction)

        # Expert 2: Reasoning, Planning & Process Control
        sys_2 = (
            "You are a Reasoning, Planning, and Process Control Attribution Expert. "
            "Focus on failures rooted in flawed internal reasoning, wrong decisions, "
            "premature conclusions, or incorrect delegation.\n"
            "FAILURE PATTERNS: wrong instructions issued, premature task termination, "
            "failure to verify prior results, incorrect algorithm or logic applied, "
            "task delegated to wrong or incapable agent.\n"
            "Analyze ALL agents. Attribute fault to the agent whose reasoning or decision "
            "is the root cause. Provide confidence scores with evidence.\n\n"
            "Before outputting JSON, reason through:\n"
            "[B — Decision Lens] Among all agents, which made a WRONG DECISION "
            "(issued a wrong instruction to a tool agent, terminated the task prematurely, "
            "delegated to the wrong agent, or failed to verify a prior result before acting on it)? "
            "If a tool agent made a physical error, trace back: was it because the Orchestrator "
            "gave a wrong directive? If so, the Orchestrator's decision step is the root cause."
        )
        dao_2 = (base_context
                 + "\n[YOUR DIMENSION] REASONING, PLANNING & PROCESS CONTROL Expert.\n\n"
                 + task_and_format_instruction)

        # Expert 3: Information Retrieval Completeness & Quality
        sys_3 = (
            "You are an Information Retrieval Completeness and Quality Attribution Expert. "
            "Focus on failures where an agent reached the RIGHT destination but obtained "
            "WRONG, INCOMPLETE, or INSUFFICIENT information.\n"
            "FAILURE PATTERNS: correct resource accessed but wrong data extracted, "
            "over-reliance on unverified source, insufficient search parameters, "
            "partial page processing, outdated information retrieved.\n"
            "Do NOT cover wrong-navigation failures (→ Expert 1) or reasoning/logic failures (→ Expert 2).\n"
            "Analyze ALL agents. Attribute fault to the agent whose information quality failure "
            "is the root cause. Provide confidence scores with evidence.\n\n"
            "Before outputting JSON, reason through:\n"
            "[B — Information Lens] Among all agents, which retrieved or extracted "
            "WRONG or INCOMPLETE information despite reaching the correct source "
            "(read the wrong data field, processed only a partial page, relied on an unverified claim, "
            "or used outdated information)? "
            "Explicitly exclude wrong-navigation failures — only flag cases where the destination "
            "was correct but the information quality caused the final failure."
        )
        dao_3 = (base_context
                 + "\n[YOUR DIMENSION] INFORMATION RETRIEVAL COMPLETENESS & QUALITY Expert.\n\n"
                 + task_and_format_instruction)

        print(f"  [Phase3] Launching 3 DAO experts (parallel) ...")

        def _call_dao_enhance(args):
            i, sys_p, usr_p = args
            msg = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
            if i == 0: return _make_api_DAO_1_call(client, model, msg, max_tokens)
            if i == 1: return _make_api_DAO_2_call(client, model, msg, max_tokens)
            return _make_api_DAO_3_call(client, model, msg, max_tokens)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futs    = {executor.submit(_call_dao_enhance, (i, s, u)): i
                       for i, (s, u) in enumerate([(sys_1, dao_1),
                                                    (sys_2, dao_2),
                                                    (sys_3, dao_3)])}
            results = [None] * 3
            for f in concurrent.futures.as_completed(futs):
                results[futs[f]] = f.result()

        # ════════════════════════════════════════════════════════════════════
        # Phase 4: Dense Soft Voting（与原函数逻辑一致）
        # ════════════════════════════════════════════════════════════════════
        agent_scores = {a: 0.0 for a in valid_agents}
        step_votes   = {}
        judge_votes  = []
        valid_resp   = 0

        for i, r in enumerate(results):
            if not r:
                continue
            print(f"--- Arbitrator {i+1} response ---\n{r}\n--- End ---\n")
            try:
                json_match = re.search(r'<json>(.*?)</json>', r, re.DOTALL)
                if json_match:
                    jstr = json_match.group(1).strip()
                else:
                    md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', r, re.DOTALL)
                    jstr = (md_match.group(1).strip() if md_match
                            else re.search(r'\{.*\}', r, re.DOTALL).group(0))
                jdata = json.loads(jstr)

                local_ev_scores = {}
                for ev in jdata.get('agent_evaluations', []):
                    ev_agent      = ev.get('agent_name', '')
                    ev_likelihood = float(ev.get('error_likelihood', 0.0))
                    ev_step       = str(ev.get('step_index', anchor_hist))
                    # 幻觉防护：跳过不在实际出场名单中的 Agent
                    if not ev_agent or ev_agent not in valid_agents:
                        if ev_agent:
                            print(f"  [Hallucination guard] Skipped phantom agent: '{ev_agent}'")
                        continue
                    agent_scores[ev_agent] = agent_scores.get(ev_agent, 0.0) + ev_likelihood
                    local_ev_scores[ev_agent] = (local_ev_scores.get(ev_agent, 0.0)
                                                  + ev_likelihood)
                    step_votes[ev_step] = step_votes.get(ev_step, 0.0) + ev_likelihood

                if local_ev_scores:
                    judge_agent = max(local_ev_scores, key=local_ev_scores.get)
                    best_ev = max(
                        (ev for ev in jdata.get('agent_evaluations', [])
                         if ev.get('agent_name', '') == judge_agent),
                        key=lambda ev: float(ev.get('error_likelihood', 0.0)),
                        default={}
                    )
                    judge_step = str(best_ev.get('step_index', anchor_hist))
                    judge_conf = float(best_ev.get('error_likelihood', 0.0))
                    judge_votes.append((judge_agent, judge_step, judge_conf))
                valid_resp += 1
            except Exception:
                pass

        error_found = False

        if valid_resp > 0:
            top_score = max(agent_scores.values()) if agent_scores else 0.0
            avg_top   = top_score / valid_resp

            if avg_top >= CONVICTION_THRESHOLD:
                final_culprit = max(agent_scores, key=agent_scores.get)

                culprit_step_votes = {}
                for jg_agent, jg_step, jg_conf in judge_votes:
                    if jg_agent == final_culprit:
                        culprit_step_votes[jg_step] = (
                            culprit_step_votes.get(jg_step, 0.0) + jg_conf)

                if culprit_step_votes:
                    best_step_str = max(culprit_step_votes, key=culprit_step_votes.get)
                elif step_votes:
                    valid_steps = [str(e["history_index"]) for e in chat_history
                                   if e["normalized_name"] == final_culprit]
                    f_votes = {k: v for k, v in step_votes.items() if k in valid_steps}
                    if f_votes:
                        max_v      = max(f_votes.values())
                        candidates = [k for k, v in f_votes.items() if v == max_v]
                        best_step_str = min(candidates,
                                            key=lambda x: int(x) if x.isdigit() else 0)
                    else:
                        best_step_str = max(step_votes, key=step_votes.get)
                else:
                    best_step_str = str(anchor_hist)

                try:
                    culprit_step = int(best_step_str)
                except ValueError:
                    culprit_step = anchor_hist

                # Ghost-step 防护
                valid_hist_steps = [str(e["history_index"]) for e in chat_history
                                    if e["normalized_name"] == final_culprit]
                if valid_hist_steps and str(culprit_step) not in valid_hist_steps:
                    culprit_step = min(int(s) for s in valid_hist_steps)
                    print(f"  [Ghost-step guard] Step corrected → {culprit_step}")

                print(f"\nPrediction for {json_file}: Error found.")
                print(f"Agent Name: {final_culprit}")
                print(f"Step Number: {culprit_step}")
                print(f"Reason: Convicted by DAO Tribunal (Dense Soft Vote, avg={avg_top:.2f})")
                error_found = True

            else:
                print(f"  DAO absolved all candidates "
                      f"(avg_top={avg_top:.2f} < {CONVICTION_THRESHOLD}).")
        else:
            print("  All DAO calls failed.")

        if not error_found:
            first_entry      = chat_history[0]
            first_exec_agent = first_entry["normalized_name"]
            first_exec_step  = first_entry["history_index"]
            print(f"\nPrediction for {json_file}: Error found (Failsafe).")
            print(f"Agent Name: {first_exec_agent}")
            print(f"Step Number: {first_exec_step}")
            print(f"Reason: (HC-Enhance Failsafe) No confident conviction reached.")
