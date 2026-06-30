"""
alg_melting.py
==============
Ablation study variants for AgentJury_alg_enhance.

Reference: AgentJury_alg_enhance  (GT version, k-partition probe + data-driven DAO)

V1 — w/o Sliding Spotlight  (AgentJury_ablation_no_spotlight)
    Change : Phase 2 replaced with all-full-text expansion (no probe, no selective filtering).
    Control: Phase 1 / Phase 3 / Phase 4 identical to reference.

V2 — w/ Static Heuristic Experts  (AgentJury_ablation_static_experts)
    Change : Phase 3 expert system prompts are completely empty strings (no specialization).
    Control: Phase 1 / Phase 2 / Phase 4 identical to reference.

V3 — w/o Recursive Localization  (AgentJury_ablation_no_probe)
    Change : Phase 2 skipped entirely; DAO receives all-summary context (no full-text zone).
    Control: Phase 1 / Phase 3 / Phase 4 identical to reference.
"""

import os
import math as _math
import concurrent.futures as _cf
import re as _re
import json as _json

from tqdm import tqdm

from Lib.api_utils import (
    _get_sorted_json_files,
    _load_json_data,
    _make_api_call_with_retry,
    _make_api_DAO_1_call,
    _make_api_DAO_2_call,
    _make_api_DAO_3_call,
)

# ──────────────────────────────────────────────────────────────────────────────
# Module-level shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _est(text: str) -> int:
    return len(text) // 4 if text else 0


def _msgs_in(messages: list) -> int:
    return sum(_est(m.get("content", "")) for m in messages)


def _is_ct(name: str) -> bool:
    return name.lower() == "computer_terminal"


def _build_half_text(half_chat: list) -> str:
    parts = []
    for entry in half_chat:
        h_idx = entry["history_index"]
        name_ = entry["normalized_name"]
        cont_ = entry["content"]
        parts.append(f"[Step {h_idx}] {name_}:\n{cont_}\n")
    return "\n".join(parts)


def _build_summary_txt(half_chat: list, half_summaries: list) -> str:
    parts = []
    for entry, summary in zip(half_chat, half_summaries):
        h_idx = entry["history_index"]
        name_ = entry["normalized_name"]
        parts.append(f"[Step {h_idx}] {name_}: (JSON SUMMARY) {summary}")
    return "\n".join(parts)


def _parse_probe_out(raw: str) -> list:
    if not raw:
        return []
    try:
        m = _re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, _re.DOTALL)
        if not m:
            m = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if not m:
            return []
        grp = m.group(1) if (m.lastindex and m.lastindex >= 1) else m.group(0)
        data = _json.loads(grp)
        result = []
        for s in data.get("suspicious_steps", []):
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


def _build_dao_context_selective(chat_history: list, summaries: list,
                                  top_flagged: list, win_half: int) -> str:
    """Standard multi-window hierarchical context (reference behaviour)."""
    high_res = set()
    for (chat_idx, _, _, _, _) in top_flagged:
        for off in range(-win_half, win_half + 1):
            pos = chat_idx + off
            if 0 <= pos < len(chat_history):
                high_res.add(pos)

    lines = []
    in_hr = None
    for i, entry in enumerate(chat_history):
        h_idx = entry["history_index"]
        name_ = entry["normalized_name"]
        cont_ = entry["content"]
        if i in high_res:
            if in_hr is not True:
                lines.append("--- [HIGH-RESOLUTION ZONE — Full Text] ---")
            disp = cont_[:2500] + ("...[TRUNCATED]" if len(cont_) > 2500 else "")
            lines.append(f"[Step {h_idx}] {name_}: (FULL TEXT)\n{disp}\n")
            in_hr = True
        else:
            if in_hr is not False:
                lines.append("--- [COMPRESSED ZONE — JSON Summaries] ---")
            lines.append(f"[Step {h_idx}] {name_}: (JSON SUMMARY) {summaries[i]}")
            in_hr = False
    return "\n".join(lines)


def _build_dao_context_all_full(chat_history: list) -> str:
    """V1: every step is shown in full text (no probe-based selection)."""
    lines = ["--- [FULL TEXT — All Steps] ---"]
    for entry in chat_history:
        h_idx = entry["history_index"]
        name_ = entry["normalized_name"]
        cont_ = entry["content"]
        disp = cont_[:2500] + ("...[TRUNCATED]" if len(cont_) > 2500 else "")
        lines.append(f"[Step {h_idx}] {name_}: (FULL TEXT)\n{disp}\n")
    return "\n".join(lines)


def _build_dao_context_all_summary(chat_history: list, summaries: list) -> str:
    """V3: every step shown as a JSON summary (no high-res zone at all)."""
    lines = ["--- [COMPRESSED — All Steps as JSON Summaries] ---"]
    for i, entry in enumerate(chat_history):
        h_idx = entry["history_index"]
        name_ = entry["normalized_name"]
        lines.append(f"[Step {h_idx}] {name_}: (JSON SUMMARY) {summaries[i]}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Shared: Phase 1 summary cache (identical across all ablation variants)
# ──────────────────────────────────────────────────────────────────────────────

_SUMMARY_SYS = (
    "You are a structured data extractor. Output ONLY a valid JSON object (no markdown). "
    "The JSON MUST have EXACTLY these three keys:\n"
    "- 'Upstream_Instruction_or_Context': The instruction or context this agent received (max 1 sentence).\n"
    "- 'Node_Action': What this agent decided or coded (max 1 sentence).\n"
    "- 'Result_and_Feedback': The direct output, resolution, or sandbox result (max 1 sentence).\n\n"
    "### FEW-SHOT EXAMPLES ###\n\n"
    "-- Example 1: Domain Expert — code success --\n"
    "Input:\nAgent: Excel_Expert\n"
    "Content: Write python code to count even-address clients.\n"
    "```python\neven = data[data['Street Number'] % 2 == 0]\nprint(len(even))\n```\n"
    "[Execution Result from Sandbox]:\nexitcode: 0\nCode output: 4\n\n"
    "Output:\n"
    "{\"Upstream_Instruction_or_Context\": \"Count clients with even street numbers.\","
    " \"Node_Action\": \"Wrote Pandas modulo-2 filter script.\","
    " \"Result_and_Feedback\": \"Sandbox returned 4 (exitcode=0).\"}\n\n"
    "-- Example 2: Domain Expert — code failure --\n"
    "Input:\nAgent: Boggle_Expert\n"
    "Content: DFS to find longest word.\n"
    "```python\nword = find_longest(board, dictionary)\nprint(word)\n```\n"
    "[Execution Result from Sandbox]:\nexitcode: 1\nCode output: NameError: 'dictionary'\n\n"
    "Output:\n"
    "{\"Upstream_Instruction_or_Context\": \"Find longest Boggle word with DFS.\","
    " \"Node_Action\": \"Wrote DFS script referencing undefined 'dictionary' variable.\","
    " \"Result_and_Feedback\": \"Sandbox failed (exitcode=1): NameError.\"}"
)


def _run_phase1(client, json_file, chat_history, cache_path, summaries_cache):
    """
    Phase 1: step summarisation with dataset-level cache (shared across all variants).
    Returns list[str] of length len(chat_history).
    """
    if json_file in summaries_cache:
        cached = summaries_cache[json_file]
        if len(cached) != len(chat_history):
            cached = (cached + [""] * len(chat_history))[:len(chat_history)]
        print(f"  [Phase1 Cache HIT] Loaded summaries for {json_file}")
        return cached, {"summary_in": 0, "summary_out": 0}

    print(f"  [Phase1] Generating summaries for {json_file} ...")
    step_summaries = [""] * len(chat_history)
    tok = {"summary_in": 0, "summary_out": 0}

    for s_idx, entry in enumerate(chat_history):
        s_agent   = entry["normalized_name"]
        s_content = entry["content"]

        past_ctx = ""
        for pi in range(max(0, s_idx - 2), s_idx):
            if step_summaries[pi] and "bundled" not in step_summaries[pi]:
                past_ctx += (f"[Step {chat_history[pi]['history_index']}] "
                             f"{chat_history[pi]['normalized_name']}: {step_summaries[pi]}\n")

        s_content_trunc = s_content[:3000] + ("...[TRUNCATED]" if len(s_content) > 3000 else "")
        summary_user = ""
        if past_ctx:
            summary_user += f"--- Previous Steps Context ---\n{past_ctx}\n\n"
        summary_user += f"--- Current Step ---\nAgent: {s_agent}\nContent: {s_content_trunc}"

        msg = [{"role": "system", "content": _SUMMARY_SYS},
               {"role": "user",   "content": summary_user}]
        tok["summary_in"] += _msgs_in(msg)
        res = _make_api_call_with_retry(client, "ds-v3.2", msg, max_tokens=512, thinking=True)
        step_summaries[s_idx] = res if res else "(Failed)"
        tok["summary_out"] += _est(res)

    summaries_cache[json_file] = step_summaries
    try:
        with open(cache_path, "w", encoding="utf-8") as _f:
            _json.dump(summaries_cache, _f, ensure_ascii=False, indent=2)
        print(f"  [Phase1] Cache saved → {cache_path}")
    except Exception as _e:
        print(f"  [Phase1] Warning: cache save failed: {_e}")

    return step_summaries, tok


# ──────────────────────────────────────────────────────────────────────────────
# Shared: DAO expert system prompts (same for V1 and V3)
# ──────────────────────────────────────────────────────────────────────────────

_DAO_SYS_BASE = (
    "You are an Objective Analysis Agent conducting an impartial investigation "
    "to determine error attribution in a multi-agent conversation.\n\n"
    "ANALYST SPECIALIZATION: {focus}\n\n"
    "Your task:\n"
    "1. Analyze ALL agents objectively\n"
    "2. Identify which agent bears primary liability for the final incorrect outcome\n"
    "3. Pinpoint the exact step where the failure was FIRST committed (not observed)\n"
    "4. Assign error_likelihood scores [0.0–1.0] with evidence-backed reasoning\n\n"
    "Liability doctrines (MUST apply before concluding):\n"
    "- [OBSERVABLE ERROR THRESHOLD]: Convict ONLY with concrete observable manifestation.\n"
    "- [ROOT CAUSE vs MANIFESTATION]: Trace the error back to its true origin.\n"
    "- [STRICT EXECUTOR EXCEPTION]: Computer_terminal is exempt.\n"
    "- [EXECUTION ACCOUNTABILITY]: If downstream agent failed to verify, liability may transfer.\n\n"
    "Steps are sequential 0-based indices."
)

_FOCUS_FACTUAL = (
    "Factual and Information Retrieval Attribution Expert.\n\n"
    "FAILURE PATTERNS YOU PRIORITIZE (n=70 empirical cluster):\n"
    "- Hallucinated content introduced without source verification\n"
    "- Reliance on outdated/static knowledge instead of dynamic retrieval\n"
    "- Incorrect temporal or sequential data extraction (wrong year, wrong version)\n"
    "- Incorrect data schema, column reference, or structural element targeting\n"
    "- Failure to verify against authoritative source before propagating\n"
    "- Tool selection failure: inappropriate tool for retrieval context\n"
    "- Incorrect transcription or misinterpretation of source data\n\n"
    "BOUNDARY: Do NOT attribute blame for reasoning/logic or coordination failures.\n\n"
    "Before outputting JSON, reason through:\n"
    "[B — Factual Lens] Which agent introduced UNVERIFIED or HALLUCINATED information? "
    "Did any agent rely on stale internal knowledge instead of fetching from the authoritative source? "
    "Trace the origin of incorrect data backward to where it was first introduced."
)

_FOCUS_LOGIC = (
    "Logic, Reasoning, and Planning Attribution Expert.\n\n"
    "FAILURE PATTERNS YOU PRIORITIZE (n=55 empirical cluster):\n"
    "- Incorrect algorithm or formula implementation\n"
    "- Incorrect initial task decomposition or goal formulation\n"
    "- Premature conclusion without completing required verification steps\n"
    "- Logical inconsistency or incorrect interpretation of problem constraints\n"
    "- Incorrect problem decomposition or task allocation among agents\n"
    "- Incorrect verification of intermediate results\n\n"
    "BOUNDARY: Do NOT blame data quality or inter-agent coordination failures.\n\n"
    "Before outputting JSON, reason through:\n"
    "[B — Decision Lens] Which agent made a WRONG REASONING DECISION "
    "(issued wrong instructions based on flawed analysis, terminated prematurely, "
    "applied incorrect algorithm, or failed to verify prior results before acting)? "
    "If a downstream agent erred, trace back: was it because an upstream planner gave a wrong directive?"
)

_FOCUS_EXECUTION = (
    "Agent Execution and Coordination Attribution Expert.\n\n"
    "FAILURE PATTERNS YOU PRIORITIZE (n=59 empirical cluster):\n"
    "- Agent provided inaccurate foundational data to downstream agents\n"
    "- Agent executed code with incorrect or non-functional logic\n"
    "- Cross-agent verification failure: downstream executed erroneous output unchecked\n"
    "- Agent bypassed required analysis and made unverified assumption\n"
    "- Agent action omission: skipped required step causing information gap\n"
    "- Inaccurate or fabricated source data introduced and consumed by others\n\n"
    "BOUNDARY: Do NOT blame reasoning quality or raw data issues.\n\n"
    "Before outputting JSON, reason through:\n"
    "[B — Execution Lens] Which agent produced WRONG OUTPUT from an execution step "
    "(wrote incorrect code, introduced fabricated data, produced wrong computation result, "
    "or silently passed incorrect data downstream)? "
    "Distinguish agents who wrote bad code vs. those who faithfully executed a bad instruction."
)

_TASK_AND_FORMAT = (
    "NOTE ON LOG FORMAT: Steps marked '(FULL TEXT)' are shown in full detail. "
    "Steps marked '(JSON SUMMARY)' are compressed to one-sentence abstracts.\n\n"
    "Output your response as valid JSON in <json></json> tags:\n"
    "{\n"
    '  "agent_evaluations": [\n'
    '    {"agent_name": "Planner", "step_index": 2, "error_likelihood": 0.9, '
    '"reasoning": "introduced hallucinated census data without verification"},\n'
    '    {"agent_name": "Planner", "step_index": 5, "error_likelihood": 0.6, '
    '"reasoning": "terminated task prematurely based on wrong intermediate result"},\n'
    '    {"agent_name": "Validator", "step_index": 8, "error_likelihood": 0.3, '
    '"reasoning": "failed to catch upstream hallucination despite verification role"}\n'
    '  ]\n'
    "}\n"
    "if it made errors at multiple steps. Sort all entries by error_likelihood descending. "
    "step_index must be the EXACT [Step N] number from the log — do NOT re-index. "
    "Flag every step where an agent produced wrong output, wrote flawed code, introduced "
    "hallucinated data, or accepted unverified information. "
    "Do NOT collapse multiple suspicious steps into one entry."
)

_PROBE_RULES = (
    "Before listing suspicious steps, reason through these checks in order:\n"
    "[CHECK 1 — Task Failure] What did the task require? Why did the final output fail to meet the ground truth?\n"
    "[CHECK 2 — Instruction Audit] For each planning step: was the instruction to downstream agents correct? "
    "If wrong, this is the root cause regardless of how the executor performed.\n"
    "[CHECK 3 — Per-Step Review] For each executor step: did the agent correctly follow its instruction? "
    "Did it produce wrong or hallucinated output?\n"
    "[CHECK 4 — Retry Scan] For each candidate: does any later step show the same agent retrying? "
    "If yes, downgrade confidence — transient failure, not lock-in.\n"
    "[CHECK 5 — Lock-in Test] Which step's wrong output was accepted downstream with no correction? "
    "That is the true lock-in point.\n\n"
    "Output your reasoning for the checks above, then output the JSON.\n"
    "List ALL suspicious steps in the window (up to 5), sorted by confidence descending:\n"
    "{\n"
    "  \"suspicious_steps\": [\n"
    "    {\"step\": <history_index_int>, \"agent\": \"<agent_name>\", "
    "\"confidence\": <0.0-1.0>, \"reason\": \"<one sentence>\"},\n"
    "    {\"step\": <history_index_int>, \"agent\": \"<agent_name>\", "
    "\"confidence\": <0.0-1.0>, \"reason\": \"<one sentence>\"}\n"
    "  ]\n"
    "}\n"
    "Do NOT collapse multiple candidates — list all if 3 suspicious steps exist. "
    "If no steps are suspicious: {\"suspicious_steps\": []}"
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared: Phase 4 Dense Soft Voting (identical across all variants)
# ──────────────────────────────────────────────────────────────────────────────

def _run_phase4(results, valid_agents, chat_history, anchor_hist,
                json_file, conviction_threshold, label):
    """
    Runs Dense Soft Voting from DAO results.
    Returns (agent_name, step_number, error_found: bool).
    """
    agent_scores = {a: 0.0 for a in valid_agents}
    step_votes   = {}
    judge_votes  = []
    valid_resp   = 0

    for i, r in enumerate(results):
        if not r:
            continue
        print(f"--- Arbitrator {i+1} response ---\n{r}\n--- End ---\n")
        try:
            json_match = _re.search(r'<json>(.*?)</json>', r, _re.DOTALL)
            if json_match:
                jstr = json_match.group(1).strip()
            else:
                md_match = _re.search(r'```(?:json)?\s*(\{.*?\})\s*```', r, _re.DOTALL)
                jstr = (md_match.group(1).strip() if md_match
                        else _re.search(r'\{.*\}', r, _re.DOTALL).group(0))
            jdata = _json.loads(jstr)

            local_ev_scores = {}
            for ev in jdata.get('agent_evaluations', []):
                ev_agent      = ev.get('agent_name', '')
                ev_likelihood = float(ev.get('error_likelihood', 0.0))
                ev_step       = str(ev.get('step_index', anchor_hist))
                if not ev_agent or ev_agent not in valid_agents:
                    if ev_agent:
                        print(f"  [Hallucination guard] Skipped phantom agent: '{ev_agent}'")
                    continue
                agent_scores[ev_agent] = agent_scores.get(ev_agent, 0.0) + ev_likelihood
                local_ev_scores[ev_agent] = local_ev_scores.get(ev_agent, 0.0) + ev_likelihood
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

    if valid_resp == 0:
        print("  All DAO calls failed.")
        return None, None, False

    top_score = max(agent_scores.values()) if agent_scores else 0.0
    avg_top   = top_score / valid_resp

    if avg_top < conviction_threshold:
        print(f"  DAO absolved all candidates (avg_top={avg_top:.2f} < {conviction_threshold}).")
        return None, None, False

    final_culprit = max(agent_scores, key=agent_scores.get)

    culprit_step_votes = {}
    for jg_agent, jg_step, jg_conf in judge_votes:
        if jg_agent == final_culprit:
            culprit_step_votes[jg_step] = culprit_step_votes.get(jg_step, 0.0) + jg_conf

    if culprit_step_votes:
        best_step_str = max(culprit_step_votes, key=culprit_step_votes.get)
    elif step_votes:
        valid_steps = [str(e["history_index"]) for e in chat_history
                       if e["normalized_name"] == final_culprit]
        f_votes = {k: v for k, v in step_votes.items() if k in valid_steps}
        if f_votes:
            max_v      = max(f_votes.values())
            candidates = [k for k, v in f_votes.items() if v == max_v]
            best_step_str = min(candidates, key=lambda x: int(x) if x.isdigit() else 0)
        else:
            best_step_str = max(step_votes, key=step_votes.get)
    else:
        best_step_str = str(anchor_hist)

    try:
        culprit_step = int(best_step_str)
    except ValueError:
        culprit_step = anchor_hist

    # Ghost-step protection
    valid_hist_steps = [str(e["history_index"]) for e in chat_history
                        if e["normalized_name"] == final_culprit]
    if valid_hist_steps and str(culprit_step) not in valid_hist_steps:
        culprit_step = min(int(s) for s in valid_hist_steps)
        print(f"  [Ghost-step guard] Step corrected → {culprit_step}")

    print(f"\nPrediction for {json_file}: Error found.")
    print(f"Agent Name: {final_culprit}")
    print(f"Step Number: {culprit_step}")
    print(f"Reason: Convicted by DAO Tribunal ({label}, avg={avg_top:.2f})")
    return final_culprit, culprit_step, True


# ──────────────────────────────────────────────────────────────────────────────
# Shared: k-partition concurrent probe (identical in V2 vs reference)
# ──────────────────────────────────────────────────────────────────────────────

def _run_phase2_kprobe(client, chat_history, step_summaries, problem, ground_truth,
                       valid_agents, probe_k,
                       probe_conf_thresh=0.3, top_k_flagged=3):
    """
    k-partition concurrent probe (same as reference AgentJury_alg_enhance).
    Returns (top_flagged_tuples, hist_to_chat, tok_in, tok_out).
    """
    N        = len(chat_history)
    actual_k = min(probe_k, N)
    part_sz  = _math.ceil(N / actual_k)

    partition_ranges = []
    for _p in range(actual_k):
        _s = _p * part_sz
        _e = min(_s + part_sz, N)
        if _s < N:
            partition_ranges.append((_s, _e))

    valid_agents_str = ", ".join(f'"{a}"' for a in valid_agents)
    probe_sys = (
        "You are a fault attribution scanner for a FAILED multi-agent task. "
        "Identify ALL suspicious steps in the assigned window that contributed to the final failure "
        "— output up to 3 candidates, ranked by confidence descending. "
        "Tasks may involve retry sequences; NEVER mark an early failed attempt if any retry follows.\n\n"
        f"AGENT SCOPE RESTRICTION: Only these agents participated in this task: [{valid_agents_str}]. "
        "You MUST NOT flag or name any agent outside this list. "
        "Think step by step before producing the final JSON output."
    )

    def _run_one_probe(p_idx, p_start, p_end):
        part_entries   = chat_history[p_start:p_end]
        part_summaries = step_summaries[p_start:p_end]
        past_entries   = chat_history[:p_start]
        past_summaries = step_summaries[:p_start]
        fut_entries    = chat_history[p_end:]
        fut_summaries  = step_summaries[p_end:]

        part_hist_start = part_entries[0]["history_index"]
        part_hist_end   = part_entries[-1]["history_index"]
        part_full_text  = _build_half_text(part_entries)

        sections = [
            f"Task: {problem}\n"
            f"Ground Truth: {ground_truth}\n\n"
            f"You are scanning PARTITION {p_idx + 1}/{len(partition_ranges)} of the log "
            f"(Steps {part_hist_start}–{part_hist_end}).\n"
        ]
        if past_entries:
            ph_s = past_entries[0]["history_index"]
            ph_e = past_entries[-1]["history_index"]
            sections.append(
                f"The PAST section shows compressed summaries of Steps {ph_s}–{ph_e}"
                f" — Use them for cross-validation and counterfactual reasoning.\n"
            )
        if fut_entries:
            fh_s = fut_entries[0]["history_index"]
            fh_e = fut_entries[-1]["history_index"]
            sections.append(
                f"The FUTURE section shows compressed summaries of Steps {fh_s}–{fh_e}"
                f" — Use them for cross-validation and counterfactual reasoning.\n"
            )
        sections.append("\n")
        if past_entries:
            ph_s = past_entries[0]["history_index"]
            ph_e = past_entries[-1]["history_index"]
            sections.append(
                f"--- PAST CONTEXT (JSON Summaries, Steps {ph_s}–{ph_e}) ---\n"
                f"{_build_summary_txt(past_entries, past_summaries)}\n\n"
            )
        sections.append(
            f"--- CURRENT PARTITION (FULL TEXT, Steps {part_hist_start}–{part_hist_end}) ---\n"
            f"{part_full_text}\n\n"
        )
        if fut_entries:
            fh_s = fut_entries[0]["history_index"]
            fh_e = fut_entries[-1]["history_index"]
            sections.append(
                f"--- FUTURE CONTEXT (JSON Summaries, Steps {fh_s}–{fh_e}) ---\n"
                f"{_build_summary_txt(fut_entries, fut_summaries)}\n\n"
            )
        sections.append(_PROBE_RULES)
        probe_user = "".join(sections)

        msg = [{"role": "system", "content": probe_sys},
               {"role": "user",   "content": probe_user}]
        _in_tok = _msgs_in(msg)
        raw = _make_api_call_with_retry(client, "ds-v3.2", msg, max_tokens=2048, thinking=True)
        return raw, _in_tok

    print(f"  [Phase2] k={len(partition_ranges)} partition probe (concurrent) ...")
    all_raw_flags = []
    tok_in = 0
    tok_out = 0
    with _cf.ThreadPoolExecutor(max_workers=len(partition_ranges)) as _exec:
        _fut_map = {
            _exec.submit(_run_one_probe, _pi, _ps, _pe): _pi
            for _pi, (_ps, _pe) in enumerate(partition_ranges)
        }
        for _fut in _cf.as_completed(_fut_map):
            _pi = _fut_map[_fut]
            _raw, _in_tok = _fut.result()
            tok_in  += _in_tok
            tok_out += _est(_raw)
            _flags = _parse_probe_out(_raw)
            print(f"    Partition {_pi + 1}: flagged "
                  f"{[(_f['step'], _f['confidence']) for _f in _flags]}")
            all_raw_flags.extend(_flags)

    # Dedup merge
    all_flagged: dict = {}
    for flag in all_raw_flags:
        s = flag["step"]
        if s < 0:
            continue
        if s not in all_flagged or flag["confidence"] > all_flagged[s]["confidence"]:
            all_flagged[s] = flag

    qualified = [f for f in all_flagged.values() if f["confidence"] >= probe_conf_thresh]
    qualified.sort(key=lambda x: x["confidence"], reverse=True)
    top_flags = qualified[:top_k_flagged]

    print(f"  [Phase2] Merged flagged steps (conf ≥ {probe_conf_thresh}): "
          f"{[(f['step'], f['confidence']) for f in top_flags]}")

    hist_to_chat = {e["history_index"]: idx for idx, e in enumerate(chat_history)}
    top_flagged_tuples = []
    for f in top_flags:
        hist_idx = f["step"]
        chat_idx = hist_to_chat.get(hist_idx, -1)
        if chat_idx < 0:
            continue
        agent_name = f.get("agent", chat_history[chat_idx]["normalized_name"])
        top_flagged_tuples.append(
            (chat_idx, hist_idx, agent_name, f["confidence"], f["reason"])
        )

    return top_flagged_tuples, hist_to_chat, tok_in, tok_out


# ──────────────────────────────────────────────────────────────────────────────
# Shared: DAO Phase 3 — 3 experts parallel
# ──────────────────────────────────────────────────────────────────────────────

def _run_phase3_dao(client, model, max_tokens, sys_1, sys_2, sys_3,
                    dao_1, dao_2, dao_3):
    """Run 3 DAO experts in parallel, return [r1, r2, r3]."""
    def _call_dao(args):
        i, sys_p, usr_p = args
        msg = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
        if i == 0: return _make_api_DAO_1_call(client, model, msg, max_tokens)
        if i == 1: return _make_api_DAO_2_call(client, model, msg, max_tokens)
        return _make_api_DAO_3_call(client, model, msg, max_tokens)

    results = [None] * 3
    with _cf.ThreadPoolExecutor(max_workers=3) as executor:
        futs = {executor.submit(_call_dao, (i, s, u)): i
                for i, (s, u) in enumerate([(sys_1, dao_1), (sys_2, dao_2), (sys_3, dao_3)])}
        for f in _cf.as_completed(futs):
            results[futs[f]] = f.result()
    return results


# ──────────────────────────────────────────────────────────────────────────────
# V1: w/o Sliding Spotlight
# ──────────────────────────────────────────────────────────────────────────────

def AgentJury_ablation_no_spotlight(client, directory_path: str, model: str,
                                    max_tokens: int):
    """
    Ablation V1 — w/o Sliding Spotlight.

    Isolated change: Phase 2 (k-partition probe + selective full-text expansion) is
    replaced by all-full-text context fed directly to the DAO.  The DAO sees the entire
    log at full resolution without any probe-based filtering.  All other phases are
    identical to the reference AgentJury_alg_enhance.
    """
    CONVICTION_THRESHOLD = 0.3
    LABEL = "[ALG-Ablation-V1]"

    print(f"\n--- {LABEL} Starting AgentJury Ablation: w/o Sliding Spotlight ---\n")
    json_files = _get_sorted_json_files(directory_path)

    dataset_name = os.path.basename(os.path.normpath(directory_path))
    cache_dir    = os.path.join(os.path.dirname(__file__), "..", "outputs", "summaries_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path   = os.path.join(cache_dir, f"{dataset_name}_alg_enhance.json")

    summaries_cache: dict = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as _f:
                summaries_cache = _json.load(_f)
            print(f"[Phase1 Cache] Loaded {len(summaries_cache)} entries from {cache_path}\n")
        except Exception:
            summaries_cache = {}

    global_tok = {"summary_in": 0, "summary_out": 0, "dao_in": 0, "dao_out": 0}

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data      = _load_json_data(file_path)
        if not data:
            continue

        history_raw  = data.get("history", [])
        problem      = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        chat_history = []
        i = 0
        while i < len(history_raw):
            entry = history_raw[i]
            name  = entry.get("name", "Unknown")
            if _is_ct(name):
                i += 1
                continue
            content = entry.get("content", "")
            if i + 1 < len(history_raw) and _is_ct(history_raw[i + 1].get("name", "")):
                content += f"\n\n[Execution Result from Sandbox]:\n{history_raw[i + 1].get('content', '')}"
            chat_history.append({
                "content":         content,
                "normalized_name": name,
                "raw_role":        name,
                "history_index":   i,
            })
            i += 1

        if not chat_history:
            continue

        system_prompt_dict = data.get("system_prompt", {})
        valid_agents = list(system_prompt_dict.keys()) if isinstance(system_prompt_dict, dict) else []
        actual_participants = set(e["normalized_name"] for e in chat_history)
        for p in sorted(actual_participants):
            if p not in valid_agents:
                valid_agents.append(p)

        roles_desc = ""
        if isinstance(system_prompt_dict, dict):
            for k, v in system_prompt_dict.items():
                roles_desc += f"--- Role: {k} ---\n{v}\n"
        if not roles_desc:
            roles_desc = "No explicit role descriptions available. Infer from agent names and behavior.\n"

        print(f"--- {LABEL} Analyzing: {json_file} ---")

        # Phase 1
        step_summaries, tok1 = _run_phase1(
            client, json_file, chat_history, cache_path, summaries_cache
        )
        global_tok["summary_in"]  += tok1["summary_in"]
        global_tok["summary_out"] += tok1["summary_out"]

        # Phase 2 (REPLACED): all-full-text — no probe
        print(f"  [Phase2-V1] Skipping probe; building all-full-text context ...")
        merged_context = _build_dao_context_all_full(chat_history)
        anchor_hist    = chat_history[0]["history_index"]
        anchor_agent   = chat_history[0]["normalized_name"]
        anchor_reason  = "No probe; all-full-text context sent to DAO."

        # Phase 3: 3 DAO experts
        sys_1 = _DAO_SYS_BASE.format(focus=_FOCUS_FACTUAL)
        sys_2 = _DAO_SYS_BASE.format(focus=_FOCUS_LOGIC)
        sys_3 = _DAO_SYS_BASE.format(focus=_FOCUS_EXECUTION)

        base_context = (
            f"[BACKGROUND] Post-mortem investigation of a FAILED multi-agent task. "
            f"The agents below did NOT produce the correct final answer.\n\n"
            f"General Task Origin: {problem}\nGround Truth Expected Outcome: {ground_truth}\n"
            f"Active Agents in This Case:\n{roles_desc}\n"
            "SCOPE CONSTRAINT: Only evaluate agents listed above. "
            "Do NOT generate evaluations for any agent not present in the log.\n\n"
            "Note: 'Computer_terminal' is a sandbox environment — passive executor with no reasoning capacity.\n\n"
            "Step numbers are sequential 0-based indices in the history array. "
            "Output the EXACT [Step N] number you see in the log — do NOT re-index.\n\n"
            "--- CASE DOSSIER (All Steps — Full Text) ---\n"
            f"{merged_context}\n"
            "--- ATTRIBUTION CONSTRAINTS ---\n"
            "manifestation of error — incorrect values propagated, hallucinated data introduced, "
            "flawed code written, or a wrong plan executed.\n"
            "[ROOT CAUSE vs MANIFESTATION]: Distinguish where the error ORIGINATED from where it became "
            "OBSERVABLE. A downstream agent that faithfully executed a wrong instruction is a MANIFESTATION "
            "POINT — the agent that ISSUED the wrong instruction is the ROOT CAUSE.\n"
            "[PROXIMAL CAUSE]: Select the EARLIEST step where the responsible agent first produced a "
            "concretely wrong output — wrong data introduced, wrong code written, wrong instruction issued.\n"
            "[STRICT EXECUTOR EXCEPTION]: Computer_terminal is exempt from conviction for faithfully "
            "returning the output of an upstream command.\n"
        )


        dao_1 = (base_context
                 + "\n[YOUR DIMENSION] FACTUAL & INFORMATION RETRIEVAL Expert.\n\n"
                 + _TASK_AND_FORMAT)
        dao_2 = (base_context
                 + "\n[YOUR DIMENSION] LOGIC, REASONING & PLANNING Expert.\n\n"
                 + _TASK_AND_FORMAT)
        dao_3 = (base_context
                 + "\n[YOUR DIMENSION] AGENT EXECUTION & COORDINATION Expert.\n\n"
                 + _TASK_AND_FORMAT)

        for _s, _u in [(sys_1, dao_1), (sys_2, dao_2), (sys_3, dao_3)]:
            global_tok["dao_in"] += _msgs_in([{"role": "system", "content": _s},
                                               {"role": "user",   "content": _u}])

        print(f"  [Phase3-V1] Launching 3 DAO experts (parallel) ...")
        results = _run_phase3_dao(client, model, max_tokens,
                                  sys_1, sys_2, sys_3, dao_1, dao_2, dao_3)
        for r in results:
            global_tok["dao_out"] += _est(r)

        # Phase 4
        _, _, error_found = _run_phase4(
            results, valid_agents, chat_history, anchor_hist,
            json_file, CONVICTION_THRESHOLD, "Dense Soft Vote (V1-no-spotlight)"
        )

        if not error_found:
            fa = chat_history[0]["normalized_name"]
            fs = chat_history[0]["history_index"]
            print(f"\nPrediction for {json_file}: Error found (Failsafe).")
            print(f"Agent Name: {fa}")
            print(f"Step Number: {fs}")
            print(f"Reason: ({LABEL} Failsafe) No confident conviction reached.")

    grand_total = sum(global_tok.values())
    print("\n" + "=" * 50)
    print(f"{LABEL} === Global Token Cost Summary (Est.) ===")
    print(f"  Phase1 Summary : in={global_tok['summary_in']:,}  out={global_tok['summary_out']:,}")
    print(f"  Phase3 DAO     : in={global_tok['dao_in']:,}  out={global_tok['dao_out']:,}")
    print(f"  Grand Total (est): {grand_total:,} tokens")
    print("=" * 50)


# ──────────────────────────────────────────────────────────────────────────────
# V2: w/ Static Heuristic Experts (empty system prompts)
# ──────────────────────────────────────────────────────────────────────────────

def AgentJury_ablation_static_experts(client, directory_path: str, model: str,
                                      max_tokens: int, probe_k: int = 4):
    """
    Ablation V2 — w/ Static Heuristic Experts.

    Isolated change: Phase 3 DAO expert system prompts are completely empty strings
    (no specialisation, no focus dimension, no failure-pattern taxonomy).  All other
    phases — Phase 1 cache, Phase 2 k-partition probe, Phase 4 Dense Soft Voting —
    are identical to the reference AgentJury_alg_enhance.
    """
    WIN_HALF             = 2
    PROBE_CONF_THRESH    = 0.3
    TOP_K_FLAGGED        = 3
    CONVICTION_THRESHOLD = 0.3
    LABEL = "[ALG-Ablation-V2]"

    print(f"\n--- {LABEL} Starting AgentJury Ablation: Static Heuristic Experts ---\n")
    json_files = _get_sorted_json_files(directory_path)

    dataset_name = os.path.basename(os.path.normpath(directory_path))
    cache_dir    = os.path.join(os.path.dirname(__file__), "..", "outputs", "summaries_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path   = os.path.join(cache_dir, f"{dataset_name}_alg_enhance.json")

    summaries_cache: dict = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as _f:
                summaries_cache = _json.load(_f)
            print(f"[Phase1 Cache] Loaded {len(summaries_cache)} entries from {cache_path}\n")
        except Exception:
            summaries_cache = {}

    global_tok = {
        "summary_in": 0, "summary_out": 0,
        "probe_in":   0, "probe_out":   0,
        "dao_in":     0, "dao_out":     0,
    }

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data      = _load_json_data(file_path)
        if not data:
            continue

        history_raw  = data.get("history", [])
        problem      = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        chat_history = []
        i = 0
        while i < len(history_raw):
            entry = history_raw[i]
            name  = entry.get("name", "Unknown")
            if _is_ct(name):
                i += 1
                continue
            content = entry.get("content", "")
            if i + 1 < len(history_raw) and _is_ct(history_raw[i + 1].get("name", "")):
                content += f"\n\n[Execution Result from Sandbox]:\n{history_raw[i + 1].get('content', '')}"
            chat_history.append({
                "content":         content,
                "normalized_name": name,
                "raw_role":        name,
                "history_index":   i,
            })
            i += 1

        if not chat_history:
            continue

        system_prompt_dict = data.get("system_prompt", {})
        valid_agents = list(system_prompt_dict.keys()) if isinstance(system_prompt_dict, dict) else []
        actual_participants = set(e["normalized_name"] for e in chat_history)
        for p in sorted(actual_participants):
            if p not in valid_agents:
                valid_agents.append(p)

        roles_desc = ""
        if isinstance(system_prompt_dict, dict):
            for k, v in system_prompt_dict.items():
                roles_desc += f"--- Role: {k} ---\n{v}\n"
        if not roles_desc:
            roles_desc = "No explicit role descriptions available. Infer from agent names and behavior.\n"

        print(f"--- {LABEL} Analyzing: {json_file} ---")

        # Phase 1
        step_summaries, tok1 = _run_phase1(
            client, json_file, chat_history, cache_path, summaries_cache
        )
        global_tok["summary_in"]  += tok1["summary_in"]
        global_tok["summary_out"] += tok1["summary_out"]

        # Phase 2 (identical to reference)
        top_flagged_tuples, hist_to_chat, p2_in, p2_out = _run_phase2_kprobe(
            client, chat_history, step_summaries, problem, ground_truth,
            valid_agents, probe_k, PROBE_CONF_THRESH, TOP_K_FLAGGED
        )
        global_tok["probe_in"]  += p2_in
        global_tok["probe_out"] += p2_out

        if not top_flagged_tuples:
            fa = chat_history[0]["normalized_name"]
            fs = chat_history[0]["history_index"]
            print(f"  [Phase2] No suspicious steps. Activating Step 0 Planning Failsafe.")
            print(f"\nPrediction for {json_file}: Error found.")
            print(f"Agent Name: {fa}")
            print(f"Step Number: {fs}")
            print(f"Reason: ({LABEL} Failsafe) Probe found no suspicious steps.")
            continue

        # Phase 3: identical context, but sys prompts = "" (ABLATION CHANGE)
        merged_context = _build_dao_context_selective(
            chat_history, step_summaries, top_flagged_tuples, WIN_HALF
        )
        anchor        = top_flagged_tuples[0]
        anchor_hist   = anchor[1]
        anchor_agent  = anchor[2]
        anchor_reason = anchor[4]

        base_context = (
            f"[BACKGROUND] Post-mortem investigation of a FAILED multi-agent task. "
            f"The agents below did NOT produce the correct final answer.\n\n"
            f"General Task Origin: {problem}\nGround Truth Expected Outcome: {ground_truth}\n"
            f"Active Agents in This Case:\n{roles_desc}\n"
            "SCOPE CONSTRAINT: Only evaluate agents listed above. "
            "Do NOT generate evaluations for any agent not present in the log.\n\n"
            "Note: 'Computer_terminal' is a sandbox environment — passive executor with no reasoning capacity.\n\n"
            "Step numbers are sequential 0-based indices in the history array. "
            "Output the EXACT [Step N] number you see in the log — do NOT re-index.\n\n"
            "--- CASE DOSSIER (Multi-Window Hierarchical Log) ---\n"
            f"{merged_context}\n"
            "--- ATTRIBUTION CONSTRAINTS ---\n"
            "manifestation of error — incorrect values propagated, hallucinated data introduced, "
            "flawed code written, or a wrong plan executed.\n"
            "[ROOT CAUSE vs MANIFESTATION]: Distinguish where the error ORIGINATED from where it became "
            "OBSERVABLE. A downstream agent that faithfully executed a wrong instruction is a MANIFESTATION "
            "POINT — the agent that ISSUED the wrong instruction is the ROOT CAUSE.\n"
            "[PROXIMAL CAUSE]: Select the EARLIEST step where the responsible agent first produced a "
            "concretely wrong output — wrong data introduced, wrong code written, wrong instruction issued.\n"
            "[STRICT EXECUTOR EXCEPTION]: Computer_terminal is exempt from conviction for faithfully "
            "returning the output of an upstream command.\n"
        )


        dao_1 = (base_context
                 + "\n[YOUR DIMENSION] FACTUAL & INFORMATION RETRIEVAL Expert.\n\n"
                 + _TASK_AND_FORMAT)
        dao_2 = (base_context
                 + "\n[YOUR DIMENSION] LOGIC, REASONING & PLANNING Expert.\n\n"
                 + _TASK_AND_FORMAT)
        dao_3 = (base_context
                 + "\n[YOUR DIMENSION] AGENT EXECUTION & COORDINATION Expert.\n\n"
                 + _TASK_AND_FORMAT)

        # V2 ABLATION: expert system prompts are completely empty
        sys_1 = ""
        sys_2 = ""
        sys_3 = ""

        for _s, _u in [(sys_1, dao_1), (sys_2, dao_2), (sys_3, dao_3)]:
            global_tok["dao_in"] += _msgs_in([{"role": "system", "content": _s},
                                               {"role": "user",   "content": _u}])

        print(f"  [Phase3-V2] Launching 3 DAO experts with EMPTY system prompts (parallel) ...")
        results = _run_phase3_dao(client, model, max_tokens,
                                  sys_1, sys_2, sys_3, dao_1, dao_2, dao_3)
        for r in results:
            global_tok["dao_out"] += _est(r)

        # Phase 4
        _, _, error_found = _run_phase4(
            results, valid_agents, chat_history, anchor_hist,
            json_file, CONVICTION_THRESHOLD, "Dense Soft Vote (V2-static-experts)"
        )

        if not error_found:
            fa = chat_history[0]["normalized_name"]
            fs = chat_history[0]["history_index"]
            print(f"\nPrediction for {json_file}: Error found (Failsafe).")
            print(f"Agent Name: {fa}")
            print(f"Step Number: {fs}")
            print(f"Reason: ({LABEL} Failsafe) No confident conviction reached.")

    grand_total = sum(global_tok.values())
    print("\n" + "=" * 50)
    print(f"{LABEL} === Global Token Cost Summary (Est.) ===")
    print(f"  Phase1 Summary : in={global_tok['summary_in']:,}  out={global_tok['summary_out']:,}")
    print(f"  Phase2 Probe   : in={global_tok['probe_in']:,}  out={global_tok['probe_out']:,}")
    print(f"  Phase3 DAO     : in={global_tok['dao_in']:,}  out={global_tok['dao_out']:,}")
    print(f"  Grand Total (est): {grand_total:,} tokens")
    print("=" * 50)


# ──────────────────────────────────────────────────────────────────────────────
# V3: w/o Recursive Localization
# ──────────────────────────────────────────────────────────────────────────────

def AgentJury_ablation_no_probe(client, directory_path: str, model: str,
                                max_tokens: int):
    """
    Ablation V3 — w/o Recursive Localization.

    Isolated change: Phase 2 (probe) is skipped entirely.  The DAO receives all steps
    as JSON summaries only — no full-text zone, no selective expansion.  All other
    phases are identical to the reference AgentJury_alg_enhance.
    """
    CONVICTION_THRESHOLD = 0.3
    LABEL = "[ALG-Ablation-V3]"

    print(f"\n--- {LABEL} Starting AgentJury Ablation: w/o Recursive Localization ---\n")
    json_files = _get_sorted_json_files(directory_path)

    dataset_name = os.path.basename(os.path.normpath(directory_path))
    cache_dir    = os.path.join(os.path.dirname(__file__), "..", "outputs", "summaries_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path   = os.path.join(cache_dir, f"{dataset_name}_alg_enhance.json")

    summaries_cache: dict = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as _f:
                summaries_cache = _json.load(_f)
            print(f"[Phase1 Cache] Loaded {len(summaries_cache)} entries from {cache_path}\n")
        except Exception:
            summaries_cache = {}

    global_tok = {"summary_in": 0, "summary_out": 0, "dao_in": 0, "dao_out": 0}

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data      = _load_json_data(file_path)
        if not data:
            continue

        history_raw  = data.get("history", [])
        problem      = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        chat_history = []
        i = 0
        while i < len(history_raw):
            entry = history_raw[i]
            name  = entry.get("name", "Unknown")
            if _is_ct(name):
                i += 1
                continue
            content = entry.get("content", "")
            if i + 1 < len(history_raw) and _is_ct(history_raw[i + 1].get("name", "")):
                content += f"\n\n[Execution Result from Sandbox]:\n{history_raw[i + 1].get('content', '')}"
            chat_history.append({
                "content":         content,
                "normalized_name": name,
                "raw_role":        name,
                "history_index":   i,
            })
            i += 1

        if not chat_history:
            continue

        system_prompt_dict = data.get("system_prompt", {})
        valid_agents = list(system_prompt_dict.keys()) if isinstance(system_prompt_dict, dict) else []
        actual_participants = set(e["normalized_name"] for e in chat_history)
        for p in sorted(actual_participants):
            if p not in valid_agents:
                valid_agents.append(p)

        roles_desc = ""
        if isinstance(system_prompt_dict, dict):
            for k, v in system_prompt_dict.items():
                roles_desc += f"--- Role: {k} ---\n{v}\n"
        if not roles_desc:
            roles_desc = "No explicit role descriptions available. Infer from agent names and behavior.\n"

        print(f"--- {LABEL} Analyzing: {json_file} ---")

        # Phase 1
        step_summaries, tok1 = _run_phase1(
            client, json_file, chat_history, cache_path, summaries_cache
        )
        global_tok["summary_in"]  += tok1["summary_in"]
        global_tok["summary_out"] += tok1["summary_out"]

        # Phase 2: SKIPPED
        print(f"  [Phase2-V3] Probe skipped; building all-summary context ...")
        merged_context = _build_dao_context_all_summary(chat_history, step_summaries)
        anchor_hist    = chat_history[0]["history_index"]
        anchor_agent   = chat_history[0]["normalized_name"]
        anchor_reason  = "No probe; all-summary context sent to DAO."

        # Phase 3: 3 DAO experts with data-driven prompts, all-summary context
        sys_1 = _DAO_SYS_BASE.format(focus=_FOCUS_FACTUAL)
        sys_2 = _DAO_SYS_BASE.format(focus=_FOCUS_LOGIC)
        sys_3 = _DAO_SYS_BASE.format(focus=_FOCUS_EXECUTION)

        base_context = (
            f"[BACKGROUND] Post-mortem investigation of a FAILED multi-agent task. "
            f"The agents below did NOT produce the correct final answer.\n\n"
            f"General Task Origin: {problem}\nGround Truth Expected Outcome: {ground_truth}\n"
            f"Active Agents in This Case:\n{roles_desc}\n"
            "SCOPE CONSTRAINT: Only evaluate agents listed above. "
            "Do NOT generate evaluations for any agent not present in the log.\n\n"
            "Note: 'Computer_terminal' is a sandbox environment — passive executor with no reasoning capacity.\n\n"
            "Step numbers are sequential 0-based indices in the history array. "
            "Output the EXACT [Step N] number you see in the log — do NOT re-index.\n\n"
            "--- CASE DOSSIER (All Steps — JSON Summaries Only) ---\n"
            f"{merged_context}\n"
            "--- ATTRIBUTION CONSTRAINTS ---\n"
            "manifestation of error — incorrect values propagated, hallucinated data introduced, "
            "flawed code written, or a wrong plan executed.\n"
            "[ROOT CAUSE vs MANIFESTATION]: Distinguish where the error ORIGINATED from where it became "
            "OBSERVABLE. A downstream agent that faithfully executed a wrong instruction is a MANIFESTATION "
            "POINT — the agent that ISSUED the wrong instruction is the ROOT CAUSE.\n"
            "[PROXIMAL CAUSE]: Select the EARLIEST step where the responsible agent first produced a "
            "concretely wrong output — wrong data introduced, wrong code written, wrong instruction issued.\n"
            "[STRICT EXECUTOR EXCEPTION]: Computer_terminal is exempt from conviction for faithfully "
            "returning the output of an upstream command.\n"
        )


        dao_1 = (base_context
                 + "\n[YOUR DIMENSION] FACTUAL & INFORMATION RETRIEVAL Expert.\n\n"
                 + _TASK_AND_FORMAT)
        dao_2 = (base_context
                 + "\n[YOUR DIMENSION] LOGIC, REASONING & PLANNING Expert.\n\n"
                 + _TASK_AND_FORMAT)
        dao_3 = (base_context
                 + "\n[YOUR DIMENSION] AGENT EXECUTION & COORDINATION Expert.\n\n"
                 + _TASK_AND_FORMAT)

        for _s, _u in [(sys_1, dao_1), (sys_2, dao_2), (sys_3, dao_3)]:
            global_tok["dao_in"] += _msgs_in([{"role": "system", "content": _s},
                                               {"role": "user",   "content": _u}])

        print(f"  [Phase3-V3] Launching 3 DAO experts (parallel) ...")
        results = _run_phase3_dao(client, model, max_tokens,
                                  sys_1, sys_2, sys_3, dao_1, dao_2, dao_3)
        for r in results:
            global_tok["dao_out"] += _est(r)

        # Phase 4
        _, _, error_found = _run_phase4(
            results, valid_agents, chat_history, anchor_hist,
            json_file, CONVICTION_THRESHOLD, "Dense Soft Vote (V3-no-probe)"
        )

        if not error_found:
            fa = chat_history[0]["normalized_name"]
            fs = chat_history[0]["history_index"]
            print(f"\nPrediction for {json_file}: Error found (Failsafe).")
            print(f"Agent Name: {fa}")
            print(f"Step Number: {fs}")
            print(f"Reason: ({LABEL} Failsafe) No confident conviction reached.")

    grand_total = sum(global_tok.values())
    print("\n" + "=" * 50)
    print(f"{LABEL} === Global Token Cost Summary (Est.) ===")
    print(f"  Phase1 Summary : in={global_tok['summary_in']:,}  out={global_tok['summary_out']:,}")
    print(f"  Phase3 DAO     : in={global_tok['dao_in']:,}  out={global_tok['dao_out']:,}")
    print(f"  Grand Total (est): {grand_total:,} tokens")
    print("=" * 50)
