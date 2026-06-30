import os
import json
import re
import random
import concurrent.futures
from openai import AzureOpenAI
from tqdm import tqdm

API_MODEL_MAP ={
    "ds-v3.2":"deepseek-ai/DeepSeek-V3.2",
    "ds-r1":"deepseek-ai/DeepSeek-R1",
    "glm-4.7":"Pro/zai-org/GLM-4.7",
    "minimax-2.5":"Pro/MiniMaxAI/MiniMax-M2.5",
    "kimi-2.5":"Pro/moonshotai/Kimi-K2.5",
    "qwen-3.5":"Qwen/Qwen3.5-397B-A17B"
}

# --- Helper Functions ---

def _get_sorted_json_files(directory_path):
    """Gets and sorts JSON files numerically from a directory."""
    try:
        files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
        return []
    except Exception as e:
        print(f"Error reading or sorting files in {directory_path}: {e}")
        return []

def _load_json_data(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

import time

def _make_api_call_with_retry(client, model_name, messages, max_tokens=None, retries=3, temperature=0.5, thinking=False):
    """Makes an API call with automatic retries for network connection drops.
    max_tokens=None means no limit (model decides output length).
    thinking=True enables DeepSeek-V3.2 extended thinking mode (SiliconFlow)."""
    for attempt in range(retries):
        try:
            call_kwargs = dict(
                model=API_MODEL_MAP[model_name],
                messages=messages,
                stream=True,
                temperature=temperature,
            )
            if max_tokens is not None:
                call_kwargs["max_tokens"] = max_tokens
            if thinking:
                call_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            response = client.chat.completions.create(**call_kwargs)
            full_content = ""
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None):
                        full_content += delta.content
            return full_content.strip()
        except Exception as e:
            print(f"Error during API call for {model_name} (Attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2)  # Wait 2 seconds before retrying
            else:
                return None

def _make_api_call(client, model, messages, max_tokens, temperature=0.3):
    return _make_api_call_with_retry(client, "qwen-3.5", messages, max_tokens, temperature=temperature)

def _make_api_DAO_1_call(client, model, messages, max_tokens=None):
    # DAO output is unlimited: complex JSON verdict must not be truncated
    return _make_api_call_with_retry(client, "qwen-3.5", messages, max_tokens=None, thinking=False)

def _make_api_DAO_2_call(client, model, messages, max_tokens=None):
    return _make_api_call_with_retry(client, "qwen-3.5", messages, max_tokens=None, thinking=False)

def _make_api_DAO_3_call(client, model, messages, max_tokens=None):
    return _make_api_call_with_retry(client, "qwen-3.5", messages, max_tokens=None, thinking=False)

# --- All-at-Once Method ---

def all_at_once_api(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    Analyzes chat history by feeding the entire conversation at once to the model.
    """
    print("\n--- Starting All-at-Once Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "") # Keep ground truth if needed for evaluation

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        chat_content = "\n".join([
            f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}" for entry in chat_history
        ])

        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
            f"The problem is:  {problem}\n"
            f"The Answer for the problem is: {ground_truth}\n" # Included as per original code - remove if ground truth shouldn't be used in prompt
            "Identify which agent made an error, at which step, and explain the reason for the error. "
            "Here's the conversation:\n\n" + chat_content +
            "\n\nBased on this conversation, please predict the following:\n"
            "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
            "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
            """
            {
                "agent a": "xx",
                "agent b": "xxxx",
                "agent c": "xxxxx",
                "agent a": "xxxxxxx"
            },
            """
            "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
            "3. The reason for your prediction."
            "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n"
        )

        messages=[
            {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
            {"role": "user", "content": prompt},
        ]

        result = _make_api_call(client, model, messages, max_tokens)

        print(f"Prediction for {json_file}:")
        if result:
            print(result)
        else:
            print("Failed to get prediction.")
        print("\n" + "="*50 + "\n")

# --- Step-by-Step Method ---

def step_by_step_api(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    Analyzes chat history step by step, asking the model at each step if an error occurred.
    """
    print("\n--- Starting Step-by-Step Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "") # Keep ground truth if needed

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        print(f"--- Analyzing File: {json_file} ---")
        current_conversation_history = ""
        error_found = False
        for idx, entry in enumerate(chat_history):
            agent_name = entry.get(index_agent, 'Unknown Agent')
            content = entry.get('content', '')
            current_conversation_history += f"Step {idx} - {agent_name}: {content}\n"

            prompt = (
                f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                f"The Answer for the problem is: {ground_truth}\n" # Included as per original code - remove if ground truth shouldn't be used
                f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                f"The most recent step ({idx}) was by '{agent_name}'.\n"
                "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
            )

            messages=[
                {"role": "system", "content": "You are a precise step-by-step conversation evaluator."},
                {"role": "user", "content": prompt},
            ]

            print(f"Evaluating Step {idx} by {agent_name}...")
            answer = _make_api_call(client, model, messages, max_tokens)

            if not answer:
                print("Failed to get evaluation for this step. Stopping analysis for this file.")
                error_found = True # Treat API error as unable to proceed
                break

            print(f"LLM Evaluation: {answer}")

            # Basic check for "Yes" at the beginning of the response
            if answer.lower().strip().startswith("1. yes"):
                print(f"\nPrediction for {json_file}: Error found.")
                print(f"Agent Name: {agent_name}")
                print(f"Step Number: {idx}")
                print(f"Reason provided by LLM: {answer.split('Reason:', 1)[-1].strip()}")
                error_found = True
                break # Stop processing this file once an error is found
            elif answer.lower().strip().startswith("1. no"):
                 print("No significant error detected in this step.")
            else:
                print("Warning: Unexpected response format from LLM. Continuing evaluation.")
                # Optionally handle unexpected format more robustly

        if not error_found:
            print(f"\nNo decisive errors found by step-by-step analysis in file {json_file}")

        print("\n" + "="*50 + "\n")


# --- Binary Search Method ---

def _construct_binary_search_prompt(problem, answer, chat_segment_content, range_description, upper_half_desc, lower_half_desc):
    """Constructs the prompt for the binary search step."""
    return (
        "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
        "Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user’s query.\n"
        f"The problem to address is as follows: {problem}\n"
        f"The Answer for the problem is: {answer}\n" # Included as per original code - remove if ground truth shouldn't be used
        f"Review the following conversation segment {range_description}:\n\n{chat_segment_content}\n\n"
        f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half ({upper_half_desc}) or the lower half ({lower_half_desc}) of this segment.\n"
        "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'. Remember, your answer should be based on identifying the mistake that directly contributes to the failure in resolving the user's query. If no single clear error is evident, consider the step you believe is most responsible for the failure, allowing for subjective judgment, and base your answer on that."
    )

def _report_binary_search_error(chat_history, step, json_file, is_handcrafted):
    """Reports the identified error step from binary search."""
    index_agent = "role" if is_handcrafted else "name"
    entry = chat_history[step]
    agent_name = entry.get(index_agent, 'Unknown Agent')

    print(f"\nPrediction for {json_file}:")
    print(f"Agent Name: {agent_name}")
    print(f"Step Number: {step}")
    print("\n" + "="*50 + "\n")

def _find_error_in_segment_recursive(client: AzureOpenAI, model: str, max_tokens: int, chat_history: list, problem: str, answer: str, start: int, end: int, json_file: str, is_handcrafted: bool):
    """Recursive helper function for binary search analysis."""
    if start > end:
         print(f"Warning: Invalid range in binary search for {json_file} (start={start}, end={end}). Reporting last valid step.")
         _report_binary_search_error(chat_history, end if end >= 0 else 0, json_file, is_handcrafted) # Report something reasonable
         return
    if start == end:
        _report_binary_search_error(chat_history, start, json_file, is_handcrafted)
        return

    index_agent = "role" if is_handcrafted else "name"

    segment_history = chat_history[start : end + 1]
    if not segment_history:
        print(f"Warning: Empty segment in binary search for {json_file} (start={start}, end={end}). Cannot proceed.")
        _report_binary_search_error(chat_history, start, json_file, is_handcrafted)
        return

    chat_content = "\n".join([
        f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}"
        for entry in segment_history
    ])

    mid = start + (end - start) // 2 

    range_description = f"from step {start} to step {end}"
    upper_half_desc = f"from step {start} to step {mid}"
    lower_half_desc = f"from step {mid + 1} to step {end}"

    prompt = _construct_binary_search_prompt(problem, answer, chat_content, range_description, upper_half_desc, lower_half_desc)

    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in localizing errors in conversation segments."},
        {"role": "user", "content": prompt}
    ]

    print(f"Analyzing step {start}-{end} for {json_file}...")
    result = _make_api_call(client, model, messages, max_tokens)

    if not result:
        print(f"API call failed for segment {start}-{end}. Stopping binary search for {json_file}.")
        return

    print(f"LLM Prediction for segment {start}-{end}: {result}")
    result_lower = result.lower() 

    if "upper half" in result_lower:
         _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, start, mid, json_file, is_handcrafted)
    elif "lower half" in result_lower:
         new_start = min(mid + 1, end)
         _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, new_start, end, json_file, is_handcrafted)
    else:
        print(f"Warning: Ambiguous response '{result}' from LLM for segment {start}-{end}. Randomly choosing a half.")
        if random.randint(0, 1) == 0:
            print("Randomly chose upper half.")
            _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, start, mid, json_file, is_handcrafted)
        else:
            print("Randomly chose lower half.")
            new_start = min(mid + 1, end)
            _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, new_start, end, json_file, is_handcrafted)


def binary_search_api(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    Analyzes chat history using a binary search approach to find the error step.
    """
    print("\n--- Starting Binary Search Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        answer = data.get("ground_truth", "") # Keep ground truth if needed

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        print(f"--- Analyzing File: {json_file} ---")
        _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, 0, len(chat_history) - 1, json_file, is_handcrafted)

# --- Step-by-Step with DAO Arbitration Method ---

def step_by_step_dao_api(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    Analyzes chat history step by step. If an error is flagged, 3 DAO agents vote to confirm.
    """
    print("\n--- Starting Step-by-Step with DAO Arbitration Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")
        
        system_prompt_dict = data.get("system_prompt", {})
        valid_agents = list(system_prompt_dict.keys()) if isinstance(system_prompt_dict, dict) else []
        valid_agents_str = ", ".join(valid_agents) if valid_agents else "Not specified"
        
        roles_desc = ""
        if isinstance(system_prompt_dict, dict):
            for k, v in system_prompt_dict.items():
                roles_desc += f"--- Role: {k} ---\n{v}\n"

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        print(f"--- Analyzing File: {json_file} ---")
        current_conversation_history = ""
        error_found = False
        
        for idx, entry in enumerate(chat_history):
            agent_name = entry.get(index_agent, 'Unknown Agent')
            content = entry.get('content', '')
            current_conversation_history += f"Step {idx} - {agent_name}: {content}\n"

            prompt = (
                f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                f"The Answer for the problem is: {ground_truth}\n"
                f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                f"The most recent step ({idx}) was by '{agent_name}'.\n"
                "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
            )

            messages=[
                {"role": "system", "content": "You are a precise step-by-step conversation evaluator."},
                {"role": "user", "content": prompt},
            ]

            print(f"Accuser checking Step {idx} by {agent_name}...")
            answer_accuser = _make_api_call(client, model, messages, max_tokens)

            if not answer_accuser:
                print("Failed to get evaluation for this step. Stopping analysis for this file.")
                error_found = True
                break
                
            print(f"Accuser Evaluation: {answer_accuser}")

            answer_lower = answer_accuser.lower().strip()
            prefix_area = answer_lower.split('reason:', 1)[0] if 'reason:' in answer_lower else answer_lower
            
            if "yes" in prefix_area:
                try:
                    reason = answer_accuser.split('Reason:', 1)[-1].strip()
                except Exception:
                    reason = answer_accuser
                
                print(f"Accuser flagged an error. Reason: {reason[:50]}...\nInitiating DAO Arbitration...")
                
                dao_votes_tally = {agent: 0 for agent in valid_agents}
                dao_member_responses = [] # Store full responses (reasoning + JSON)

                base_prompt = (
                    f"You are an impartial Arbitrator reviewing an accusation in a multi-agent system.\n"
                    f"Problem: {problem}\nAnswer: {ground_truth}\n"
                    f"Conversation History:\n{current_conversation_history}\n"
                    f"### THE COMPOSITION OF THE MAS ###\n{roles_desc}\n"
                    f"An Accuser claims that the problem-solving process derailed at Step {idx} (executed by '{agent_name}') for the following reason:\n"
                    f"\"{reason}\"\n"
                    "CRITICAL INSTRUCTION: Analyze the ENTIRE conversation history to determine the root cause(s) of the failure based on these strict guidelines:\n"
                    "1. Evaluate EVERY agent's behavior. Assign a guilt probability reflecting your confidence in their responsibility.\n"
                    "2. You can distribute guilt among MULTIPLE agents if multiple agents contributed to the failure.\n"
                    "3. Do NOT blame 'Computer_terminal' (the execution environment). The error must lie with the Agent who wrote the flawed code or logic.\n"
                    "4. Also identify the STEP(S) most likely to contain the root cause error.\n"
                    f"You must output a JSON object with EXACTLY TWO keys:\n"
                    f"  - 'agents': a dict mapping each valid Agent ({valid_agents_str}) to an object with:\n"
                    f"      'score' [0.0, 1.0]: guilt probability (0.0=innocent, 1.0=definitely guilty)\n"
                    f"      'reason': one concise sentence explaining WHY this score was assigned.\n"
                    f"  - 'steps': a dict mapping suspect step numbers (as strings) to an object with:\n"
                    f"      'score' [0.0, 1.0]: suspicion score (1.0=this step is the root cause)\n"
                    f"      'reason': one concise sentence explaining what went wrong at this step.\n"
                    f"Example:\n"
                    f"{{\"agents\": {{\"Excel_Expert\": {{\"score\": 0.9, \"reason\": \"Used .astype(int) without NaN guard.\"}}, \"DataVerification_Expert\": {{\"score\": 0.1, \"reason\": \"No error detected.\"}}}}, \"steps\": {{\"0\": {{\"score\": 0.9, \"reason\": \"Root cause introduced here.\"}}, \"2\": {{\"score\": 0.1, \"reason\": \"Downstream effect only.\"}}}}}}\n"
                    "Respond ONLY with this JSON object and nothing else."
                )


                # --- Expert 1: Factual / Retrieval Expert ---
                sys_prompt_1 = (
                    "You are a Factual Attribution Expert specializing in identifying failures caused by "
                    "incorrect information retrieval, hallucinated content, unverified external data, or "
                    "inappropriate tool selection. Your analytical lens focuses exclusively on WHAT data was "
                    "retrieved or used, WHERE it came from, and WHETHER it was verified against ground truth "
                    "before being passed downstream. Patterns you prioritize: hallucination, reliance on "
                    "unverified sources, tool misuse, failure to verify information, incorrect data extraction."
                )
                dao_member_prompt_1 = (
                    base_prompt +
                    "\n[YOUR EXPERT DIMENSION] You are the FACTUAL / RETRIEVAL Expert. "
                    "Focus your analysis on whether any agent: (1) hallucinated or fabricated data, "
                    "(2) relied on unverified or incorrect external sources, (3) selected the wrong tool "
                    "for information retrieval, (4) failed to verify retrieved information against ground truth, "
                    "or (5) extracted incorrect or temporally mismatched data. "
                    "Only assign guilt based on evidence of factual or retrieval-dimension failures."
                )

                # --- Expert 2: Logic / Reasoning Expert ---
                sys_prompt_2 = (
                    "You are a Logic and Reasoning Attribution Expert specializing in identifying failures "
                    "caused by flawed internal reasoning, incorrect algorithm or formula implementation, "
                    "premature conclusions, task decomposition errors, or misaligned problem understanding. "
                    "Your analytical lens focuses on the agent's INTERNAL reasoning chain and decision logic, "
                    "NOT on its data sources. Patterns you prioritize: incorrect algorithm, premature conclusion, "
                    "logical inconsistency, task misidentification, flawed solution strategy, incorrect calculation."
                )
                dao_member_prompt_2 = (
                    base_prompt +
                    "\n[YOUR EXPERT DIMENSION] You are the LOGIC / REASONING Expert. "
                    "Focus your analysis on whether any agent: (1) applied an incorrect algorithm or formula, "
                    "(2) reached a premature or unsupported conclusion, (3) decomposed the task incorrectly, "
                    "(4) exhibited logical inconsistency or contradictory reasoning, or (5) misidentified the "
                    "problem domain or objective. Do NOT blame data quality; focus solely on reasoning-layer failures."
                )

                # --- Expert 3: Agent Execution / Coordination Expert ---
                sys_prompt_3 = (
                    "You are an Agent Execution and Coordination Attribution Expert specializing in identifying "
                    "failures caused by inter-agent coordination breakdowns, role boundary violations, code "
                    "execution errors by sub-agents, or cross-agent verification failures. Your analytical lens "
                    "focuses on HOW agents interact, delegate, and verify each other's outputs. Patterns you "
                    "prioritize: agent role violation, cross-agent verification failure, agent provided inaccurate "
                    "foundational data to downstream agents, agent exceeded step/resource limits, code execution "
                    "failure by a dependent agent, orchestrator over-delegation or under-specification."
                )
                dao_member_prompt_3 = (
                    base_prompt +
                    "\n[YOUR EXPERT DIMENSION] You are the AGENT EXECUTION / COORDINATION Expert. "
                    "Focus your analysis on whether any agent: (1) violated its role boundary or responsibilities, "
                    "(2) failed to verify or correctly pass outputs to downstream agents, (3) executed faulty or "
                    "non-functional code, (4) caused a cascading failure by providing incorrect foundational data "
                    "to other agents, or (5) exceeded computational/step limits due to repeated erroneous actions. "
                    "Focus on agent-level behavioral and coordination failures, not on data quality or reasoning logic."
                )

                dao_member_prompts = [dao_member_prompt_1, dao_member_prompt_2, dao_member_prompt_3]
                sys_prompts = [sys_prompt_1, sys_prompt_2, sys_prompt_3]
                
                # --- Concurrent DAO calls ---
                def _call_dao_member(args):
                    i, sys_p, user_p = args
                    dao_messages = [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}]
                    if i == 0:
                        result = _make_api_DAO_1_call(client, model, dao_messages, max_tokens)
                    elif i == 1:
                        result = _make_api_DAO_2_call(client, model, dao_messages, max_tokens)
                    else:
                        result = _make_api_DAO_3_call(client, model, dao_messages, max_tokens)
                    return i, result

                call_args = [(i, sys_prompts[i], dao_member_prompts[i]) for i in range(len(dao_member_prompts))]
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures_map = {executor.submit(_call_dao_member, arg): arg[0] for arg in call_args}
                    ordered_results = [None] * len(dao_member_prompts)
                    for future in concurrent.futures.as_completed(futures_map):
                        i_res, dao_answer = future.result()
                        ordered_results[i_res] = dao_answer

                # --- Confidence aggregation ---
                # Accumulate agent confidence scores and step suspicion scores across all 3 experts
                agent_confidence = {agent: 0.0 for agent in valid_agents}
                step_confidence   = {}  # step_str -> accumulated suspicion
                agent_reasons     = {}  # agent -> list of reason strings
                step_reasons      = {}  # step_str -> list of reason strings
                n_valid_responses  = 0

                for i, dao_answer in enumerate(ordered_results):
                    if not dao_answer:
                        print(f"Arbitrator {i+1} returned no response.")
                        continue
                    dao_member_responses.append(dao_answer)
                    print(f"--- Arbitrator {i+1} response ---\n{dao_answer}\n--- End Arbitrator {i+1} ---\n")
                    try:
                        json_match = re.search(r'\{.*\}', dao_answer, re.DOTALL)
                        if not json_match:
                            print(f"Arbitrator {i+1} JSON parse failed (no JSON found).")
                            continue
                        parsed = json.loads(json_match.group())
                        # Agent-level confidence — supports both {agent: float} and {agent: {"score": float, "reason": str}}
                        for agent, val in parsed.get("agents", {}).items():
                            if agent in agent_confidence:
                                try:
                                    score = float(val["score"]) if isinstance(val, dict) else float(val)
                                    agent_confidence[agent] += score
                                    if isinstance(val, dict) and val.get("reason"):
                                        agent_reasons.setdefault(agent, []).append(val["reason"])
                                except (TypeError, ValueError, KeyError):
                                    pass
                        # Step-level suspicion — supports both {step: float} and {step: {"score": float, "reason": str}}
                        for step_str, val in parsed.get("steps", {}).items():
                            s = str(step_str)
                            try:
                                score = float(val["score"]) if isinstance(val, dict) else float(val)
                                step_confidence[s] = step_confidence.get(s, 0.0) + score
                                if isinstance(val, dict) and val.get("reason"):
                                    step_reasons.setdefault(s, []).append(val["reason"])
                            except (TypeError, ValueError, KeyError):
                                pass
                        n_valid_responses += 1
                    except Exception as e:
                        print(f"Arbitrator {i+1} decode error: {e}")

                # Normalise to [0,1] by dividing by number of responding experts
                if n_valid_responses > 0:
                    agent_confidence = {k: v / n_valid_responses for k, v in agent_confidence.items()}
                    step_confidence  = {k: v / n_valid_responses for k, v in step_confidence.items()}

                print(f"DAO Arbitration closed.")
                for agent, conf in sorted(agent_confidence.items(), key=lambda x: -x[1]):
                    reasons_str = " | ".join(agent_reasons.get(agent, []))
                    print(f"  [Agent] {agent}: {conf:.2f}" + (f"  → {reasons_str}" if reasons_str else ""))
                for step_s, conf in sorted(step_confidence.items(), key=lambda x: -x[1]):
                    reasons_str = " | ".join(step_reasons.get(step_s, []))
                    print(f"  [Step]  {step_s}: {conf:.2f}" + (f"  → {reasons_str}" if reasons_str else ""))

                # Conviction threshold: winning agent must avg confidence >= 0.5
                CONVICTION_THRESHOLD = 0.5
                if agent_confidence and max(agent_confidence.values()) >= CONVICTION_THRESHOLD:
                    final_culprit = max(agent_confidence, key=agent_confidence.get)

                    # Determine culprit step: prefer DAO step-level attribution, fall back to forward search
                    if step_confidence:
                        best_step_str = max(step_confidence, key=step_confidence.get)
                        try:
                            culprit_step = int(best_step_str)
                        except ValueError:
                            culprit_step = idx
                    else:
                        # Fallback: first appearance of final_culprit in history up to flagged step
                        culprit_step = idx
                        for forward_idx in range(0, idx + 1):
                            if chat_history[forward_idx].get(index_agent) == final_culprit:
                                culprit_step = forward_idx
                                break

                    # Use DAO aggregated reason if available, else fall back to Accuser reason
                    dao_reason_parts = agent_reasons.get(final_culprit, [])
                    final_reason = " | ".join(dao_reason_parts) if dao_reason_parts else reason

                    print(f"\nPrediction for {json_file}: Error found.")
                    print(f"Agent Name: {final_culprit}")
                    print(f"Step Number: {culprit_step}")
                    print(f"Reason provided by DAO: {final_reason}")
                    error_found = True
                    break
                else:
                    print(f"Accusation rejected (max agent confidence {max(agent_confidence.values()) if agent_confidence else 0:.2f} < {CONVICTION_THRESHOLD}). Proceeding to next step.")
            elif "no" in prefix_area:
                print("No significant error detected by Accuser in this step.")
            else:
                print("Warning: Unexpected response format from Accuser. Continuing evaluation.")

        if not error_found:
            print(f"\nNo decisive errors found by step-by-step analysis with dao in file {json_file}")

        print("\n" + "="*50 + "\n")

# --- Backward Step-by-Step with DAO Arbitration Method ---

def step_by_step_dao_backward_api(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    Analyzes chat history by scanning backward from the last step to the first. 
    It asks an Accuser if the error at the current step was self-produced or inherited.
    If self-produced, 3 DAO agents vote to confirm.
    """
    print("\n--- Starting Backward Step-by-Step with DAO Arbitration Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")
        
        system_prompt_dict = data.get("system_prompt", {})
        valid_agents = list(system_prompt_dict.keys()) if isinstance(system_prompt_dict, dict) else []
        valid_agents_str = ", ".join(valid_agents) if valid_agents else "Not specified"
        
        roles_desc = ""
        if isinstance(system_prompt_dict, dict):
            for k, v in system_prompt_dict.items():
                roles_desc += f"--- Role: {k} ---\n{v}\n"

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        print(f"--- Analyzing File: {json_file} ---")

        full_conversation_history = ""
        for i_hist, entry_hist in enumerate(chat_history):
            a_name = entry_hist.get(index_agent, 'Unknown Agent')
            c = entry_hist.get('content', '')
            full_conversation_history += f"Step {i_hist} - {a_name}: {c}\n"

        error_found = False
        
        for idx in range(len(chat_history) - 1, -1, -1):
            entry = chat_history[idx]
            agent_name = entry.get(index_agent, 'Unknown Agent')

            prompt = (
                f"You are an AI assistant tasked with identifying the root cause of failure in a multi-agent conversation. "
                f"We know the final outcome was incorrect. We are now scanning backward from the end to find the origin of the error.\n"
                f"The problem being addressed is: {problem}\n"
                f"The Answer for the problem is: {ground_truth}\n"
                f"Here is the COMPLETE conversation history:\n{full_conversation_history}\n\n"
                f"Please analyze Step {idx} by '{agent_name}'.\n"
                "CRITICAL INSTRUCTION: You must perform Counterfactual Causal Attribution to distinguish the true root cause from downstream propagated symptoms. Follow these four reasoning stages internally:\n"
                "Stage A (Local Attribution): Check if there is any UPSTREAM step that can causally explain the anomaly here. [AXIOM]: You CANNOT blame the 'Problem', the user's prompt, or the 'Manager's instruction'. The origin MUST be attributed to an Agent's action. If Step 0 generates a flawed plan based on misinterpreting or blindly following a flawed problem description, Step 0 IS the origin (Self-produced). If this step received valid inputs but produced wrong outputs (or hallucinated without basis), it is Self-produced.\n"
                "Stage B (Planning-Control Attribution): If this step is within a repeated failure loop, distinguish Planner vs Executor responsibility. If the planner repeats identically flawed strategies despite failures, the planner is at fault. If the planner gives valid updated instructions but the executor fails to follow constraints, the executor is at fault.\n"
                "Stage C (Data-Flow Attribution): Trace how key data is produced. Does this step FABRICATE data (no upstream basis), MISINTERPRET upstream data, or MISUSE correct data? If so, the corruption starts here.\n"
                "Stage D (Final Screening): Deviation-Aware Filter - Is the deviation introduced here completely self-corrected in a later step? If it is successfully recovered later, it is NOT the root cause.\n\n"
                "Based on this counterfactual analysis, determine whether the error ORIGINATED at Step {idx} ('Self-produced', meaning it is the genuine unrecovered origin point) or if it was merely INHERITING/PROPAGATING an earlier error or is harmless ('Inherited/No error').\n"
                "Please respond ONLY in the following format:\n"
                "1. Self-produced/Inherited\n"
                "2. Reason: [Your concise explanation referencing Stage A/B/C/D logic]"
            )

            messages=[
                {"role": "system", "content": "You are a precise backward conversation evaluator looking for the origin of errors."},
                {"role": "user", "content": prompt},
            ]

            print(f"Backward Accuser checking Step {idx} by {agent_name}...")
            answer_accuser = _make_api_call(client, model, messages, max_tokens)

            if not answer_accuser:
                print("Failed to get evaluation for this step. Stopping analysis for this file.")
                error_found = True
                break
                
            print(f"Backward Accuser Evaluation: {answer_accuser}")

            answer_lower = answer_accuser.lower().strip()
            prefix_area = answer_lower.split('reason:', 1)[0] if 'reason:' in answer_lower else answer_lower
            
            if "self-produced" in prefix_area or "self produced" in prefix_area:
                try:
                    reason = answer_accuser.split('Reason:', 1)[-1].strip()
                except Exception:
                    reason = answer_accuser
                
                print(f"Backward Accuser flagged an error origin. Reason: {reason[:50]}...\nInitiating DAO Arbitration...")
                
                current_conversation_history = full_conversation_history # Backward dao sees entire history anyway
                base_context = (
                    f"You are a Strict, Independent Senior Expert reviewing a junior automated probe's flag in a multi-agent system.\n"
                    f"Problem: {problem}\nAnswer: {ground_truth}\n"
                    f"### THE COMPOSITION OF THE MAS ###\n{roles_desc}\n"
                    f"Conversation History:\n{current_conversation_history}\n"
                    f"--- PROBE HYPOTHESIS ---\n"
                    f"A basic monitoring probe has flagged Step {idx} (by '{agent_name}') as a suspect, with these notes:\n"
                    f"\"{reason}\"\n"
                )

                task_and_format_instruction = (
                    f"--- YOUR MISSION ---\n"
                    "Your primary duty is to STRESS-TEST and attempt to DEBUNK this probe's hypothesis. Probes often flag the wrong step (e.g., mistaking a downstream symptom for the root cause). "
                    "CRITICAL INSTRUCTION: Analyze the ENTIRE scenario based on these guidelines:\n"
                    "1. Play Devil's Advocate: Actively look for evidence that the true error originated in an earlier step or by a different agent.\n"
                    "2. Be Skeptical: Do NOT simply agree with the probe. You must override the probe if its reasoning is flawed or shallow.\n"
                    "3. Evaluate EVERY agent independently. If an agent made a root mistake, assign them a high guilt probability. If they just propagated a previous error, assign 0.\n"
                    "4. Do NOT blame 'Computer_terminal' (the execution environment).\n"
                    f"You must output a JSON object with EXACTLY TWO keys:\n"
                    f"  - 'agents': a dict mapping each valid Agent ({valid_agents_str}) to an object with:\n"
                    f"      'score' [0.0, 1.0]: guilt probability (0.0=innocent, 1.0=definitely the root cause)\n"
                    f"      'reason': one concise sentence explaining why. If you agree with the probe, explain why it survived your skepticism. If you disagree, state the real root cause.\n"
                    f"  - 'steps': a dict mapping suspect step numbers (as strings) to an object with:\n"
                    f"      'score' [0.0, 1.0]: suspicion score (1.0=this step is the root cause)\n"
                    f"      'reason': one concise sentence explaining what went wrong at this step.\n"
                    f"Example:\n"
                    f"{{\"agents\": {{\"Excel_Expert\": {{\"score\": 0.9, \"reason\": \"Used .astype(int) without NaN guard.\"}}, \"DataVerification_Expert\": {{\"score\": 0.1, \"reason\": \"No error detected.\"}}}}, \"steps\": {{\"0\": {{\"score\": 0.9, \"reason\": \"Root cause introduced here.\"}}, \"2\": {{\"score\": 0.1, \"reason\": \"Downstream effect only.\"}}}}}}\n"
                    "Respond ONLY with this JSON object and nothing else."
                )


                # --- Expert prompts identical to forward scan ---
                sys_prompt_1 = (
                    "You are a Factual Attribution Expert specializing in identifying failures caused by "
                    "incorrect information retrieval, hallucinated content, unverified external data, or "
                    "inappropriate tool selection. Your analytical lens focuses exclusively on WHAT data was "
                    "retrieved or used, WHERE it came from, and WHETHER it was verified against ground truth "
                    "before being passed downstream. Patterns you prioritize: hallucination, reliance on "
                    "unverified sources, tool misuse, failure to verify information, incorrect data extraction."
                )
                dao_member_prompt_1 = (
                    base_context +
                    "\n[YOUR EXPERT DIMENSION] You are the FACTUAL / RETRIEVAL Expert. "
                    "Focus your analysis on whether any agent: (1) hallucinated or fabricated data, "
                    "(2) relied on unverified or incorrect external sources, (3) selected the wrong tool "
                    "for information retrieval, (4) failed to verify retrieved information against ground truth, "
                    "or (5) extracted incorrect or temporally mismatched data. "
                    "Only assign guilt based on evidence of factual or retrieval-dimension failures.\n\n" +
                    task_and_format_instruction
                )

                sys_prompt_2 = (
                    "You are a Logic and Reasoning Attribution Expert specializing in identifying failures "
                    "caused by flawed internal reasoning, incorrect algorithm or formula implementation, "
                    "premature conclusions, task decomposition errors, or misaligned problem understanding. "
                    "Your analytical lens focuses on the agent's INTERNAL reasoning chain and decision logic, "
                    "NOT on its data sources. Patterns you prioritize: incorrect algorithm, premature conclusion, "
                    "logical inconsistency, task misidentification, flawed solution strategy, incorrect calculation."
                )
                dao_member_prompt_2 = (
                    base_context +
                    "\n[YOUR EXPERT DIMENSION] You are the LOGIC / REASONING Expert. "
                    "Focus your analysis on whether any agent: (1) applied an incorrect algorithm or formula, "
                    "(2) reached a premature or unsupported conclusion, (3) decomposed the task incorrectly, "
                    "(4) exhibited logical inconsistency or contradictory reasoning, or (5) misidentified the "
                    "problem domain or objective. Do NOT blame data quality; focus solely on reasoning-layer failures.\n\n" +
                    task_and_format_instruction
                )

                sys_prompt_3 = (
                    "You are an Agent Execution and Coordination Attribution Expert specializing in identifying "
                    "failures caused by inter-agent coordination breakdowns, role boundary violations, code "
                    "execution errors by sub-agents, or cross-agent verification failures. Your analytical lens "
                    "focuses on HOW agents interact, delegate, and verify each other's outputs. Patterns you "
                    "prioritize: agent role violation, cross-agent verification failure, agent provided inaccurate "
                    "foundational data to downstream agents, agent exceeded step/resource limits, code execution "
                    "failure by a dependent agent, orchestrator over-delegation or under-specification."
                )
                dao_member_prompt_3 = (
                    base_context +
                    "\n[YOUR EXPERT DIMENSION] You are the AGENT EXECUTION / COORDINATION Expert. "
                    "Focus your analysis on whether any agent: (1) violated its role boundary or responsibilities, "
                    "(2) failed to verify or correctly pass outputs to downstream agents, (3) executed faulty or "
                    "non-functional code, (4) caused a cascading failure by providing incorrect foundational data "
                    "to other agents, or (5) exceeded computational/step limits due to repeated erroneous actions. "
                    "Focus on agent-level behavioral and coordination failures, not on data quality or reasoning logic.\n\n" +
                    task_and_format_instruction
                )

                dao_member_prompts = [dao_member_prompt_1, dao_member_prompt_2, dao_member_prompt_3]
                sys_prompts = [sys_prompt_1, sys_prompt_2, sys_prompt_3]
                
                # --- Concurrent DAO calls ---
                dao_member_responses = []
                def _call_dao_member(args):
                    i, sys_p, user_p = args
                    dao_messages = [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}]
                    if i == 0:
                        result = _make_api_DAO_1_call(client, model, dao_messages, max_tokens)
                    elif i == 1:
                        result = _make_api_DAO_2_call(client, model, dao_messages, max_tokens)
                    else:
                        result = _make_api_DAO_3_call(client, model, dao_messages, max_tokens)
                    return i, result

                call_args = [(i, sys_prompts[i], dao_member_prompts[i]) for i in range(len(dao_member_prompts))]
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures_map = {executor.submit(_call_dao_member, arg): arg[0] for arg in call_args}
                    ordered_results = [None] * len(dao_member_prompts)
                    for future in concurrent.futures.as_completed(futures_map):
                        i_res, dao_answer = future.result()
                        ordered_results[i_res] = dao_answer

                # --- Confidence aggregation ---
                agent_confidence = {agent: 0.0 for agent in valid_agents}
                step_confidence   = {}
                agent_reasons     = {}
                step_reasons      = {}
                n_valid_responses  = 0

                for i, dao_answer in enumerate(ordered_results):
                    if not dao_answer:
                        print(f"Arbitrator {i+1} returned no response.")
                        continue
                    dao_member_responses.append(dao_answer)
                    print(f"--- Arbitrator {i+1} response ---\n{dao_answer}\n--- End Arbitrator {i+1} ---\n")
                    try:
                        json_match = re.search(r'\{.*\}', dao_answer, re.DOTALL)
                        if not json_match:
                            continue
                        parsed = json.loads(json_match.group())
                        for agent, val in parsed.get("agents", {}).items():
                            if agent in agent_confidence:
                                try:
                                    score = float(val["score"]) if isinstance(val, dict) else float(val)
                                    agent_confidence[agent] += score
                                    if isinstance(val, dict) and val.get("reason"):
                                        agent_reasons.setdefault(agent, []).append(val["reason"])
                                except (TypeError, ValueError, KeyError):
                                    pass
                        for step_str, val in parsed.get("steps", {}).items():
                            s = str(step_str)
                            try:
                                score = float(val["score"]) if isinstance(val, dict) else float(val)
                                step_confidence[s] = step_confidence.get(s, 0.0) + score
                                if isinstance(val, dict) and val.get("reason"):
                                    step_reasons.setdefault(s, []).append(val["reason"])
                            except (TypeError, ValueError, KeyError):
                                pass
                        n_valid_responses += 1
                    except Exception as e:
                        print(f"Arbitrator {i+1} decode error: {e}")

                if n_valid_responses > 0:
                    agent_confidence = {k: v / n_valid_responses for k, v in agent_confidence.items()}
                    step_confidence  = {k: v / n_valid_responses for k, v in step_confidence.items()}

                print(f"DAO Arbitration closed.")
                for agent, conf in sorted(agent_confidence.items(), key=lambda x: -x[1]):
                    reasons_str = " | ".join(agent_reasons.get(agent, []))
                    print(f"  [Agent] {agent}: {conf:.2f}" + (f"  → {reasons_str}" if reasons_str else ""))
                for step_s, conf in sorted(step_confidence.items(), key=lambda x: -x[1]):
                    reasons_str = " | ".join(step_reasons.get(step_s, []))
                    print(f"  [Step]  {step_s}: {conf:.2f}" + (f"  → {reasons_str}" if reasons_str else ""))

                # Conviction threshold
                CONVICTION_THRESHOLD = 0.5
                if agent_confidence and max(agent_confidence.values()) >= CONVICTION_THRESHOLD:
                    final_culprit = max(agent_confidence, key=agent_confidence.get)

                    if step_confidence:
                        best_step_str = max(step_confidence, key=step_confidence.get)
                        try:
                            culprit_step = int(best_step_str)
                        except ValueError:
                            culprit_step = idx
                    else:
                        culprit_step = idx
                        for forward_idx in range(0, idx + 1):
                            if chat_history[forward_idx].get(index_agent) == final_culprit:
                                culprit_step = forward_idx
                                break

                    dao_reason_parts = agent_reasons.get(final_culprit, [])
                    final_reason = " | ".join(dao_reason_parts) if dao_reason_parts else reason

                    print(f"\nPrediction for {json_file}: Error found.")
                    print(f"Agent Name: {final_culprit}")
                    print(f"Step Number: {culprit_step}")
                    print(f"Reason provided by DAO: {final_reason}")
                    error_found = True
                    break
                else:
                    print(f"Accusation rejected (max agent conf < {CONVICTION_THRESHOLD}). Continuing searching backward.")
            else:
                print("No error origin detected by Accuser (Inherited/No error). Searching previous step.")

        if not error_found:
            print(f"\nNo decisive error origins found by backward analysis in file {json_file}")

        print("\n" + "="*50 + "\n")






# --- Forward Step-by-Step with ECHo Logic ---

def step_by_step_dao_echo_api(client, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    FORWARD scan step-by-step applying the ECHo hierarchical context logic.
    """
    import os, re, concurrent.futures
    from tqdm import tqdm
    from Lib.api_utils import _load_json_data, _get_sorted_json_files, _make_api_call, _make_api_DAO_1_call, _make_api_DAO_2_call, _make_api_DAO_3_call
    
    print("\n--- Starting Forward Step-by-Step with ECHo Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"
    CONVICTION_THRESHOLD = 0.5 

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data: continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")
        
        system_prompt_dict = data.get("system_prompt", {})
        valid_agents = list(system_prompt_dict.keys()) if isinstance(system_prompt_dict, dict) else []
        valid_agents_str = ", ".join(valid_agents) if valid_agents else "Not specified"
        
        roles_desc = ""
        if isinstance(system_prompt_dict, dict):
            for k, v in system_prompt_dict.items():
                roles_desc += f"--- Role: {k} ---\n{v}\n"

        if not chat_history:
            continue

        print(f"--- Analyzing File: {json_file} ---")
        print(f"--- Comprehensively Extracting Contextual Log (Sequential) for {json_file} ---")
        step_summaries = [""] * len(chat_history)
        
        for s_idx in range(len(chat_history)):
            s_agent = chat_history[s_idx].get(index_agent, 'Unknown')
            s_content = chat_history[s_idx].get('content', '')
            
            if s_agent.lower() == "computer_terminal":
                step_summaries[s_idx] = "(Execution Feedback bundled in previous step)"
                continue
                
            combined_content = s_content
            if s_idx + 1 < len(chat_history):
                next_agent = chat_history[s_idx + 1].get(index_agent, 'Unknown')
                if next_agent.lower() == "computer_terminal":
                    next_content = chat_history[s_idx + 1].get('content', '')
                    combined_content += f"\n\n[Execution Result from Sandbox]:\n{next_content}"

            past_summaries_context = ""
            for i in range(max(0, s_idx - 2), s_idx):
                if step_summaries[i] and "bundled" not in step_summaries[i]:
                    past_summaries_context += f"[Step {i}] {chat_history[i].get(index_agent, 'Unknown')}: {step_summaries[i]}\n"

            summary_sys = (
                "You are a structured data extractor. You must analyze the agent's interaction and output ONLY a valid JSON object. "
                "Do NOT include markdown formatting like ```json. The JSON object MUST have EXACTLY these three keys:\n"
                "- 'Upstream_Instruction_or_Context': Briefly describe the instruction, goal, or context passed to this agent (max 1 sentence).\n"
                "- 'Node_Action': Briefly describe what this agent decided or code it wrote (max 1 sentence).\n"
                "- 'Result_and_Feedback': Briefly describe the final output, resolution, or sandbox execution result (max 1 sentence).\n\n"
                "### FEW-SHOT EXAMPLE ###\n"
                "Input:\n"
                "Agent: Excel_Expert\n"
                "Content: Write python code to determine how many clients will receive sunset awnings (even address).\n"
                "```python\neven_address = data[data['Street Number'] % 2 == 0]\nprint(len(even_address))\n```\n"
                "[Execution Result from Sandbox]:\n"
                "exitcode: 0\nCode output: 4\n\n"
                "Output:\n"
                "{\n"
                "  \"Upstream_Instruction_or_Context\": \"Received the task to count the number of clients with even street numbers (facing west) from the Excel file.\",\n"
                "  \"Node_Action\": \"Wrote a Pandas script using a modulo 2 operation on the street number field to filter and print the count of even addresses.\",\n"
                "  \"Result_and_Feedback\": \"The terminal sandbox successfully executed the code, returning a final count of 4.\"\n"
                "}"
            )
            
            summary_user = ""
            if past_summaries_context:
                summary_user += f"--- Previous Steps Context ---\n{past_summaries_context}\n\n"
            summary_user += f"--- Current Step to Extract ---\nAgent: {s_agent}\nContent: {combined_content}"
            
            msg = [{"role": "system", "content": summary_sys}, {"role": "user", "content": summary_user}]
            res = _make_api_DAO_3_call(client, model, msg, 512)
            step_summaries[s_idx] = res if res else "(Failed)"

        error_found = False

        for idx, entry in enumerate(chat_history):
            agent_name = entry.get(index_agent, 'Unknown Agent')
            
            # 如果是纯物理反馈环境，跳过独立审查阶段
            if agent_name.lower() == "computer_terminal":
                continue
                
            current_full_text = entry.get('content', '')
            # L1: 将紧随其后的 Sandbox 执行结果直接捆绑进当前步骤的 L1 视角里
            if idx + 1 < len(chat_history):
                next_agent = chat_history[idx + 1].get(index_agent, 'Unknown')
                if next_agent.lower() == "computer_terminal":
                    current_full_text += f"\n\n[Sandbox Execution Result (Step {idx+1})]:\n{chat_history[idx+1].get('content', '')}"

            # L3 Distant Past (结构化 JSON 摘要池：0 到 idx-1)
            dist_past = ""
            for i in range(0, idx):
                if chat_history[i].get(index_agent, '').lower() == "computer_terminal": continue
                dist_past += f"[Step {i}] {chat_history[i].get(index_agent, 'Unknown')}: (JSON SUMMARY) {step_summaries[i]}\n"
            
            # Forward High-Resolution Window (包含当前涉案人员 idx，动态顺延捕获恰好 3 个 Agent 实体)
            valid_agent_count = 0
            window_end = idx
            while valid_agent_count < 3 and window_end < len(chat_history):
                # 遇到真实 Agent 计数+1，遇到沙盒不占用视窗名额
                if chat_history[window_end].get(index_agent, '').lower() != "computer_terminal":
                    valid_agent_count += 1
                window_end += 1
                
            forward_window = ""
            for i in range(idx, window_end):
                if chat_history[i].get(index_agent, '').lower() == "computer_terminal": continue
                fw_text = chat_history[i].get('content', '')
                # 这里就是保障 “如果紧接着的是沙盒输出，一定捆绑带上” 的核心逻辑
                if i + 1 < len(chat_history) and chat_history[i+1].get(index_agent, '').lower() == "computer_terminal":
                    fw_text += f"\n[Sandbox Feedback]: {chat_history[i+1].get('content', '')}"
                forward_window += f"[Step {i}] {chat_history[i].get(index_agent, 'Unknown')}: (FULL TEXT)\n{fw_text}\n\n"
            
            # L4 Distant Future (遥远的未来走势：从 window_end 到结尾，转回 JSON 摘要)
            future_down = ""
            for i in range(window_end, len(chat_history)):
                if chat_history[i].get(index_agent, '').lower() == "computer_terminal": continue
                future_down += f"[Step {i}] {chat_history[i].get(index_agent, 'Unknown')}: (JSON SUMMARY) {step_summaries[i]}\n"
            
            hierarchical_context = ""
            if dist_past: hierarchical_context += f"--- SCANNED PAST (JSON Summaries) ---\n{dist_past}\n"
            hierarchical_context += f"--- HIGH-RESOLUTION WINDOW (Steps {idx} to {min(len(chat_history)-1, idx+2)}) ---\n{forward_window}\n"
            if future_down: hierarchical_context += f"--- DISTANT FUTURE (JSON Summaries) ---\n{future_down}\n"

            prompt = (
                "You are an AI Accuser evaluating a multi-agent conversation to identify the ROOT CAUSE of failure.\n"
                "Note:  'Computer_terminal' is a sandbox environment output.\n"
                f"Problem: {problem}\nGround Truth Answer: {ground_truth}\n\n"
                f"Hierarchical view:\n{hierarchical_context}\n"
                f"--- COUNTERFACTUAL BACKTRACKING TASK ---\n"
                f"Evaluate Step {idx} by '{agent_name}'. You MUST employ counterfactual reasoning before deciding:\n"
                f"- INHERITANCE CHECK (Upstream): Did '{agent_name}' inherit corrupt data, false assumptions, or flawed instructions from the 'SCANNED PAST'? If they merely executed a flawed upstream command perfectly, they are INNOCENT (the error is further upstream).\n"
                f"- COUNTERFACTUAL CHECK (Downstream): If we magically corrected Step {idx}'s action, would the disaster in the 'HIGH-RESOLUTION WINDOW' have been prevented? If yes, and the mistake was NOT inherited, they are GUILTY.\n\n"
                "Respond ONLY in the exact format:\n"
                "1. Root Error: [Yes/No]\n"
                "2. Reason: [Your rigorous counterfactual deduction]"
            )

            messages=[{"role": "system", "content": "You are a precise step-by-step evaluator."}, {"role": "user", "content": prompt}]
            answer = _make_api_call(client, model, messages, max_tokens)
            
            if not answer: continue
            print(f"Probe Evaluation for Step {idx}: {answer}")
            
            ans_lower = answer.lower()
            if "yes" in ans_lower[:20]:
                reason = answer.split('Reason:', 1)[-1].strip() if 'Reason:' in answer else answer
                print(f"Probe flagged error at Step {idx}. Invoking DAO Arbitration...")
                
                base_context = (
                    f"General Task Origin: {problem}\nGround Truth Expected Outcome: {ground_truth}\n"
                    f"MAS Agent Roles Info:\n{roles_desc}\n"
                    "Note: 'Computer_terminal' is a sandbox environment output.\n\n"
                    "You have access to hierarchical context showing:\n"
                    "Immediate agents: Full details\n"
                    "Nearby agents: Key decisions\n"
                    "Distant agents: Brief summaries\n\n"
                    "The agents are numbered sequentially (Step 0, Step 1, etc.) corresponding to their turn index.\n\n"
                    "--- CASE DOSSIER (Hierarchical Log) ---\n"
                    f"{hierarchical_context}\n"
                    "--- FLAG FOR REVIEW ---\n"
                    f"A frontline scanner has flagged Step {idx} (Agent: '{agent_name}') for review because an error manifested nearby.\n"
                    "Do not blindly trust this flag. You must objectively evaluate the ENTIRE sequence above to find the TRUE originator of the error.\n"
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
                    '    "type": "single_agent",\n'
                    '    "attribution": ["agent_name"],\n'
                    '    "mistake_step": 1,\n'
                    '    "confidence": 1.0,\n'
                    '    "reasoning": "Detailed explanation of your primary conclusion including which step the error occurred"\n'
                    '  }\n'
                    "}\n"
                    "Be thorough, objective, and consider all possibilities including that no single agent may be clearly at fault.\n"
                    "Pay special attention to identifying the specific step/turn where the error truly originated. "
                    "RULE FOR STEP SELECTION: If an agent's failure spans multiple steps, select the EXACT step where the agent first committed an ACTIVE error (e.g., hallucinating data, writing flawed code, or making a false assumption), NOT a step where a tool simply returned empty results."
                )
                
                
                
                sys_base = (
                    "You are an Objective Analysis Agent conducting an impartial investigation to determine error attribution in a multi-agent conversation.\n"
                    "Your task:\n"
                    "1. Analyze ALL agents in the conversation objectively (not just one specific agent)\n"
                    "2. Determine which agent(s) most likely caused the final wrong answer\n"
                    "3. Determine which step/turn in the conversation the mistake occurred\n"
                    "4. Provide confidence scores and reasoning for your conclusions"
                )
                sys_1 = sys_base + "\n\nANALYST SPECIALIZATION: Factual Attribution Expert. Focus strictly on fact-checking, constraint extraction, and data alignment."
                sys_2 = sys_base + "\n\nANALYST SPECIALIZATION: Logic Attribution Expert. Focus strictly on logical deduction, algorithmic correctness, and task logic."
                sys_3 = sys_base + "\n\nANALYST SPECIALIZATION: Agent Execution Expert. Focus strictly on agent tool usage, workflow formatting, and sandbox environmental interaction."
                
                def _call_dao(args):
                    i, sys_p, usr_p = args
                    msg = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
                    if i==0: return _make_api_DAO_1_call(client, model, msg, max_tokens)
                    if i==1: return _make_api_DAO_2_call(client, model, msg, max_tokens)
                    return _make_api_DAO_3_call(client, model, msg, max_tokens)
                
                import json
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futs = {executor.submit(_call_dao, (i, s, u)): i for i, (s,u) in enumerate([(sys_1, dao_1), (sys_2, dao_2), (sys_3, dao_3)])}
                    results = [None]*3
                    for f in concurrent.futures.as_completed(futs):
                        results[futs[f]] = f.result()
                
                agent_scores = {a: 0.0 for a in valid_agents}
                max_single_agent_scores = {a: 0.0 for a in valid_agents}
                step_scores = {}
                valid_resp = 0
                for i, r in enumerate(results):
                    if r:
                        print(f"--- Arbitrator {i+1} response ---\n{r}\n--- End Arbitrator {i+1} ---\n")
                        try:
                            jstr = re.search(r'\{.*\}', r, re.DOTALL).group(0)
                            jdata = json.loads(jstr)
                            if 'primary_conclusion' in jdata:
                                pc = jdata['primary_conclusion']
                                conf = float(pc.get('confidence', 1.0))
                                
                                attr = pc.get('attribution', [])
                                if isinstance(attr, str): attr = [attr]
                                if isinstance(attr, list):
                                    for a in attr:
                                        if a not in agent_scores: agent_scores[a] = 0.0
                                        if a not in max_single_agent_scores: max_single_agent_scores[a] = 0.0
                                        agent_scores[a] += conf
                                        if conf > max_single_agent_scores[a]:
                                            max_single_agent_scores[a] = conf
                                            
                                step_val = str(pc.get('mistake_step', idx))
                                step_scores[step_val] = step_scores.get(step_val, 0.0) + conf
                                
                            valid_resp += 1
                        except: pass
                
                if valid_resp > 0:
                    for a in agent_scores: agent_scores[a] /= valid_resp
                    for s_str in step_scores: step_scores[s_str] /= valid_resp
                    
                    avg_val = max(agent_scores.values()) if agent_scores else 0
                    max_vote = max(max_single_agent_scores.values()) if max_single_agent_scores else 0
                    
                    if avg_val >= CONVICTION_THRESHOLD or max_vote > 0.85:
                        final_culprit = max(agent_scores, key=agent_scores.get) if agent_scores else agent_name
                        if step_scores:
                            valid_steps = [
                                str(s) for s, data in enumerate(chat_history) 
                                if data.get(index_agent) == final_culprit
                            ]
                            f_scores = {k: v for k, v in step_scores.items() if k in valid_steps}
                            if f_scores:
                                max_score = max(f_scores.values())
                                candidates = [k for k, v in f_scores.items() if v == max_score]
                                best_step_str = min(candidates, key=lambda x: int(x))
                            else:
                                best_step_str = max(step_scores, key=step_scores.get)
                            try:
                                culprit_step = int(best_step_str)
                            except ValueError:
                                culprit_step = idx
                        else:
                            culprit_step = idx
                            
                        print(f"Error conclusively found at step {culprit_step}. (Avg Score: {avg_val:.2f}, Max Vote: {max_vote:.2f}) Halting forward scan.")
                        print(f"\nPrediction for {json_file}: Error found.")
                        print(f"Agent Name: {final_culprit}")
                        print(f"Step Number: {culprit_step}")
                        print(f"Reason provided by DAO: Convicted by ECHo Tribunal. (Threshold Matched)")
                        error_found = True
                        break
                    else:
                        print(f"DAO absolved step {idx} (Avg Score: {avg_val:.2f}, Max Vote: {max_vote:.2f}). Continuing scan...")
                else:
                    print("All DAO calls failed. Assuming error.")
                    print(f"\nPrediction for {json_file}: Error found.")
                    print(f"Agent Name: {agent_name}")
                    print(f"Step Number: {idx}")
                    error_found = True
                    break

        if not error_found:
            agent_0_name = chat_history[0].get(index_agent, 'Unknown Agent')
            print(f"Probe missed the error entirely. Activating Step 0 Planning Failsafe.")
            print(f"\nPrediction for {json_file}: Error found.")
            print(f"Agent Name: {agent_0_name}")
            print(f"Step Number: 0")
            print(f"Reason provided by DAO: (Failsafe) Initial planning failure or hallucination bypassing downstream probes.")

def step_by_step_dao_echo_api_noGT(client, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int, sys_prompts=None):
    """
    No-Ground-Truth variant of step_by_step_dao_echo_api.

    Identical pipeline (Probe → DAO → Soft Voting) but all references to
    `ground_truth` are removed from every prompt.  The model must detect
    errors purely from internal conversation consistency, role-alignment
    violations, and observable output anomalies.

    Additions vs. GT version:
    - Per-phase token consumption tracking (Phase1 / Probe / DAO).
    - Phase 1 summary cache: saves to outputs/summaries_cache/<dataset>.json
      and reuses on subsequent runs to avoid redundant LLM calls.
    """
    import os, re, json as _json, concurrent.futures
    from tqdm import tqdm
    from Lib.api_utils import _load_json_data, _get_sorted_json_files

    print("\n--- [noGT] Starting Forward Step-by-Step ECHo Analysis (No Ground Truth) ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"
    CONVICTION_THRESHOLD = 0.5

    # ── Token tracking ────────────────────────────────────────────────────────
    token_stats = {"phase1": 0, "probe": 0, "dao": 0}

    def _tracked_call(model_alias: str, messages: list, max_tokens_val: int, phase: str) -> str | None:
        """
        API call with token usage tracking.
        Uses stream_options={"include_usage": True} to capture exact usage
        from the final streaming chunk (OpenAI-compatible API feature).
        ds-v3.2 automatically enables thinking mode via extra_body.
        """
        full_model = API_MODEL_MAP.get(model_alias, model_alias)
        for attempt in range(3):
            try:
                call_kwargs = dict(
                    model=full_model,
                    messages=messages,
                    max_tokens=max_tokens_val,
                    stream=True,
                    temperature=0.5,
                    stream_options={"include_usage": True},
                )
                if model_alias == "ds-v3.2":
                    call_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                response = client.chat.completions.create(**call_kwargs)
                full_content = ""
                usage_tokens = 0
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta:
                        delta_content = getattr(chunk.choices[0].delta, "content", None)
                        if delta_content:
                            full_content += delta_content
                    # Final chunk carries usage when stream_options include_usage=True
                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        usage_tokens = getattr(chunk.usage, "total_tokens", 0)
                token_stats[phase] += usage_tokens
                return full_content.strip() or None
            except Exception as e:
                print(f"[noGT] {phase} API call attempt {attempt+1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(2)
        return None

    # ── Phase 1 summary cache ─────────────────────────────────────────────────
    dataset_name = os.path.basename(os.path.normpath(directory_path))
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "summaries_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_name}.json")

    summaries_cache: dict = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as _f:
                summaries_cache = _json.load(_f)
            print(f"[Phase1 Cache] Loaded {len(summaries_cache)} cached files from {cache_path}\n")
        except Exception:
            summaries_cache = {}

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data: continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        # ground_truth intentionally not used

        system_prompt_dict = data.get("system_prompt", {})
        valid_agents = list(system_prompt_dict.keys()) if isinstance(system_prompt_dict, dict) else []

        roles_desc = ""
        if isinstance(system_prompt_dict, dict):
            for k, v in system_prompt_dict.items():
                roles_desc += f"--- Role: {k} ---\n{v}\n"

        if not chat_history:
            continue

        print(f"--- Analyzing File: {json_file} ---")

        # ── Phase 1: Structured extraction with cache ─────────────────────────
        if json_file in summaries_cache:
            step_summaries = summaries_cache[json_file]
            # Pad/trim to match current history length (defensive)
            if len(step_summaries) != len(chat_history):
                step_summaries = (step_summaries + [""] * len(chat_history))[:len(chat_history)]
            print(f"[Phase1 Cache] HIT — skipping LLM extraction for {json_file}")
        else:
            print(f"--- [Phase1] Extracting Contextual Log for {json_file} ---")
            step_summaries = [""] * len(chat_history)

            if is_handcrafted:
                summary_sys = (
                    "You are a structured data extractor for multi-agent web-research task logs. "
                    "Output ONLY a valid JSON object with EXACTLY these three keys:\n"
                    "- 'Upstream_Instruction_or_Context': The instruction, goal, or context this agent received (max 1 sentence).\n"
                    "- 'Node_Action': What this agent decided, planned, delegated, navigated, or retrieved (max 1 sentence).\n"
                    "- 'Result_and_Feedback': The direct outcome — page content obtained, sub-task result returned, or conclusion reached (max 1 sentence).\n"
                    "Do NOT include markdown formatting like ```json.\n\n"
                    "### FEW-SHOT EXAMPLES ###\n\n"
                    "-- Example 1: human --\n"
                    "Input:\nAgent: human\n"
                    "Content: Where can I take martial arts classes within a five-minute walk from the New York Stock Exchange after work (7-9 pm)?\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"User initiated the task requesting martial arts classes near NYSE available 7-9 pm.\","
                    " \"Node_Action\": \"Submitted the original question to the multi-agent system.\","
                    " \"Result_and_Feedback\": \"Task handed off to Orchestrator for planning and delegation.\"}\n\n"
                    "-- Example 2: Orchestrator (thought) — initial planning --\n"
                    "Input:\nAgent: Orchestrator (thought)\n"
                    "Content: Initial plan: We need to find martial arts schools within 5-minute walk from NYSE with 7-9 pm classes. "
                    "Team: WebSurfer for web navigation. Step 1: search for nearby schools. Step 2: verify addresses and schedules.\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Received user task to find nearby martial arts classes after work hours.\","
                    " \"Node_Action\": \"Formulated a two-step plan: first search for schools near NYSE, then verify addresses and evening schedules.\","
                    " \"Result_and_Feedback\": \"Plan established; delegating web search to WebSurfer as next action.\"}\n\n"
                    "-- Example 3: Orchestrator (thought) — ledger update --\n"
                    "Input:\nAgent: Orchestrator (thought)\n"
                    "Content: Updated Ledger: {\"is_request_satisfied\": {\"reason\": \"No confirmed martial arts school with evening hours found yet.\", \"answer\": false}, "
                    "\"is_in_loop\": {\"reason\": \"WebSurfer navigated to an unrelated page last step, no forward progress.\", \"answer\": true}}\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Reviewing WebSurfer's last action which landed on an irrelevant product page.\","
                    " \"Node_Action\": \"Updated task ledger: marked request as unsatisfied and loop detected due to repeated off-target navigation.\","
                    " \"Result_and_Feedback\": \"Determined the agent is stuck in a loop; will issue corrective delegation to redirect WebSurfer.\"}\n\n"
                    "-- Example 4: Orchestrator (-> WebSurfer) — delegation --\n"
                    "Input:\nAgent: Orchestrator (-> WebSurfer)\n"
                    "Content: Please visit the Astronomy Picture of the Day Archive 2015 page on nasa.gov and navigate to the first week of August 2015 to find the specific image.\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Ledger shows the NASA APOD image for August 2015 has not been identified yet.\","
                    " \"Node_Action\": \"Delegated a targeted navigation instruction to WebSurfer to visit the NASA APOD 2015 archive page.\","
                    " \"Result_and_Feedback\": \"Instruction sent; awaiting WebSurfer's page content and image identification.\"}\n\n"
                    "-- Example 5: WebSurfer — successful navigation --\n"
                    "Input:\nAgent: WebSurfer\n"
                    "Content: I clicked 'Thriller (album) - Wikipedia'. "
                    "Here is a screenshot of [Thriller (album) - Wikipedia](https://en.wikipedia.org/wiki/Thriller_(album)). "
                    "The page shows singles released: The Girl Is Mine (1st), Billie Jean (2nd), Beat It (3rd), Wanna Be Startin' Somethin' (4th), Human Nature (5th).\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Received instruction to confirm the fifth single from Michael Jackson's Thriller album.\","
                    " \"Node_Action\": \"Navigated to the Thriller Wikipedia page and located the singles release order in the article.\","
                    " \"Result_and_Feedback\": \"Successfully retrieved: the fifth single from Thriller is 'Human Nature'.\"}\n\n"
                    "-- Example 6: WebSurfer — wrong navigation / irrelevant page --\n"
                    "Input:\nAgent: WebSurfer\n"
                    "Content: I clicked 'NY Jidokwan Taekwondo'. "
                    "Here is a screenshot of [薄膜/透明體測厚專家 - VK系列雷射顯微鏡 | KEYENCE 台灣基恩斯](https://www.keyence.com.tw/...). "
                    "The viewport shows a microscopy equipment product page unrelated to martial arts.\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Received instruction to click on martial arts school links and note their addresses and schedules.\","
                    " \"Node_Action\": \"Clicked 'NY Jidokwan Taekwondo' link but was redirected to a KEYENCE microscopy product page.\","
                    " \"Result_and_Feedback\": \"Navigation failed: landed on an irrelevant commercial page with no martial arts school information.\"}\n\n"
                    "-- Example 7: FileSurfer --\n"
                    "Input:\nAgent: FileSurfer\n"
                    "Content: Opening the uploaded file about monday.com IPO documents. "
                    "File contains a table of C-suite executives as of June 2021 IPO date: Roy Mann (Co-CEO), Eran Zinman (Co-CEO), Eliran Glazer (CFO).\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Received instruction to read the monday.com IPO document and extract C-suite executives at the time of IPO.\","
                    " \"Node_Action\": \"Opened and parsed the IPO document file, located the executive leadership table.\","
                    " \"Result_and_Feedback\": \"Extracted three C-suite members as of IPO (June 2021): Co-CEOs Roy Mann and Eran Zinman, CFO Eliran Glazer.\"}"
                )
            else:
                summary_sys = (
                    "You are a structured data extractor. You must analyze the agent's interaction and output ONLY a valid JSON object. "
                    "Do NOT include markdown formatting like ```json. The JSON object MUST have EXACTLY these three keys:\n"
                    "- 'Upstream_Instruction_or_Context': Briefly describe the instruction, goal, or context passed to this agent (max 1 sentence).\n"
                    "- 'Node_Action': Briefly describe what this agent decided or code it wrote (max 1 sentence).\n"
                    "- 'Result_and_Feedback': Briefly describe the final output, resolution, or sandbox execution result (max 1 sentence).\n\n"
                    "### FEW-SHOT EXAMPLES ###\n\n"
                    "-- Example 1: Domain Expert — code execution success --\n"
                    "Input:\nAgent: Excel_Expert\n"
                    "Content: Write python code to determine how many clients will receive sunset awnings (even address).\n"
                    "```python\neven_address = data[data['Street Number'] % 2 == 0]\nprint(len(even_address))\n```\n"
                    "[Execution Result from Sandbox]:\nexitcode: 0\nCode output: 4\n\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Received the task to count clients with even street numbers from the Excel file.\","
                    " \"Node_Action\": \"Wrote a Pandas script using modulo 2 on the Street Number field to filter and count even addresses.\","
                    " \"Result_and_Feedback\": \"Sandbox executed successfully (exitcode=0), returning a count of 4.\"}\n\n"
                    "-- Example 2: Domain Expert — code execution failure (exitcode=1) --\n"
                    "Input:\nAgent: Boggle_Board_Expert\n"
                    "Content: Here is the DFS implementation to find the longest word on the Boggle board:\n"
                    "```python\nlongest_word = find_longest_word(boggle_board, dictionary)\nprint(longest_word)\n```\n"
                    "[Execution Result from Sandbox]:\nexitcode: 1\n"
                    "Code output: NameError: name 'dictionary' is not defined\n\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Received the task to implement DFS to find the longest valid word on the Boggle board using a pre-loaded dictionary.\","
                    " \"Node_Action\": \"Wrote a DFS-based word search script referencing the variable 'dictionary' without loading it in the same code block.\","
                    " \"Result_and_Feedback\": \"Sandbox execution failed (exitcode=1) with NameError: 'dictionary' is not defined.\"}\n\n"
                    "-- Example 3: Domain Expert — wrong factual assumption, no code error --\n"
                    "Input:\nAgent: Validation_Expert\n"
                    "Content: The following are the population figures retrieved from data.census.gov:\n"
                    "- Seattle, Washington (2020 Census): 737,015\n"
                    "- Colville, Washington (2020 Census): 4,965\n"
                    "```python\npopulation_seattle = 737015\npopulation_colville = 4965\nprint(population_seattle - population_colville)\n```\n"
                    "[Execution Result from Sandbox]:\nexitcode: 0\nCode output: 732050\n\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Received task to validate population figures for Seattle and Colville from the 2020 census and compute the difference.\","
                    " \"Node_Action\": \"Stated population figures (Seattle=737,015; Colville=4,965) as if retrieved from census.gov without executing an actual data fetch, then computed the difference.\","
                    " \"Result_and_Feedback\": \"Sandbox returned 732,050 (exitcode=0), but the input population figures were asserted without web verification.\"}\n\n"
                    "-- Example 4: Verification/DataVerification Expert — reviewing upstream output --\n"
                    "Input:\nAgent: DataVerification_Expert\n"
                    "Content: Given the outputs from the script, the number of revisions before the release month is listed as zero, "
                    "which is highly improbable for such a notable game. We need to re-examine the Wikipedia revision history collection logic.\n"
                    "[Execution Result from Sandbox]:\nexitcode: 0\nCode output: Name: God of War, Release: April 2018, Revisions before release: 0\n\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Received code output showing 0 Wikipedia revisions before God of War's April 2018 release, which needs verification.\","
                    " \"Node_Action\": \"Flagged the result as implausible and proposed re-examining the revision history fetching logic in the script.\","
                    " \"Result_and_Feedback\": \"Identified a likely bug in the upstream script; revision count of 0 for a major game release is considered incorrect.\"}"
                )

            for s_idx in range(len(chat_history)):
                s_agent = chat_history[s_idx].get(index_agent, 'Unknown')
                s_content = chat_history[s_idx].get('content', '')

                if s_agent.lower() == "computer_terminal":
                    step_summaries[s_idx] = "(Execution Feedback bundled in previous step)"
                    continue

                combined_content = s_content
                if s_idx + 1 < len(chat_history):
                    next_agent = chat_history[s_idx + 1].get(index_agent, 'Unknown')
                    if next_agent.lower() == "computer_terminal":
                        combined_content += f"\n\n[Execution Result from Sandbox]:\n{chat_history[s_idx+1].get('content', '')}"

                past_summaries_context = ""
                for i in range(max(0, s_idx - 2), s_idx):
                    if step_summaries[i] and "bundled" not in step_summaries[i]:
                        past_summaries_context += f"[Step {i}] {chat_history[i].get(index_agent, 'Unknown')}: {step_summaries[i]}\n"

                summary_user = ""
                if past_summaries_context:
                    summary_user += f"--- Previous Steps Context ---\n{past_summaries_context}\n\n"
                summary_user += f"--- Current Step to Extract ---\nAgent: {s_agent}\nContent: {combined_content}"

                msg = [{"role": "system", "content": summary_sys}, {"role": "user", "content": summary_user}]
                res = _tracked_call("ds-v3.2", msg, 512, "phase1")
                step_summaries[s_idx] = res if res else "(Failed)"

            # Persist to cache after successful extraction
            summaries_cache[json_file] = step_summaries
            try:
                with open(cache_path, "w", encoding="utf-8") as _f:
                    _json.dump(summaries_cache, _f, ensure_ascii=False, indent=2)
            except Exception as _e:
                print(f"[Phase1 Cache] Warning: could not save cache: {_e}")

        error_found = False

        # Phase 2: Forward probe scan (GT removed from prompt)
        for idx, entry in enumerate(chat_history):
            agent_name = entry.get(index_agent, 'Unknown Agent')

            if agent_name.lower() == "computer_terminal":
                continue

            current_full_text = entry.get('content', '')
            if idx + 1 < len(chat_history):
                next_agent = chat_history[idx + 1].get(index_agent, 'Unknown')
                if next_agent.lower() == "computer_terminal":
                    current_full_text += f"\n\n[Sandbox Execution Result (Step {idx+1})]:\n{chat_history[idx+1].get('content', '')}"

            dist_past = ""
            for i in range(0, idx):
                if chat_history[i].get(index_agent, '').lower() == "computer_terminal": continue
                dist_past += f"[Step {i}] {chat_history[i].get(index_agent, 'Unknown')}: (JSON SUMMARY) {step_summaries[i]}\n"

            valid_agent_count = 0
            window_end = idx
            while valid_agent_count < 3 and window_end < len(chat_history):
                if chat_history[window_end].get(index_agent, '').lower() != "computer_terminal":
                    valid_agent_count += 1
                window_end += 1

            forward_window = ""
            for i in range(idx, window_end):
                if chat_history[i].get(index_agent, '').lower() == "computer_terminal": continue
                fw_text = chat_history[i].get('content', '')
                if i + 1 < len(chat_history) and chat_history[i+1].get(index_agent, '').lower() == "computer_terminal":
                    fw_text += f"\n[Sandbox Feedback]: {chat_history[i+1].get('content', '')}"
                forward_window += f"[Step {i}] {chat_history[i].get(index_agent, 'Unknown')}: (FULL TEXT)\n{fw_text}\n\n"

            future_down = ""
            for i in range(window_end, len(chat_history)):
                if chat_history[i].get(index_agent, '').lower() == "computer_terminal": continue
                future_down += f"[Step {i}] {chat_history[i].get(index_agent, 'Unknown')}: (JSON SUMMARY) {step_summaries[i]}\n"

            hierarchical_context = ""
            if dist_past: hierarchical_context += f"--- SCANNED PAST (JSON Summaries) ---\n{dist_past}\n"
            hierarchical_context += f"--- HIGH-RESOLUTION WINDOW (Steps {idx} to {min(len(chat_history)-1, idx+2)}) ---\n{forward_window}\n"
            if future_down: hierarchical_context += f"--- DISTANT FUTURE (JSON Summaries) ---\n{future_down}\n"

            # [noGT] Ground truth line removed; model judges from role + context only
            prompt = (
                "You are an AI Accuser evaluating a localized multi-agent conversation window to identify the Final Proximal Cause of failure.\n"
                "Note: 'Computer_terminal' is a sandbox environment output.\n"
                f"Problem: {problem}\n\n"
                f"Hierarchical view:\n{hierarchical_context}\n"
                f"--- ROLE-BASED EVALUATION TASK ---\n"
                f"Evaluate Step {idx} by '{agent_name}'. You MUST employ the following Strict Liability logic before deciding:\n"
                f"- OBSERVABLE ERROR THRESHOLD [CRITICAL]: A step should ONLY be marked as 'Root Error: Yes' if there is a concrete, observable manifestation of error in THAT SPECIFIC STEP's output — such as incorrect values propagated, failed assertions reported, hallucinated data introduced, or explicitly executing a catastrophic plan. Merely possessing a verification/planning role without producing a demonstrable error output in this exact step is INSUFFICIENT grounds for conviction.\n"
                f"- ROLE-ALIGNMENT VALIDATION: Did '{agent_name}' passively accept, execute, or propagate erroneous upstream data or plans without raising issues or performing independent verifications? If '{agent_name}' has any capability for reasoning or verification based on their role, unverified compliance with erroneous upstream directives constitutes an active failure. In this case, mark 'Root Error: Yes'.\n"
                f"- STRICT EXECUTOR EXCEPTION: If '{agent_name}' is a strictly passive actuator (e.g., 'Computer_terminal') with absolutely no capacity for reasoning, and it merely returned the execution output of an upstream command without modification, mark 'Root Error: No'.\n"
                f"- UPSTREAM INHERITANCE CHECK [CRITICAL — BACKWARD SCAN]: Before marking 'Root Error: Yes', examine the INPUTS received by Step {idx}. "
                f"If the error was ALREADY PRESENT in the upstream data or plan that '{agent_name}' received, then '{agent_name}' is PROPAGATING — not ORIGINATING — the error. In that case, mark 'Root Error: No'. "
                f"Only mark 'Root Error: Yes' if there is clear evidence that the error was INTRODUCED or MATERIALLY AMPLIFIED by '{agent_name}' at this specific step "
                f"(i.e., the upstream input was correct or neutral, but this step's output was wrong or corrupted). "
                f"A step that 'could have caught' an upstream error but did not is a MISSED VERIFICATION, not a root cause — do NOT mark it 'Root Error: Yes' for this reason alone.\n\n"
                "Respond ONLY in the exact format:\n"
                "1. Root Error: [Yes/No]\n"
                "2. Reason: [Your rigorous deduction based strictly on the agent's role constraints and verification duties]"
            )

            messages = [{"role": "system", "content": "You are a precise step-by-step evaluator."}, {"role": "user", "content": prompt}]
            answer = _tracked_call("ds-v3.2", messages, max_tokens, "probe")

            if not answer: continue
            print(f"Probe Evaluation for Step {idx}: {answer}")

            ans_lower = answer.lower()
            if "yes" in ans_lower[:20]:
                reason = answer.split('Reason:', 1)[-1].strip() if 'Reason:' in answer else answer
                print(f"Probe flagged error at Step {idx}. Invoking DAO Arbitration...")

                # [noGT] Ground Truth line removed from base_context
                base_context = (
                    f"General Task Origin: {problem}\n"
                    f"MAS Agent Roles Info:\n{roles_desc}\n"
                    "Note: 'Computer_terminal' is a sandbox environment output.\n\n"
                    "You have access to hierarchical context showing:\n"
                    "Immediate agents: Full details\n"
                    "Nearby agents: Key decisions\n"
                    "Distant agents: Brief summaries\n\n"
                    "The agents are numbered sequentially (Step 0, Step 1, etc.) corresponding to their turn index.\n\n"
                    "--- CASE DOSSIER (Hierarchical Log) ---\n"
                    f"{hierarchical_context}\n"
                    "--- FLAG FOR REVIEW & ATTRIBUTION CONSTRAINTS ---\n"
                    f"A frontline scanner has flagged Step {idx} (Agent: '{agent_name}') for review.\n"
                    "You are evaluating a LOCALIZED window of action. You must strictly adhere to the following liability doctrines:\n"
                    "1. [Role-Alignment Validation]: Consult the 'MAS Agent Roles Info' to identify each agent's system prompt. Evaluate an agent's performance strictly against its predefined responsibilities. If an agent's system prompt specifies duties such as verification, data interpretation, or factual accuracy, and the agent processes incorrect upstream input without detecting the error, this constitutes a failure in its designated role.\n"
                    "2. [Execution Accountability]: Avoid automatically attributing failures to the initial planner or data source. If an upstream step provides erroneous instructions or data, and a downstream agent executes or utilizes it without performing the verifications required by its role, the primary fault lies with the downstream agent for unverified execution.\n"
                    "3. [Proximal Cause Attribution]: The objective is to identify the final proximal cause of the failure. This corresponds to the precise step where the error was finalized due to an agent not fulfilling the functional requirements specified in its system prompt.\n"
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
                    '    "type": "single_agent",\n'
                    '    "attribution": ["agent_name"],\n'
                    '    "mistake_step": 1,\n'
                    '    "confidence": 1.0,\n'
                    '    "reasoning": "Detailed explanation of your primary conclusion including which step the error occurred"\n'
                    '  }\n'
                    "}\n"
                    "Be thorough, objective, and consider all possibilities including that no single agent may be clearly at fault.\n"
                    "Pay special attention to identifying the specific step/turn where the error truly originated. "
                    "RULE FOR STEP SELECTION: If an agent's failure spans multiple steps, select the EXACT step where the agent first committed an ACTIVE error (e.g., hallucinating data, writing flawed code, or making a false assumption), NOT a step where a tool simply returned empty results."
                )



                dao_sys_prompt_base = (
                    "You are an Objective Analysis Agent conducting an impartial investigation "
                    "to determine error attribution in a multi-agent conversation.\n\n"
                    "ANALYST SPECIALIZATION: {focus}\n\n"
                    "Your task:\n"
                    "1. Analyze ALL agents in the conversation objectively (not just one specific agent)\n"
                    "2. Identify which agent bears primary liability for the final incorrect outcome\n"
                    "3. Pinpoint the exact step where the agent's failure was FIRST committed (not where it was observed)\n"
                    "4. Assign error likelihood scores [0.0–1.0] with evidence-backed reasoning\n\n"
                    "You have access to a three-layer hierarchical log:\n"
                    "- SCANNED PAST (JSON Summaries): Compressed upstream context before the flagged window\n"
                    "- HIGH-RESOLUTION WINDOW: Full-text of the flagged step and its immediate neighbors — "
                    "this is the primary zone of investigation\n"
                    "- DISTANT FUTURE (JSON Summaries): Compressed downstream propagation of the error\n\n"
                    "You MUST apply the following liability doctrines before reaching any conclusion:\n"
                    "- [OBSERVABLE ERROR THRESHOLD]: Convict ONLY when there is a concrete, observable "
                    "manifestation of error in the agent's output — incorrect values propagated, hallucinated "
                    "data introduced, flawed code written, or a catastrophic plan executed. Role possession "
                    "alone is insufficient grounds for conviction.\n"
                    "- [ROLE-ALIGNMENT VALIDATION]: An agent with verification or reasoning duties who "
                    "silently accepts and propagates erroneous upstream output has actively failed its role. "
                    "This constitutes a culpable act, not passive compliance.\n"
                    "- [EXECUTION ACCOUNTABILITY]: Do not default to blaming the planner or the initial "
                    "data source. If a downstream agent executes upstream errors without performing "
                    "the verifications its role requires, liability transfers to that downstream agent.\n"
                    "- [PROXIMAL CAUSE DOCTRINE]: Identify the final proximal cause — the last agent "
                    "whose failure, if corrected, would have prevented the outcome.\n"
                    "- [ROOT CAUSE vs MANIFESTATION POINT]: Critically distinguish between where the error "
                    "ORIGINATED and where it became OBSERVABLE. A Verification or downstream agent that "
                    "FAILED TO CATCH an upstream error is a MANIFESTATION POINT — it amplified the error "
                    "but did not create it. The ROOT CAUSE is the agent whose output FIRST introduced the "
                    "incorrect information or flawed decision into the pipeline. Always trace the error "
                    "backward to its true ORIGIN before assigning primary liability. Do NOT assign high "
                    "guilt scores to agents solely because they are the last to touch incorrect data.\n"
                    "- [STRICT EXECUTOR EXCEPTION]: A purely passive executor (e.g., Computer_terminal) "
                    "with no reasoning capacity is exempt from conviction for faithfully returning "
                    "the output of an upstream command.\n\n"
                    "The conversation steps are numbered sequentially (Step 0, Step 1, ...) "
                    "corresponding to their turn index in the MAS interaction log."
                )

                focus_factual = (
                    "You are a Factual and Information Retrieval Attribution Expert "
                    "targeting failures rooted in unverified, hallucinated, temporally stale, "
                    "or structurally incorrect information.\n\n"
                    "FAILURE PATTERNS YOU PRIORITIZE (derived from empirical cluster analysis, n=70):\n"
                    "- Hallucinated content introduced without source verification\n"
                    "- Reliance on outdated or static internal knowledge instead of dynamic external retrieval\n"
                    "- Incorrect temporal or sequential data extraction (wrong year, wrong version, wrong period)\n"
                    "- Incorrect data schema, column reference, or structural element targeting\n"
                    "- Failure to verify information against the authoritative source before propagating downstream\n"
                    "- Tool selection failure: using an inappropriate tool for the retrieval context\n"
                    "- Incorrect transcription or misinterpretation of ambiguous source data\n"
                    "- Reliance on secondary/unverified external data instead of primary authoritative source\n"
                    "- Missing or inaccessible required data source causing silent incorrect assumption\n"
                    "- Failure to challenge improbable retrieved results (sanity-check omission)\n\n"
                    "BOUNDARY: Do NOT attribute blame based on reasoning logic or coordination failures."
                )

                focus_logic = (
                    "You are a Logic, Reasoning, and Planning Attribution Expert "
                    "targeting failures rooted in flawed internal reasoning, incorrect algorithms, "
                    "or erroneous plans regardless of input data quality.\n\n"
                    "FAILURE PATTERNS YOU PRIORITIZE (derived from empirical cluster analysis, n=55):\n"
                    "- Incorrect algorithm or formula implementation (e.g., wrong modulo, wrong formula variant)\n"
                    "- Incorrect initial task decomposition or goal formulation by the planning agent\n"
                    "- Task domain misidentification or misaligned focus on irrelevant task aspects\n"
                    "- Premature conclusion without completing required action or verification steps\n"
                    "- Premature solution synthesis without systematic intermediate reasoning\n"
                    "- Incorrect interpretation of notation, units, or output format leading to miscalculation\n"
                    "- Logical inconsistency in handling contradictory truth conditions\n"
                    "- Incorrect conversion between numerical representation systems (e.g., binary, decimal, base-N)\n"
                    "- Simulation logic flaw due to incorrect state transition modeling\n"
                    "- Incorrect problem decomposition or task allocation among agents\n"
                    "- Incorrect verification of intermediate results (accepting wrong intermediate output)\n\n"
                    "BOUNDARY: Do NOT blame data quality or inter-agent coordination failures."
                )

                focus_execution = (
                    "You are an Agent Execution and Coordination Attribution Expert "
                    "targeting failures in role execution, corrupted output propagation, "
                    "or cross-agent coordination contract violations.\n\n"
                    "FAILURE PATTERNS YOU PRIORITIZE (derived from empirical cluster analysis, n=59):\n"
                    "- Agent provided inaccurate foundational data to downstream agents (cascading corruption source)\n"
                    "- Agent role boundary violation causing process termination or incorrect delegation\n"
                    "- Agent executed code with incorrect or non-functional logic (runtime/semantic errors)\n"
                    "- Agent exceeded computational or interaction limits due to repeated faulty execution loops\n"
                    "- Cross-agent verification failure: downstream agent executed erroneous upstream output unchecked\n"
                    "- Agent bypassed required analysis and made an unverified assumption\n"
                    "- Verification agent provided erroneous procedural instructions instead of correcting them\n"
                    "- Agent misapplied data generation instead of retrieval (fabricating instead of fetching)\n"
                    "- Agent action omission: skipped a required step causing information gap downstream\n"
                    "- Inaccurate or fabricated source data introduced by an agent and consumed by others\n"
                    "- Agent dependency failure: non-functional code or tool call blocking pipeline progress\n\n"
                    "BOUNDARY: Do NOT blame reasoning quality or raw data issues."
                )

                sys_1 = dao_sys_prompt_base.format(focus=focus_factual)
                sys_2 = dao_sys_prompt_base.format(focus=focus_logic)
                sys_3 = dao_sys_prompt_base.format(focus=focus_execution)

                if sys_prompts is not None and len(sys_prompts) == 3:
                    sys_1, sys_2, sys_3 = sys_prompts

                _DAO_MODEL_ALIASES = ["ds-v3.2", "ds-v3.2", "ds-v3.2"]

                def _call_dao(args):
                    i, sys_p, usr_p = args
                    msg = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
                    return _tracked_call(_DAO_MODEL_ALIASES[i], msg, max_tokens, "dao")

                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futs = {executor.submit(_call_dao, (i, s, u)): i for i, (s, u) in enumerate([(sys_1, dao_1), (sys_2, dao_2), (sys_3, dao_3)])}
                    results = [None] * 3
                    for f in concurrent.futures.as_completed(futs):
                        results[futs[f]] = f.result()

                CONVICTION_THRESHOLD = 0.5
                agent_scores = {a: 0.0 for a in valid_agents}
                max_single_agent_scores = {a: 0.0 for a in valid_agents}
                step_scores = {}
                judge_votes = []
                valid_resp = 0

                for i, r in enumerate(results):
                    if r:
                        print(f"--- Arbitrator {i+1} response ---\n{r}\n--- End Arbitrator {i+1} ---\n")
                        try:
                            jstr = re.search(r'\{.*\}', r, re.DOTALL).group(0)
                            jdata = json.loads(jstr)
                            if 'primary_conclusion' not in jdata:
                                continue
                            pc = jdata['primary_conclusion']
                            conf = float(pc.get('confidence', 1.0))

                            attr = pc.get('attribution', [])
                            if isinstance(attr, str): attr = [attr]
                            if isinstance(attr, list):
                                for a in attr:
                                    if a not in agent_scores: agent_scores[a] = 0.0
                                    if a not in max_single_agent_scores: max_single_agent_scores[a] = 0.0
                                    agent_scores[a] += conf
                                    if conf > max_single_agent_scores[a]:
                                        max_single_agent_scores[a] = conf

                            step_val = str(pc.get('mistake_step', idx))
                            step_scores[step_val] = step_scores.get(step_val, 0.0) + conf

                            judge_agent = attr[0] if attr else agent_name
                            judge_votes.append((judge_agent, step_val, conf))
                            valid_resp += 1
                        except: pass

                if valid_resp > 0:
                    for a in agent_scores: agent_scores[a] /= valid_resp
                    for s in step_scores:  step_scores[s]  /= valid_resp

                    avg_val  = max(agent_scores.values()) if agent_scores else 0
                    max_vote = max(max_single_agent_scores.values()) if max_single_agent_scores else 0

                    if avg_val >= CONVICTION_THRESHOLD or max_vote > 0.85:
                        final_culprit = max(agent_scores, key=agent_scores.get) if agent_scores else agent_name

                        culprit_step_votes = {}
                        for jg_agent, jg_step, jg_conf in judge_votes:
                            if jg_agent == final_culprit:
                                culprit_step_votes[jg_step] = culprit_step_votes.get(jg_step, 0.0) + jg_conf

                        if culprit_step_votes:
                            best_step_str = max(culprit_step_votes, key=culprit_step_votes.get)
                        elif step_scores:
                            valid_steps = [str(s) for s, d in enumerate(chat_history) if d.get(index_agent) == final_culprit]
                            f_scores = {k: v for k, v in step_scores.items() if k in valid_steps}
                            if f_scores:
                                max_score = max(f_scores.values())
                                candidates = [k for k, v in f_scores.items() if v == max_score]
                                best_step_str = min(candidates, key=lambda x: int(x) if x.isdigit() else 0)
                            else:
                                best_step_str = max(step_scores, key=step_scores.get)
                        else:
                            best_step_str = str(idx)

                        try:
                            culprit_step = int(best_step_str)
                        except ValueError:
                            culprit_step = idx

                        valid_steps_for_culprit = [str(s) for s, d in enumerate(chat_history) if d.get(index_agent) == final_culprit]
                        if valid_steps_for_culprit and str(culprit_step) not in valid_steps_for_culprit:
                            culprit_step = min(int(s) for s in valid_steps_for_culprit)
                            print(f"  [Ghost-step guard] Step corrected to earliest real appearance: {culprit_step}")

                        print(f"Error conclusively found at step {culprit_step}. (Avg: {avg_val:.2f}, MaxVote: {max_vote:.2f}) Halting forward scan.")
                        print(f"\nPrediction for {json_file}: Error found.")
                        print(f"Agent Name: {final_culprit}")
                        print(f"Step Number: {culprit_step}")
                        print(f"Reason provided by DAO: Convicted by ECHo Tribunal [noGT]. (Threshold Matched)")
                        error_found = True
                        break
                    else:
                        print(f"DAO absolved step {idx} (Avg: {avg_val:.2f}, MaxVote: {max_vote:.2f}). Continuing scan...")
                else:
                    print("All DAO calls failed. Assuming error.")
                    print(f"\nPrediction for {json_file}: Error found.")
                    print(f"Agent Name: {agent_name}")
                    print(f"Step Number: {idx}")
                    error_found = True
                    break

        if not error_found:
            agent_0_name = chat_history[0].get(index_agent, 'Unknown Agent')
            print(f"Probe missed the error entirely. Activating Step 0 Planning Failsafe.")
            print(f"\nPrediction for {json_file}: Error found.")
            print(f"Agent Name: {agent_0_name}")
            print(f"Step Number: 0")
            print(f"Reason provided by DAO: (Failsafe) [noGT] Initial planning failure or hallucination bypassing downstream probes.")

    # ── Token consumption summary ─────────────────────────────────────────────
    total = sum(token_stats.values())
    print("\n" + "=" * 50)
    print("[noGT] Token Consumption Summary")
    print(f"  Phase 1 (Extraction, DS-V3.2): {token_stats['phase1']:>8,} tokens")
    print(f"  Phase 2 (Probe,     DS-R1    ): {token_stats['probe']:>8,} tokens")
    print(f"  Phase 3 (DAO,       3-model  ): {token_stats['dao']:>8,} tokens")
    print(f"  {'TOTAL':30s}: {total:>8,} tokens")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# AgentJury Enhanced: Dual-Trigger Probe + DAO
# ---------------------------------------------------------------------------

def agentJury_enhance(client, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int, sys_prompts=None):
    """
    Enhanced variant of step_by_step_dao_echo_api_noGT.

    Probe logic change:
      - First probe hit  → save context window, CONTINUE scanning (do not invoke DAO yet)
      - Second probe hit → combine both flagged windows into a dual-dossier and invoke DAO
      - Only one hit     → fallback: invoke DAO on that single flag (same as original)
      - Zero hits        → Step 0 Failsafe

    Rationale: the first hit often captures a downstream symptom (manifestation point).
    The second hit provides a contrasting anchor, giving the DAO tribunal richer signal
    to distinguish root cause from propagation.

    identical to step_by_step_dao_echo_api_noGT.
    """
    import os, re, json as _json, concurrent.futures
    from tqdm import tqdm
    from Lib.api_utils import _load_json_data, _get_sorted_json_files

    print("\n--- [AgentJury-Enhanced] Dual-Trigger Probe + DAO Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"
    CONVICTION_THRESHOLD = 0.5

    # ── Token tracking ────────────────────────────────────────────────────────
    token_stats = {"phase1": 0, "probe": 0, "dao": 0}

    def _tracked_call(model_alias: str, messages: list, max_tokens_val: int, phase: str) -> str | None:
        full_model = API_MODEL_MAP.get(model_alias, model_alias)
        for attempt in range(3):
            try:
                call_kwargs = dict(
                    model=full_model,
                    messages=messages,
                    max_tokens=max_tokens_val,
                    stream=True,
                    temperature=0.5,
                    stream_options={"include_usage": True},
                )
                if model_alias == "ds-v3.2":
                    call_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                response = client.chat.completions.create(**call_kwargs)
                full_content = ""
                last_usage = None
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta:
                        delta_content = getattr(chunk.choices[0].delta, "content", None)
                        if delta_content:
                            full_content += delta_content
                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        last_usage = chunk.usage
                if last_usage is not None:
                    token_stats[phase] += getattr(last_usage, "total_tokens", 0)
                return full_content.strip() or None
            except Exception as e:
                print(f"[AJE] {phase} API call attempt {attempt+1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(2)
        return None

    # ── Phase 1 summary cache ─────────────────────────────────────────────────
    dataset_name = os.path.basename(os.path.normpath(directory_path))
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "summaries_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_name}.json")

    summaries_cache: dict = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as _f:
                summaries_cache = _json.load(_f)
            print(f"[Phase1 Cache] Loaded {len(summaries_cache)} cached files from {cache_path}\n")
        except Exception:
            summaries_cache = {}

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data: continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")

        system_prompt_dict = data.get("system_prompt", {})
        valid_agents = list(system_prompt_dict.keys()) if isinstance(system_prompt_dict, dict) else []

        roles_desc = ""
        if isinstance(system_prompt_dict, dict):
            for k, v in system_prompt_dict.items():
                roles_desc += f"--- Role: {k} ---\n{v}\n"

        if not chat_history:
            continue

        print(f"--- Analyzing File: {json_file} ---")

        # ── Phase 1: Structured extraction with cache ─────────────────────────
        if json_file in summaries_cache:
            step_summaries = summaries_cache[json_file]
            if len(step_summaries) != len(chat_history):
                step_summaries = (step_summaries + [""] * len(chat_history))[:len(chat_history)]
            print(f"[Phase1 Cache] HIT — skipping LLM extraction for {json_file}")
        else:
            print(f"--- [Phase1] Extracting Contextual Log for {json_file} ---")
            step_summaries = [""] * len(chat_history)

            if is_handcrafted:
                summary_sys = (
                    "You are a structured data extractor for multi-agent web-research task logs. "
                    "Output ONLY a valid JSON object with EXACTLY these three keys:\n"
                    "- 'Upstream_Instruction_or_Context': The instruction, goal, or context this agent received (max 1 sentence).\n"
                    "- 'Node_Action': What this agent decided, planned, delegated, navigated, or retrieved (max 1 sentence).\n"
                    "- 'Result_and_Feedback': The direct outcome — page content obtained, sub-task result returned, or conclusion reached (max 1 sentence).\n"
                    "Do NOT include markdown formatting like ```json.\n\n"
                    "### FEW-SHOT EXAMPLES ###\n\n"
                    "-- Example 1: human --\n"
                    "Input:\nAgent: human\n"
                    "Content: Where can I take martial arts classes within a five-minute walk from the New York Stock Exchange after work (7-9 pm)?\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"User initiated the task requesting martial arts classes near NYSE available 7-9 pm.\","
                    " \"Node_Action\": \"Submitted the original question to the multi-agent system.\","
                    " \"Result_and_Feedback\": \"Task handed off to Orchestrator for planning and delegation.\"}\n\n"
                    "-- Example 2: Orchestrator (thought) — initial planning --\n"
                    "Input:\nAgent: Orchestrator (thought)\n"
                    "Content: Initial plan: We need to find martial arts schools within 5-minute walk from NYSE with 7-9 pm classes. "
                    "Team: WebSurfer for web navigation. Step 1: search for nearby schools. Step 2: verify addresses and schedules.\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Received user task to find nearby martial arts classes after work hours.\","
                    " \"Node_Action\": \"Formulated a two-step plan: first search for schools near NYSE, then verify addresses and evening schedules.\","
                    " \"Result_and_Feedback\": \"Plan established; delegating web search to WebSurfer as next action.\"}\n\n"
                    "-- Example 3: Orchestrator (thought) — ledger update --\n"
                    "Input:\nAgent: Orchestrator (thought)\n"
                    "Content: Updated Ledger: {\"is_request_satisfied\": {\"reason\": \"No confirmed martial arts school with evening hours found yet.\", \"answer\": false}, "
                    "\"is_in_loop\": {\"reason\": \"WebSurfer navigated to an unrelated page last step, no forward progress.\", \"answer\": true}}\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Reviewing WebSurfer's last action which landed on an irrelevant product page.\","
                    " \"Node_Action\": \"Updated task ledger: marked request as unsatisfied and loop detected due to repeated off-target navigation.\","
                    " \"Result_and_Feedback\": \"Determined the agent is stuck in a loop; will issue corrective delegation to redirect WebSurfer.\"}\n\n"
                    "-- Example 4: Orchestrator (-> WebSurfer) — delegation --\n"
                    "Input:\nAgent: Orchestrator (-> WebSurfer)\n"
                    "Content: Please visit the Astronomy Picture of the Day Archive 2015 page on nasa.gov and navigate to the first week of August 2015 to find the specific image.\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Ledger shows the NASA APOD image for August 2015 has not been identified yet.\","
                    " \"Node_Action\": \"Delegated a targeted navigation instruction to WebSurfer to visit the NASA APOD 2015 archive page.\","
                    " \"Result_and_Feedback\": \"Instruction sent; awaiting WebSurfer's page content and image identification.\"}\n\n"
                    "-- Example 5: WebSurfer — successful navigation --\n"
                    "Input:\nAgent: WebSurfer\n"
                    "Content: I clicked 'Thriller (album) - Wikipedia'. "
                    "Here is a screenshot of [Thriller (album) - Wikipedia](https://en.wikipedia.org/wiki/Thriller_(album)). "
                    "The page shows singles released: The Girl Is Mine (1st), Billie Jean (2nd), Beat It (3rd), Wanna Be Startin' Somethin' (4th), Human Nature (5th).\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Received instruction to confirm the fifth single from Michael Jackson's Thriller album.\","
                    " \"Node_Action\": \"Navigated to the Thriller Wikipedia page and located the singles release order in the article.\","
                    " \"Result_and_Feedback\": \"Successfully retrieved: the fifth single from Thriller is 'Human Nature'.\"}\n\n"
                    "-- Example 6: WebSurfer — wrong navigation / irrelevant page --\n"
                    "Input:\nAgent: WebSurfer\n"
                    "Content: I clicked 'NY Jidokwan Taekwondo'. "
                    "Here is a screenshot of [薄膜/透明體測厚專家 - VK系列雷射顯微鏡 | KEYENCE 台灣基恩斯](https://www.keyence.com.tw/...). "
                    "The viewport shows a microscopy equipment product page unrelated to martial arts.\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Received instruction to click on martial arts school links and note their addresses and schedules.\","
                    " \"Node_Action\": \"Clicked 'NY Jidokwan Taekwondo' link but was redirected to a KEYENCE microscopy product page.\","
                    " \"Result_and_Feedback\": \"Navigation failed: landed on an irrelevant commercial page with no martial arts school information.\"}\n\n"
                    "-- Example 7: FileSurfer --\n"
                    "Input:\nAgent: FileSurfer\n"
                    "Content: Opening the uploaded file about monday.com IPO documents. "
                    "File contains a table of C-suite executives as of June 2021 IPO date: Roy Mann (Co-CEO), Eran Zinman (Co-CEO), Eliran Glazer (CFO).\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Received instruction to read the monday.com IPO document and extract C-suite executives at the time of IPO.\","
                    " \"Node_Action\": \"Opened and parsed the IPO document file, located the executive leadership table.\","
                    " \"Result_and_Feedback\": \"Extracted three C-suite members as of IPO (June 2021): Co-CEOs Roy Mann and Eran Zinman, CFO Eliran Glazer.\"}"
                )
            else:
                summary_sys = (
                    "You are a structured data extractor. You must analyze the agent's interaction and output ONLY a valid JSON object. "
                    "Do NOT include markdown formatting like ```json. The JSON object MUST have EXACTLY these three keys:\n"
                    "- 'Upstream_Instruction_or_Context': Briefly describe the instruction, goal, or context passed to this agent (max 1 sentence).\n"
                    "- 'Node_Action': Briefly describe what this agent decided or code it wrote (max 1 sentence).\n"
                    "- 'Result_and_Feedback': Briefly describe the final output, resolution, or sandbox execution result (max 1 sentence).\n\n"
                    "### FEW-SHOT EXAMPLES ###\n\n"
                    "-- Example 1: Domain Expert — code execution success --\n"
                    "Input:\nAgent: Excel_Expert\n"
                    "Content: Write python code to determine how many clients will receive sunset awnings (even address).\n"
                    "```python\neven_address = data[data['Street Number'] % 2 == 0]\nprint(len(even_address))\n```\n"
                    "[Execution Result from Sandbox]:\nexitcode: 0\nCode output: 4\n\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Received the task to count clients with even street numbers from the Excel file.\","
                    " \"Node_Action\": \"Wrote a Pandas script using modulo 2 on the Street Number field to filter and count even addresses.\","
                    " \"Result_and_Feedback\": \"Sandbox executed successfully (exitcode=0), returning a count of 4.\"}\n\n"
                    "-- Example 2: Domain Expert — code execution failure (exitcode=1) --\n"
                    "Input:\nAgent: Boggle_Board_Expert\n"
                    "Content: Here is the DFS implementation to find the longest word on the Boggle board:\n"
                    "```python\nlongest_word = find_longest_word(boggle_board, dictionary)\nprint(longest_word)\n```\n"
                    "[Execution Result from Sandbox]:\nexitcode: 1\n"
                    "Code output: NameError: name 'dictionary' is not defined\n\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Received the task to implement DFS to find the longest valid word on the Boggle board using a pre-loaded dictionary.\","
                    " \"Node_Action\": \"Wrote a DFS-based word search script referencing the variable 'dictionary' without loading it in the same code block.\","
                    " \"Result_and_Feedback\": \"Sandbox execution failed (exitcode=1) with NameError: 'dictionary' is not defined.\"}\n\n"
                    "-- Example 3: Domain Expert — wrong factual assumption, no code error --\n"
                    "Input:\nAgent: Validation_Expert\n"
                    "Content: The following are the population figures retrieved from data.census.gov:\n"
                    "- Seattle, Washington (2020 Census): 737,015\n"
                    "- Colville, Washington (2020 Census): 4,965\n"
                    "```python\npopulation_seattle = 737015\npopulation_colville = 4965\nprint(population_seattle - population_colville)\n```\n"
                    "[Execution Result from Sandbox]:\nexitcode: 0\nCode output: 732050\n\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Received task to validate population figures for Seattle and Colville from the 2020 census and compute the difference.\","
                    " \"Node_Action\": \"Stated population figures (Seattle=737,015; Colville=4,965) as if retrieved from census.gov without executing an actual data fetch, then computed the difference.\","
                    " \"Result_and_Feedback\": \"Sandbox returned 732,050 (exitcode=0), but the input population figures were asserted without web verification.\"}\n\n"
                    "-- Example 4: Verification/DataVerification Expert — reviewing upstream output --\n"
                    "Input:\nAgent: DataVerification_Expert\n"
                    "Content: Given the outputs from the script, the number of revisions before the release month is listed as zero, "
                    "which is highly improbable for such a notable game. We need to re-examine the Wikipedia revision history collection logic.\n"
                    "[Execution Result from Sandbox]:\nexitcode: 0\nCode output: Name: God of War, Release: April 2018, Revisions before release: 0\n\n"
                    "Output:\n"
                    "{\"Upstream_Instruction_or_Context\": \"Received code output showing 0 Wikipedia revisions before God of War's April 2018 release, which needs verification.\","
                    " \"Node_Action\": \"Flagged the result as implausible and proposed re-examining the revision history fetching logic in the script.\","
                    " \"Result_and_Feedback\": \"Identified a likely bug in the upstream script; revision count of 0 for a major game release is considered incorrect.\"}"
                )

            for s_idx in range(len(chat_history)):
                s_agent = chat_history[s_idx].get(index_agent, 'Unknown')
                s_content = chat_history[s_idx].get('content', '')

                if s_agent.lower() == "computer_terminal":
                    step_summaries[s_idx] = "(Execution Feedback bundled in previous step)"
                    continue

                combined_content = s_content
                if s_idx + 1 < len(chat_history):
                    next_agent = chat_history[s_idx + 1].get(index_agent, 'Unknown')
                    if next_agent.lower() == "computer_terminal":
                        combined_content += f"\n\n[Execution Result from Sandbox]:\n{chat_history[s_idx+1].get('content', '')}"

                past_summaries_context = ""
                for i in range(max(0, s_idx - 2), s_idx):
                    if step_summaries[i] and "bundled" not in step_summaries[i]:
                        past_summaries_context += f"[Step {i}] {chat_history[i].get(index_agent, 'Unknown')}: {step_summaries[i]}\n"

                summary_user = ""
                if past_summaries_context:
                    summary_user += f"--- Previous Steps Context ---\n{past_summaries_context}\n\n"
                summary_user += f"--- Current Step to Extract ---\nAgent: {s_agent}\nContent: {combined_content}"

                msg = [{"role": "system", "content": summary_sys}, {"role": "user", "content": summary_user}]
                res = _tracked_call("ds-v3.2", msg, 512, "phase1")
                step_summaries[s_idx] = res if res else "(Failed)"

            summaries_cache[json_file] = step_summaries
            try:
                with open(cache_path, "w", encoding="utf-8") as _f:
                    _json.dump(summaries_cache, _f, ensure_ascii=False, indent=2)
            except Exception as _e:
                print(f"[Phase1 Cache] Warning: could not save cache: {_e}")

        # ── DAO invocation helper (shared by single-flag and dual-flag paths) ──
        dao_sys_prompt_base = (
            "You are an Objective Analysis Agent conducting an impartial investigation "
            "to determine error attribution in a multi-agent conversation.\n\n"
            "ANALYST SPECIALIZATION: {focus}\n\n"
            "Your task:\n"
            "1. Analyze ALL agents in the conversation objectively (not just one specific agent)\n"
            "2. Identify which agent bears primary liability for the final incorrect outcome\n"
            "3. Pinpoint the exact step where the agent's failure was FIRST committed (not where it was observed)\n"
            "4. Assign error likelihood scores [0.0–1.0] with evidence-backed reasoning\n\n"
            "You have access to a three-layer hierarchical log:\n"
            "- SCANNED PAST (JSON Summaries): Compressed upstream context before the flagged window\n"
            "- HIGH-RESOLUTION WINDOW: Full-text of the flagged step and its immediate neighbors — "
            "this is the primary zone of investigation\n"
            "- DISTANT FUTURE (JSON Summaries): Compressed downstream propagation of the error\n\n"
            "You MUST apply the following liability doctrines before reaching any conclusion:\n"
            "- [OBSERVABLE ERROR THRESHOLD]: Convict ONLY when there is a concrete, observable "
            "manifestation of error in the agent's output — incorrect values propagated, hallucinated "
            "data introduced, flawed code written, or a catastrophic plan executed. Role possession "
            "alone is insufficient grounds for conviction.\n"
            "- [ROLE-ALIGNMENT VALIDATION]: An agent with verification or reasoning duties who "
            "silently accepts and propagates erroneous upstream output has actively failed its role. "
            "This constitutes a culpable act, not passive compliance.\n"
            "- [EXECUTION ACCOUNTABILITY]: Do not default to blaming the planner or the initial "
            "data source. If a downstream agent executes upstream errors without performing "
            "the verifications its role requires, liability transfers to that downstream agent.\n"
            "- [PROXIMAL CAUSE DOCTRINE]: Identify the final proximal cause — the last agent "
            "whose failure, if corrected, would have prevented the outcome.\n"
            "- [ROOT CAUSE vs MANIFESTATION POINT]: Critically distinguish between where the error "
            "ORIGINATED and where it became OBSERVABLE. A Verification or downstream agent that "
            "FAILED TO CATCH an upstream error is a MANIFESTATION POINT — it amplified the error "
            "but did not create it. The ROOT CAUSE is the agent whose output FIRST introduced the "
            "incorrect information or flawed decision into the pipeline. Always trace the error "
            "backward to its true ORIGIN before assigning primary liability. Do NOT assign high "
            "guilt scores to agents solely because they are the last to touch incorrect data.\n"
            "- [STRICT EXECUTOR EXCEPTION]: A purely passive executor (e.g., Computer_terminal) "
            "with no reasoning capacity is exempt from conviction for faithfully returning "
            "the output of an upstream command.\n\n"
            "The conversation steps are numbered sequentially (Step 0, Step 1, ...) "
            "corresponding to their turn index in the MAS interaction log."
        )

        focus_factual = (
            "You are a Factual and Information Retrieval Attribution Expert "
            "targeting failures rooted in unverified, hallucinated, temporally stale, "
            "or structurally incorrect information.\n\n"
            "FAILURE PATTERNS YOU PRIORITIZE (derived from empirical cluster analysis, n=70):\n"
            "- Hallucinated content introduced without source verification\n"
            "- Reliance on outdated or static internal knowledge instead of dynamic external retrieval\n"
            "- Incorrect temporal or sequential data extraction (wrong year, wrong version, wrong period)\n"
            "- Incorrect data schema, column reference, or structural element targeting\n"
            "- Failure to verify information against the authoritative source before propagating downstream\n"
            "- Tool selection failure: using an inappropriate tool for the retrieval context\n"
            "- Incorrect transcription or misinterpretation of ambiguous source data\n"
            "- Reliance on secondary/unverified external data instead of primary authoritative source\n"
            "- Missing or inaccessible required data source causing silent incorrect assumption\n"
            "- Failure to challenge improbable retrieved results (sanity-check omission)\n\n"
            "BOUNDARY: Do NOT attribute blame based on reasoning logic or coordination failures."
        )
        focus_logic = (
            "You are a Logic, Reasoning, and Planning Attribution Expert "
            "targeting failures rooted in flawed internal reasoning, incorrect algorithms, "
            "or erroneous plans regardless of input data quality.\n\n"
            "FAILURE PATTERNS YOU PRIORITIZE (derived from empirical cluster analysis, n=55):\n"
            "- Incorrect algorithm or formula implementation (e.g., wrong modulo, wrong formula variant)\n"
            "- Incorrect initial task decomposition or goal formulation by the planning agent\n"
            "- Task domain misidentification or misaligned focus on irrelevant task aspects\n"
            "- Premature conclusion without completing required action or verification steps\n"
            "- Premature solution synthesis without systematic intermediate reasoning\n"
            "- Incorrect interpretation of notation, units, or output format leading to miscalculation\n"
            "- Logical inconsistency in handling contradictory truth conditions\n"
            "- Incorrect conversion between numerical representation systems (e.g., binary, decimal, base-N)\n"
            "- Simulation logic flaw due to incorrect state transition modeling\n"
            "- Incorrect problem decomposition or task allocation among agents\n"
            "- Incorrect verification of intermediate results (accepting wrong intermediate output)\n\n"
            "BOUNDARY: Do NOT blame data quality or inter-agent coordination failures."
        )
        focus_execution = (
            "You are an Agent Execution and Coordination Attribution Expert "
            "targeting failures in role execution, corrupted output propagation, "
            "or cross-agent coordination contract violations.\n\n"
            "FAILURE PATTERNS YOU PRIORITIZE (derived from empirical cluster analysis, n=59):\n"
            "- Agent provided inaccurate foundational data to downstream agents (cascading corruption source)\n"
            "- Agent role boundary violation causing process termination or incorrect delegation\n"
            "- Agent executed code with incorrect or non-functional logic (runtime/semantic errors)\n"
            "- Agent exceeded computational or interaction limits due to repeated faulty execution loops\n"
            "- Cross-agent verification failure: downstream agent executed erroneous upstream output unchecked\n"
            "- Agent bypassed required analysis and made an unverified assumption\n"
            "- Verification agent provided erroneous procedural instructions instead of correcting them\n"
            "- Agent misapplied data generation instead of retrieval (fabricating instead of fetching)\n"
            "- Agent action omission: skipped a required step causing information gap downstream\n"
            "- Inaccurate or fabricated source data introduced by an agent and consumed by others\n"
            "- Agent dependency failure: non-functional code or tool call blocking pipeline progress\n\n"
            "BOUNDARY: Do NOT blame reasoning quality or raw data issues."
        )

        sys_1 = dao_sys_prompt_base.format(focus=focus_factual)
        sys_2 = dao_sys_prompt_base.format(focus=focus_logic)
        sys_3 = dao_sys_prompt_base.format(focus=focus_execution)
        if sys_prompts is not None and len(sys_prompts) == 3:
            sys_1, sys_2, sys_3 = sys_prompts

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
            '    "type": "single_agent",\n'
            '    "attribution": ["agent_name"],\n'
            '    "mistake_step": 1,\n'
            '    "confidence": 1.0,\n'
            '    "reasoning": "Detailed explanation of your primary conclusion including which step the error occurred"\n'
            '  }\n'
            "}\n"
            "Be thorough, objective, and consider all possibilities including that no single agent may be clearly at fault.\n"
            "Pay special attention to identifying the specific step/turn where the error truly originated. "
            "RULE FOR STEP SELECTION: If an agent's failure spans multiple steps, select the EXACT step where the agent first committed an ACTIVE error (e.g., hallucinating data, writing flawed code, or making a false assumption), NOT a step where a tool simply returned empty results."
        )

        def _invoke_dao(base_context_str: str, dao_reason: str, probe_step_idx: int, probe_agent: str) -> bool:
            """
            Returns True if error is conclusively found, False if DAO absolved.
            Prints final prediction on success.
            """
            nonlocal error_found



            _DAO_MODEL_ALIASES = ["ds-v3.2", "ds-v3.2", "ds-v3.2"]

            def _call_dao(args):
                i, sys_p, usr_p = args
                msg = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
                return _tracked_call(_DAO_MODEL_ALIASES[i], msg, max_tokens, "dao")

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futs = {executor.submit(_call_dao, (i, s, u)): i
                        for i, (s, u) in enumerate([(sys_1, dao_1), (sys_2, dao_2), (sys_3, dao_3)])}
                results = [None] * 3
                for f in concurrent.futures.as_completed(futs):
                    results[futs[f]] = f.result()

            agent_scores = {a: 0.0 for a in valid_agents}
            max_single_agent_scores = {a: 0.0 for a in valid_agents}
            step_scores = {}
            judge_votes = []
            valid_resp = 0

            for i, r in enumerate(results):
                if r:
                    print(f"--- Arbitrator {i+1} response ---\n{r}\n--- End Arbitrator {i+1} ---\n")
                    try:
                        jstr = re.search(r'\{.*\}', r, re.DOTALL).group(0)
                        jdata = json.loads(jstr)
                        if 'primary_conclusion' not in jdata:
                            continue
                        pc = jdata['primary_conclusion']
                        conf = float(pc.get('confidence', 1.0))

                        attr = pc.get('attribution', [])
                        if isinstance(attr, str): attr = [attr]
                        if isinstance(attr, list):
                            for a in attr:
                                if a not in agent_scores: agent_scores[a] = 0.0
                                if a not in max_single_agent_scores: max_single_agent_scores[a] = 0.0
                                agent_scores[a] += conf
                                if conf > max_single_agent_scores[a]:
                                    max_single_agent_scores[a] = conf

                        step_val = str(pc.get('mistake_step', probe_step_idx))
                        step_scores[step_val] = step_scores.get(step_val, 0.0) + conf

                        judge_agent = attr[0] if attr else probe_agent
                        judge_votes.append((judge_agent, step_val, conf))
                        valid_resp += 1
                    except: pass

            if valid_resp == 0:
                print("All DAO calls failed. Assuming error.")
                print(f"\nPrediction for {json_file}: Error found.")
                print(f"Agent Name: {probe_agent}")
                print(f"Step Number: {probe_step_idx}")
                error_found = True
                return True

            for a in agent_scores: agent_scores[a] /= valid_resp
            for s in step_scores:  step_scores[s]  /= valid_resp

            avg_val  = max(agent_scores.values()) if agent_scores else 0
            max_vote = max(max_single_agent_scores.values()) if max_single_agent_scores else 0

            if avg_val >= CONVICTION_THRESHOLD or max_vote > 0.85:
                final_culprit = max(agent_scores, key=agent_scores.get) if agent_scores else probe_agent

                culprit_step_votes = {}
                for jg_agent, jg_step, jg_conf in judge_votes:
                    if jg_agent == final_culprit:
                        culprit_step_votes[jg_step] = culprit_step_votes.get(jg_step, 0.0) + jg_conf

                if culprit_step_votes:
                    best_step_str = max(culprit_step_votes, key=culprit_step_votes.get)
                elif step_scores:
                    valid_steps = [str(s) for s, d in enumerate(chat_history) if d.get(index_agent) == final_culprit]
                    f_sc = {k: v for k, v in step_scores.items() if k in valid_steps}
                    if f_sc:
                        max_score = max(f_sc.values())
                        candidates = [k for k, v in f_sc.items() if v == max_score]
                        best_step_str = min(candidates, key=lambda x: int(x) if x.isdigit() else 0)
                    else:
                        best_step_str = max(step_scores, key=step_scores.get)
                else:
                    best_step_str = str(probe_step_idx)

                try:
                    culprit_step = int(best_step_str)
                except ValueError:
                    culprit_step = probe_step_idx

                valid_steps_for_culprit = [str(s) for s, d in enumerate(chat_history) if d.get(index_agent) == final_culprit]
                if valid_steps_for_culprit and str(culprit_step) not in valid_steps_for_culprit:
                    culprit_step = min(int(s) for s in valid_steps_for_culprit)
                    print(f"  [Ghost-step guard] Step corrected to earliest real appearance: {culprit_step}")

                print(f"Error conclusively found at step {culprit_step}. (Avg: {avg_val:.2f}, MaxVote: {max_vote:.2f})")
                print(f"\nPrediction for {json_file}: Error found.")
                print(f"Agent Name: {final_culprit}")
                print(f"Step Number: {culprit_step}")
                print(f"Reason provided by DAO: Convicted by AgentJury-Enhanced Tribunal. (Threshold Matched)")
                error_found = True
                return True
            else:
                print(f"DAO absolved (Avg: {avg_val:.2f}, MaxVote: {max_vote:.2f}).")
                return False

        # ── Phase 2: Dual-trigger probe scan ─────────────────────────────────
        error_found = False
        first_hit = None  # dict: {idx, agent_name, reason, hierarchical_context}

        for idx, entry in enumerate(chat_history):
            agent_name = entry.get(index_agent, 'Unknown Agent')

            if agent_name.lower() == "computer_terminal":
                continue

            current_full_text = entry.get('content', '')
            if idx + 1 < len(chat_history):
                next_agent = chat_history[idx + 1].get(index_agent, 'Unknown')
                if next_agent.lower() == "computer_terminal":
                    current_full_text += f"\n\n[Sandbox Execution Result (Step {idx+1})]:\n{chat_history[idx+1].get('content', '')}"

            dist_past = ""
            for i in range(0, idx):
                if chat_history[i].get(index_agent, '').lower() == "computer_terminal": continue
                dist_past += f"[Step {i}] {chat_history[i].get(index_agent, 'Unknown')}: (JSON SUMMARY) {step_summaries[i]}\n"

            valid_agent_count = 0
            window_end = idx
            while valid_agent_count < 3 and window_end < len(chat_history):
                if chat_history[window_end].get(index_agent, '').lower() != "computer_terminal":
                    valid_agent_count += 1
                window_end += 1

            forward_window = ""
            for i in range(idx, window_end):
                if chat_history[i].get(index_agent, '').lower() == "computer_terminal": continue
                fw_text = chat_history[i].get('content', '')
                if i + 1 < len(chat_history) and chat_history[i+1].get(index_agent, '').lower() == "computer_terminal":
                    fw_text += f"\n[Sandbox Feedback]: {chat_history[i+1].get('content', '')}"
                forward_window += f"[Step {i}] {chat_history[i].get(index_agent, 'Unknown')}: (FULL TEXT)\n{fw_text}\n\n"

            future_down = ""
            for i in range(window_end, len(chat_history)):
                if chat_history[i].get(index_agent, '').lower() == "computer_terminal": continue
                future_down += f"[Step {i}] {chat_history[i].get(index_agent, 'Unknown')}: (JSON SUMMARY) {step_summaries[i]}\n"

            hierarchical_context = ""
            if dist_past: hierarchical_context += f"--- SCANNED PAST (JSON Summaries) ---\n{dist_past}\n"
            hierarchical_context += f"--- HIGH-RESOLUTION WINDOW (Steps {idx} to {min(len(chat_history)-1, idx+2)}) ---\n{forward_window}\n"
            if future_down: hierarchical_context += f"--- DISTANT FUTURE (JSON Summaries) ---\n{future_down}\n"

            prompt = (
                "You are an AI Accuser evaluating a localized multi-agent conversation window to identify the Final Proximal Cause of failure.\n"
                "Note: 'Computer_terminal' is a sandbox environment output.\n"
                f"Problem: {problem}\n\n"
                f"Hierarchical view:\n{hierarchical_context}\n"
                f"--- ROLE-BASED EVALUATION TASK ---\n"
                f"Evaluate Step {idx} by '{agent_name}'. You MUST employ the following Strict Liability logic before deciding:\n"
                f"- OBSERVABLE ERROR THRESHOLD [CRITICAL]: A step should ONLY be marked as 'Root Error: Yes' if there is a concrete, observable manifestation of error in THAT SPECIFIC STEP's output — such as incorrect values propagated, failed assertions reported, hallucinated data introduced, or explicitly executing a catastrophic plan. Merely possessing a verification/planning role without producing a demonstrable error output in this exact step is INSUFFICIENT grounds for conviction.\n"
                f"- ROLE-ALIGNMENT VALIDATION: Did '{agent_name}' passively accept, execute, or propagate erroneous upstream data or plans without raising issues or performing independent verifications? If '{agent_name}' has any capability for reasoning or verification based on their role, unverified compliance with erroneous upstream directives constitutes an active failure. In this case, mark 'Root Error: Yes'.\n"
                f"- STRICT EXECUTOR EXCEPTION: If '{agent_name}' is a strictly passive actuator (e.g., 'Computer_terminal') with absolutely no capacity for reasoning, and it merely returned the execution output of an upstream command without modification, mark 'Root Error: No'.\n"
                f"- UPSTREAM INHERITANCE CHECK [CRITICAL — BACKWARD SCAN]: Before marking 'Root Error: Yes', examine the INPUTS received by Step {idx}. "
                f"If the error was ALREADY PRESENT in the upstream data or plan that '{agent_name}' received, then '{agent_name}' is PROPAGATING — not ORIGINATING — the error. In that case, mark 'Root Error: No'. "
                f"Only mark 'Root Error: Yes' if there is clear evidence that the error was INTRODUCED or MATERIALLY AMPLIFIED by '{agent_name}' at this specific step "
                f"(i.e., the upstream input was correct or neutral, but this step's output was wrong or corrupted). "
                f"A step that 'could have caught' an upstream error but did not is a MISSED VERIFICATION, not a root cause — do NOT mark it 'Root Error: Yes' for this reason alone.\n\n"
                "Respond ONLY in the exact format:\n"
                "1. Root Error: [Yes/No]\n"
                "2. Reason: [Your rigorous deduction based strictly on the agent's role constraints and verification duties]"
            )

            messages = [{"role": "system", "content": "You are a precise step-by-step evaluator."}, {"role": "user", "content": prompt}]
            answer = _tracked_call("ds-v3.2", messages, max_tokens, "probe")

            if not answer: continue
            print(f"Probe Evaluation for Step {idx}: {answer}")

            ans_lower = answer.lower()
            if "yes" not in ans_lower[:20]:
                continue

            reason = answer.split('Reason:', 1)[-1].strip() if 'Reason:' in answer else answer

            if first_hit is None:
                # ── First hit: save and continue scanning ──────────────────
                first_hit = {
                    "idx": idx,
                    "agent_name": agent_name,
                    "reason": reason,
                    "hierarchical_context": hierarchical_context,
                }
                print(f"[AJE] First probe hit at Step {idx} ({agent_name}). Saving context, continuing scan...")
                continue  # do NOT invoke DAO yet

            else:
                # ── Second hit: build dual-dossier and invoke DAO ──────────
                print(f"[AJE] Second probe hit at Step {idx} ({agent_name}). Building dual-flag dossier...")

                dual_flag_context = (
                    f"=== PROBE FLAG #1 — Step {first_hit['idx']} (Agent: '{first_hit['agent_name']}') ===\n"
                    f"{first_hit['hierarchical_context']}\n"
                    f"Probe Reason: {first_hit['reason']}\n\n"
                    f"=== PROBE FLAG #2 — Step {idx} (Agent: '{agent_name}') ===\n"
                    f"{hierarchical_context}\n"
                    f"Probe Reason: {reason}\n"
                )

                base_context = (
                    f"General Task Origin: {problem}\n"
                    f"MAS Agent Roles Info:\n{roles_desc}\n"
                    "Note: 'Computer_terminal' is a sandbox environment output.\n\n"
                    "TWO probe flags have been raised. Your primary task is to determine which flag "
                    "represents the TRUE ROOT CAUSE and which is a downstream MANIFESTATION POINT.\n"
                    "The agents are numbered sequentially (Step 0, Step 1, etc.) corresponding to their turn index.\n\n"
                    "--- DUAL PROBE FLAG DOSSIER ---\n"
                    f"{dual_flag_context}\n"
                    "--- ATTRIBUTION CONSTRAINTS ---\n"
                    f"Flagged steps: Step {first_hit['idx']} and Step {idx}. "
                    "Apply the ROOT CAUSE vs MANIFESTATION POINT doctrine: trace the error backward "
                    "to determine which agent first introduced the corruption. The earlier flag may be "
                    "the true origin or an unrelated noise trigger — reason carefully.\n"
                    "1. [Role-Alignment Validation]: Evaluate each agent's performance against its predefined responsibilities.\n"
                    "2. [Execution Accountability]: Avoid automatically attributing failures to the initial planner.\n"
                    "3. [Proximal Cause Attribution]: Identify the precise step where error was finalized.\n"
                )

                combined_reason = first_hit['reason'] + " " + reason

                convicted = _invoke_dao(base_context, combined_reason, first_hit['idx'], first_hit['agent_name'])
                if convicted:
                    break

        # ── Fallback: only one hit, no second trigger ─────────────────────
        if not error_found and first_hit is not None:
            print(f"[AJE] Only one probe hit found (Step {first_hit['idx']}). Falling back to single-flag DAO...")

            base_context = (
                f"General Task Origin: {problem}\n"
                f"MAS Agent Roles Info:\n{roles_desc}\n"
                "Note: 'Computer_terminal' is a sandbox environment output.\n\n"
                "You have access to hierarchical context showing:\n"
                "Immediate agents: Full details\nNearby agents: Key decisions\nDistant agents: Brief summaries\n\n"
                "The agents are numbered sequentially (Step 0, Step 1, etc.) corresponding to their turn index.\n\n"
                "--- CASE DOSSIER (Hierarchical Log) ---\n"
                f"{first_hit['hierarchical_context']}\n"
                "--- FLAG FOR REVIEW & ATTRIBUTION CONSTRAINTS ---\n"
                f"A frontline scanner has flagged Step {first_hit['idx']} (Agent: '{first_hit['agent_name']}') for review.\n"
                "You are evaluating a LOCALIZED window of action. You must strictly adhere to the following liability doctrines:\n"
                "1. [Role-Alignment Validation]: Evaluate each agent's performance against its predefined responsibilities.\n"
                "2. [Execution Accountability]: Avoid automatically attributing failures to the initial planner.\n"
                "3. [Proximal Cause Attribution]: Identify the precise step where error was finalized.\n"
            )
            _invoke_dao(base_context, first_hit['reason'], first_hit['idx'], first_hit['agent_name'])

        # ── Step 0 Failsafe ───────────────────────────────────────────────
        if not error_found:
            agent_0_name = chat_history[0].get(index_agent, 'Unknown Agent')
            print(f"[AJE] Probe fired zero times. Activating Step 0 Planning Failsafe.")
            print(f"\nPrediction for {json_file}: Error found.")
            print(f"Agent Name: {agent_0_name}")
            print(f"Step Number: 0")
            print(f"Reason provided by DAO: (Failsafe) [AJE] Initial planning failure bypassing all probes.")

        print("\n" + "=" * 50 + "\n")

    # ── Token consumption summary ─────────────────────────────────────────────
    total = sum(token_stats.values())
    print("\n" + "=" * 50)
    print("[AgentJury-Enhanced] Token Consumption Summary")
    print(f"  Phase 1 (Extraction, DS-V3.2): {token_stats['phase1']:>8,} tokens")
    print(f"  Phase 2 (Probe,     DS-R1    ): {token_stats['probe']:>8,} tokens")
    print(f"  Phase 3 (DAO,       3-model  ): {token_stats['dao']:>8,} tokens")
    print(f"  {'TOTAL':30s}: {total:>8,} tokens")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# AgentJury_alg_enhance — Half-Split Probe + Multi-Window DAO (ALG Dataset)
# Architecture mirrors AgentJury_hc_enhance; DAO prompts keep ALG cluster taxonomy
# ─────────────────────────────────────────────────────────────────────────────

def AgentJury_alg_enhance(client, directory_path: str, model: str,
                           max_tokens: int, probe_k: int = 2):
    """
    Enhanced AgentJury for Algorithm-Generated dataset.

    Architecture (mirrors AgentJury_hc_enhance):
    Phase 1: Step summarization with dataset-level cache
    Phase 2: Fixed 2-call half-split probe (Planner-first check, multi-step CoT output)
    Phase 3: Single DAO call (3 experts parallel, multi-window hierarchical context)
    Phase 4: Dense Soft Voting from agent_evaluations only (hallucination guard + ghost-step)

    ALG-specific adaptations:
    - Agent ID: entry["name"] (not "role")
    - Team info: data["system_prompt"] dict (not parsed from initial plan message)
    - computer_terminal entries are passive executors (skipped; output bundled with preceding step)
    - Step indices: position in raw history[] array (0-based, aligned with GT mistake_step)
    - DAO expert prompts: ALG cluster taxonomy (factual / logic / execution)
    """
    import re as _re
    import json as _json
    import concurrent.futures as _cf

    # ── Token 开销估算（字符数 ÷ 4，仅本函数内部使用）────────────────────────────
    def _est(text: str) -> int:
        return len(text) // 4 if text else 0

    def _msgs_in(messages: list) -> int:
        return sum(_est(m.get("content", "")) for m in messages)

    global_tok = {
        "summary_in": 0, "summary_out": 0,
        "probe_in":   0, "probe_out":   0,
        "dao_in":     0, "dao_out":     0,
    }

    WIN_HALF             = 2
    PROBE_CONF_THRESH    = 0.3
    TOP_K_FLAGGED        = 3
    CONVICTION_THRESHOLD = 0.3

    print("\n--- [ALG-Enhance] Starting AgentJury Half-Split Probe + Multi-Window DAO ---\n")
    json_files = _get_sorted_json_files(directory_path)

    # ── Phase 1 summary cache (dataset-level JSON, shared across all files) ──────
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

    # ── Local helpers ──────────────────────────────────────────────────────────────
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

    def _build_dao_context(chat_history_: list, summaries_: list,
                            top_flagged_: list, win_half_: int) -> str:
        high_res = set()
        for (chat_idx, _, _, _, _) in top_flagged_:
            for off in range(-win_half_, win_half_ + 1):
                pos = chat_idx + off
                if 0 <= pos < len(chat_history_):
                    high_res.add(pos)
        lines = []
        in_hr = None
        for i, entry in enumerate(chat_history_):
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
                lines.append(f"[Step {h_idx}] {name_}: (JSON SUMMARY) {summaries_[i]}")
                in_hr = False
        return "\n".join(lines)

    # ── Per-file processing loop ───────────────────────────────────────────────────
    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data      = _load_json_data(file_path)
        if not data:
            continue

        history_raw  = data.get("history", [])
        problem      = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        # ── Build normalized chat_history (skip CT, bundle execution output) ──────
        # history_index = raw position in history_raw → aligned with GT mistake_step
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

        # ── Team info from system_prompt dict ─────────────────────────────────────
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

        print(f"--- [ALG-Enhance] Analyzing: {json_file} ---")
        print(f"    Valid agents: {valid_agents}")
        file_tok = {
            "summary_in": 0, "summary_out": 0,
            "probe_in":   0, "probe_out":   0,
            "dao_in":     0, "dao_out":     0,
        }

        # ════════════════════════════════════════════════════════════════════════════
        # Phase 1: Step summarization with dataset-level cache
        # ════════════════════════════════════════════════════════════════════════════
        if json_file in summaries_cache:
            step_summaries = summaries_cache[json_file]
            if len(step_summaries) != len(chat_history):
                step_summaries = (step_summaries + [""] * len(chat_history))[:len(chat_history)]
            print(f"  [Phase1 Cache HIT] Loaded summaries for {json_file}")
        else:
            print(f"  [Phase1] Generating summaries for {json_file} ...")
            step_summaries = [""] * len(chat_history)

            summary_sys = (
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

                msg = [{"role": "system", "content": summary_sys},
                       {"role": "user",   "content": summary_user}]
                file_tok["summary_in"] += _msgs_in(msg)
                res = _make_api_call_with_retry(client, "ds-v3.2", msg, max_tokens=512, thinking=True)
                step_summaries[s_idx] = res if res else "(Failed)"
                file_tok["summary_out"] += _est(res)

            summaries_cache[json_file] = step_summaries
            try:
                with open(cache_path, "w", encoding="utf-8") as _f:
                    _json.dump(summaries_cache, _f, ensure_ascii=False, indent=2)
                print(f"  [Phase1] Cache saved → {cache_path}")
            except Exception as _e:
                print(f"  [Phase1] Warning: cache save failed: {_e}")

        # ════════════════════════════════════════════════════════════════════════════
        # Phase 2: k-partition concurrent probe
        # ════════════════════════════════════════════════════════════════════════════
        import math as _math

        N        = len(chat_history)
        actual_k = min(probe_k, N)
        part_sz  = _math.ceil(N / actual_k)

        # 每个分片存 (chat_history 切片起始索引, 结束索引)
        partition_ranges = []
        for _p in range(actual_k):
            _s = _p * part_sz
            _e = min(_s + part_sz, N)
            if _s < N:
                partition_ranges.append((_s, _e))

        probe_rules = (
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

        def _run_one_probe(p_idx: int, p_start: int, p_end: int):
            """构建分片 prompt 并调用 LLM，返回 (raw_response, input_token_count)。"""
            part_entries   = chat_history[p_start:p_end]
            part_summaries = step_summaries[p_start:p_end]
            past_entries   = chat_history[:p_start]
            past_summaries = step_summaries[:p_start]
            fut_entries    = chat_history[p_end:]
            fut_summaries  = step_summaries[p_end:]

            part_hist_start = part_entries[0]["history_index"]
            part_hist_end   = part_entries[-1]["history_index"]

            part_full_text = _build_half_text(part_entries)

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

            sections.append(probe_rules)
            probe_user = "".join(sections)

            msg = [{"role": "system", "content": probe_sys},
                   {"role": "user",   "content": probe_user}]
            _in_tok = _msgs_in(msg)
            raw = _make_api_call_with_retry(client, "kimi-2.5", msg, max_tokens=2048, thinking=False)
            return raw, _in_tok

        # ── k 次并发调用 ──────────────────────────────────────────────────────────
        print(f"  [Phase2] k={len(partition_ranges)} partition probe (concurrent) ...")
        all_raw_flags = []
        with _cf.ThreadPoolExecutor(max_workers=len(partition_ranges)) as _exec:
            _fut_map = {
                _exec.submit(_run_one_probe, _pi, _ps, _pe): _pi
                for _pi, (_ps, _pe) in enumerate(partition_ranges)
            }
            for _fut in _cf.as_completed(_fut_map):
                _pi = _fut_map[_fut]
                _raw, _in_tok = _fut.result()
                file_tok["probe_in"]  += _in_tok
                file_tok["probe_out"] += _est(_raw)
                _flags = _parse_probe_out(_raw)
                print(f"    Partition {_pi + 1}: flagged "
                      f"{[(_f['step'], _f['confidence']) for _f in _flags]}")
                all_raw_flags.extend(_flags)

        # ── Dedup merge: 同一步骤保留最高置信度 ───────────────────────────────────
        all_flagged: dict = {}
        for flag in all_raw_flags:
            s = flag["step"]
            if s < 0:
                continue
            if s not in all_flagged or flag["confidence"] > all_flagged[s]["confidence"]:
                all_flagged[s] = flag

        qualified = [f for f in all_flagged.values() if f["confidence"] >= PROBE_CONF_THRESH]
        qualified.sort(key=lambda x: x["confidence"], reverse=True)
        top_flags = qualified[:TOP_K_FLAGGED]

        print(f"  [Phase2] Merged flagged steps (conf ≥ {PROBE_CONF_THRESH}): "
              f"{[(f['step'], f['confidence']) for f in top_flags]}")

        if not top_flags:
            first_exec_agent = chat_history[0]["normalized_name"]
            first_exec_step  = chat_history[0]["history_index"]
            print(f"  [Phase2] No suspicious steps. Activating Step-0 Failsafe.")
            print(f"\nPrediction for {json_file}: Error found.")
            print(f"Agent Name: {first_exec_agent}")
            print(f"Step Number: {first_exec_step}")
            print(f"Reason: (ALG-Enhance Failsafe) Probe found no suspicious steps.")
            continue

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

        if not top_flagged_tuples:
            first_exec_agent = chat_history[0]["normalized_name"]
            first_exec_step  = chat_history[0]["history_index"]
            print(f"  [Phase2] All flagged steps unmappable. Activating Step-0 Failsafe.")
            print(f"\nPrediction for {json_file}: Error found.")
            print(f"Agent Name: {first_exec_agent}")
            print(f"Step Number: {first_exec_step}")
            print(f"Reason: (ALG-Enhance Failsafe) Flagged steps unmappable.")
            continue

        # ════════════════════════════════════════════════════════════════════════════
        # Phase 3: Multi-window DAO — 3 experts parallel
        # ════════════════════════════════════════════════════════════════════════════
        merged_context = _build_dao_context(
            chat_history, step_summaries, top_flagged_tuples, WIN_HALF
        )

        anchor        = top_flagged_tuples[0]
        anchor_hist   = anchor[1]
        anchor_agent  = anchor[2]
        anchor_reason = anchor[4]

        base_context = (
            "[BACKGROUND] Post-mortem investigation of a FAILED multi-agent task. "
            "The agents below did NOT produce the correct final answer.\n\n"
            f"General Task Origin: {problem}\nGround Truth Expected Outcome: {ground_truth}\n"
            f"Active Agents in This Case:\n{roles_desc}\n"
            "SCOPE CONSTRAINT: Only evaluate agents listed above. "
            "Do NOT generate evaluations for any agent not present in the log.\n\n"
            "Note: 'Computer_terminal' is a sandbox environment — it is a passive executor with no reasoning capacity.\n\n"
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
            "executed upstream errors without performing the verifications its role requires, "
            "liability transfers to that downstream agent.\n"
        )

        task_and_format_instruction = (
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

        # ── ALG DAO expert system prompts (empirical cluster taxonomy) ─────────────
        dao_sys_base = (
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

        focus_factual = (
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
        focus_logic = (
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
        focus_execution = (
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

        sys_1 = dao_sys_base.format(focus=focus_factual)
        sys_2 = dao_sys_base.format(focus=focus_logic)
        sys_3 = dao_sys_base.format(focus=focus_execution)


        dao_1 = (base_context
                 + "\n[YOUR DIMENSION] FACTUAL & INFORMATION RETRIEVAL Expert.\n\n"
                 + task_and_format_instruction)
        dao_2 = (base_context
                 + "\n[YOUR DIMENSION] LOGIC, REASONING & PLANNING Expert.\n\n"
                 + task_and_format_instruction)
        dao_3 = (base_context
                 + "\n[YOUR DIMENSION] AGENT EXECUTION & COORDINATION Expert.\n\n"
                 + task_and_format_instruction)

        print(f"  [Phase3] Launching 3 DAO experts (parallel) ...")

        # 统计 Phase 3 输入 token（调用前）
        for _s, _u in [(sys_1, dao_1), (sys_2, dao_2), (sys_3, dao_3)]:
            file_tok["dao_in"] += _msgs_in([{"role": "system", "content": _s},
                                             {"role": "user",   "content": _u}])

        def _call_dao_alg(args):
            i, sys_p, usr_p = args
            msg = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
            if i == 0: return _make_api_DAO_1_call(client, model, msg, max_tokens)
            if i == 1: return _make_api_DAO_2_call(client, model, msg, max_tokens)
            return _make_api_DAO_3_call(client, model, msg, max_tokens)

        with _cf.ThreadPoolExecutor(max_workers=3) as executor:
            futs    = {executor.submit(_call_dao_alg, (i, s, u)): i
                       for i, (s, u) in enumerate([(sys_1, dao_1), (sys_2, dao_2), (sys_3, dao_3)])}
            results = [None] * 3
            for f in _cf.as_completed(futs):
                results[futs[f]] = f.result()

        # 统计 Phase 3 输出 token
        for r in results:
            file_tok["dao_out"] += _est(r)

        # ════════════════════════════════════════════════════════════════════════════
        # Phase 4: Dense Soft Voting (from agent_evaluations only)
        # ════════════════════════════════════════════════════════════════════════════
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

                # Ghost-step protection
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
            first_exec_agent = chat_history[0]["normalized_name"]
            first_exec_step  = chat_history[0]["history_index"]
            print(f"\nPrediction for {json_file}: Error found (Failsafe).")
            print(f"Agent Name: {first_exec_agent}")
            print(f"Step Number: {first_exec_step}")
            print(f"Reason: (ALG-Enhance Failsafe) No confident conviction reached.")

        # ── Per-file token 开销报告 ─────────────────────────────────────────────────
        file_total = sum(file_tok.values())
        print(f"\n[Token Est.] {json_file}")
        print(f"  Phase1 Summary : in={file_tok['summary_in']:,}  out={file_tok['summary_out']:,}")
        print(f"  Phase2 Probe   : in={file_tok['probe_in']:,}  out={file_tok['probe_out']:,}")
        print(f"  Phase3 DAO     : in={file_tok['dao_in']:,}  out={file_tok['dao_out']:,}")
        print(f"  File Total (est): {file_total:,}")
        for _k in global_tok:
            global_tok[_k] += file_tok[_k]

    # ── 全局 token 汇总 ────────────────────────────────────────────────────────────
    grand_total = sum(global_tok.values())
    print("\n" + "=" * 50)
    print("[ALG-Enhance] === Global Token Cost Summary (Est.) ===")
    print(f"  Phase1 Summary : in={global_tok['summary_in']:,}  out={global_tok['summary_out']:,}")
    print(f"  Phase2 Probe   : in={global_tok['probe_in']:,}  out={global_tok['probe_out']:,}")
    print(f"  Phase3 DAO     : in={global_tok['dao_in']:,}  out={global_tok['dao_out']:,}")
    print(f"  Grand Total (est): {grand_total:,} tokens")
    print("=" * 50)


def AgentJury_alg_enhance_noGT(client, directory_path: str, model: str,
                           max_tokens: int, probe_k: int = 4):
    """
    Enhanced AgentJury for Algorithm-Generated dataset.

    Architecture (mirrors AgentJury_hc_enhance):
    Phase 1: Step summarization with dataset-level cache
    Phase 2: Fixed 2-call half-split probe (Planner-first check, multi-step CoT output)
    Phase 3: Single DAO call (3 experts parallel, multi-window hierarchical context)
    Phase 4: Dense Soft Voting from agent_evaluations only (hallucination guard + ghost-step)

    ALG-specific adaptations:
    - Agent ID: entry["name"] (not "role")
    - Team info: data["system_prompt"] dict (not parsed from initial plan message)
    - computer_terminal entries are passive executors (skipped; output bundled with preceding step)
    - Step indices: position in raw history[] array (0-based, aligned with GT mistake_step)
    - DAO expert prompts: ALG cluster taxonomy (factual / logic / execution)
    """
    import re as _re
    import json as _json
    import concurrent.futures as _cf

    # ── Token 开销估算（字符数 ÷ 4，仅本函数内部使用）────────────────────────────
    def _est(text: str) -> int:
        return len(text) // 4 if text else 0

    def _msgs_in(messages: list) -> int:
        return sum(_est(m.get("content", "")) for m in messages)

    global_tok = {
        "summary_in": 0, "summary_out": 0,
        "probe_in":   0, "probe_out":   0,
        "dao_in":     0, "dao_out":     0,
    }

    WIN_HALF             = 2
    PROBE_CONF_THRESH    = 0.3
    TOP_K_FLAGGED        = 3
    CONVICTION_THRESHOLD = 0.3

    print("\n--- [ALG-Enhance-noGT] Starting AgentJury Half-Split Probe + Multi-Window DAO ---\n")
    json_files = _get_sorted_json_files(directory_path)

    # ── Phase 1 summary cache (dataset-level JSON, shared across all files) ──────
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

    # ── Local helpers ──────────────────────────────────────────────────────────────
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

    def _build_dao_context(chat_history_: list, summaries_: list,
                            top_flagged_: list, win_half_: int) -> str:
        high_res = set()
        for (chat_idx, _, _, _, _) in top_flagged_:
            for off in range(-win_half_, win_half_ + 1):
                pos = chat_idx + off
                if 0 <= pos < len(chat_history_):
                    high_res.add(pos)
        lines = []
        in_hr = None
        for i, entry in enumerate(chat_history_):
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
                lines.append(f"[Step {h_idx}] {name_}: (JSON SUMMARY) {summaries_[i]}")
                in_hr = False
        return "\n".join(lines)

    # ── Per-file processing loop ───────────────────────────────────────────────────
    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data      = _load_json_data(file_path)
        if not data:
            continue

        history_raw  = data.get("history", [])
        problem      = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        # ── Build normalized chat_history (skip CT, bundle execution output) ──────
        # history_index = raw position in history_raw → aligned with GT mistake_step
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

        # ── Team info from system_prompt dict ─────────────────────────────────────
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

        print(f"--- [ALG-Enhance] Analyzing: {json_file} ---")
        print(f"    Valid agents: {valid_agents}")
        file_tok = {
            "summary_in": 0, "summary_out": 0,
            "probe_in":   0, "probe_out":   0,
            "dao_in":     0, "dao_out":     0,
        }

        # ════════════════════════════════════════════════════════════════════════════
        # Phase 1: Step summarization with dataset-level cache
        # ════════════════════════════════════════════════════════════════════════════
        if json_file in summaries_cache:
            step_summaries = summaries_cache[json_file]
            if len(step_summaries) != len(chat_history):
                step_summaries = (step_summaries + [""] * len(chat_history))[:len(chat_history)]
            print(f"  [Phase1 Cache HIT] Loaded summaries for {json_file}")
        else:
            print(f"  [Phase1] Generating summaries for {json_file} ...")
            step_summaries = [""] * len(chat_history)

            summary_sys = (
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

                msg = [{"role": "system", "content": summary_sys},
                       {"role": "user",   "content": summary_user}]
                file_tok["summary_in"] += _msgs_in(msg)
                res = _make_api_call_with_retry(client, "ds-v3.2", msg, max_tokens=512, thinking=True)
                step_summaries[s_idx] = res if res else "(Failed)"
                file_tok["summary_out"] += _est(res)

            summaries_cache[json_file] = step_summaries
            try:
                with open(cache_path, "w", encoding="utf-8") as _f:
                    _json.dump(summaries_cache, _f, ensure_ascii=False, indent=2)
                print(f"  [Phase1] Cache saved → {cache_path}")
            except Exception as _e:
                print(f"  [Phase1] Warning: cache save failed: {_e}")

        # ════════════════════════════════════════════════════════════════════════════
        # Phase 2: k-partition concurrent probe
        # ════════════════════════════════════════════════════════════════════════════
        import math as _math

        N        = len(chat_history)
        actual_k = min(probe_k, N)
        part_sz  = _math.ceil(N / actual_k)

        # 每个分片存 (chat_history 切片起始索引, 结束索引)
        partition_ranges = []
        for _p in range(actual_k):
            _s = _p * part_sz
            _e = min(_s + part_sz, N)
            if _s < N:
                partition_ranges.append((_s, _e))

        probe_rules = (
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

        def _run_one_probe(p_idx: int, p_start: int, p_end: int):
            """构建分片 prompt 并调用 LLM，返回 (raw_response, input_token_count)。"""
            part_entries   = chat_history[p_start:p_end]
            part_summaries = step_summaries[p_start:p_end]
            past_entries   = chat_history[:p_start]
            past_summaries = step_summaries[:p_start]
            fut_entries    = chat_history[p_end:]
            fut_summaries  = step_summaries[p_end:]

            part_hist_start = part_entries[0]["history_index"]
            part_hist_end   = part_entries[-1]["history_index"]

            part_full_text = _build_half_text(part_entries)

            sections = [
                f"Task: {problem}\n"
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

            sections.append(probe_rules)
            probe_user = "".join(sections)

            msg = [{"role": "system", "content": probe_sys},
                   {"role": "user",   "content": probe_user}]
            _in_tok = _msgs_in(msg)
            raw = _make_api_call_with_retry(client, "kimi-2.5", msg, max_tokens=2048, thinking=False)
            return raw, _in_tok

        # ── k 次并发调用 ──────────────────────────────────────────────────────────
        print(f"  [Phase2] k={len(partition_ranges)} partition probe (concurrent) ...")
        all_raw_flags = []
        with _cf.ThreadPoolExecutor(max_workers=len(partition_ranges)) as _exec:
            _fut_map = {
                _exec.submit(_run_one_probe, _pi, _ps, _pe): _pi
                for _pi, (_ps, _pe) in enumerate(partition_ranges)
            }
            for _fut in _cf.as_completed(_fut_map):
                _pi = _fut_map[_fut]
                _raw, _in_tok = _fut.result()
                file_tok["probe_in"]  += _in_tok
                file_tok["probe_out"] += _est(_raw)
                _flags = _parse_probe_out(_raw)
                print(f"    Partition {_pi + 1}: flagged "
                      f"{[(_f['step'], _f['confidence']) for _f in _flags]}")
                all_raw_flags.extend(_flags)

        # ── Dedup merge: 同一步骤保留最高置信度 ───────────────────────────────────
        all_flagged: dict = {}
        for flag in all_raw_flags:
            s = flag["step"]
            if s < 0:
                continue
            if s not in all_flagged or flag["confidence"] > all_flagged[s]["confidence"]:
                all_flagged[s] = flag

        qualified = [f for f in all_flagged.values() if f["confidence"] >= PROBE_CONF_THRESH]
        qualified.sort(key=lambda x: x["confidence"], reverse=True)
        top_flags = qualified[:TOP_K_FLAGGED]

        print(f"  [Phase2] Merged flagged steps (conf ≥ {PROBE_CONF_THRESH}): "
              f"{[(f['step'], f['confidence']) for f in top_flags]}")

        if not top_flags:
            first_exec_agent = chat_history[0]["normalized_name"]
            first_exec_step  = chat_history[0]["history_index"]
            print(f"  [Phase2] No suspicious steps. Activating Step-0 Failsafe.")
            print(f"\nPrediction for {json_file}: Error found.")
            print(f"Agent Name: {first_exec_agent}")
            print(f"Step Number: {first_exec_step}")
            print(f"Reason: (ALG-Enhance Failsafe) Probe found no suspicious steps.")
            continue

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

        if not top_flagged_tuples:
            first_exec_agent = chat_history[0]["normalized_name"]
            first_exec_step  = chat_history[0]["history_index"]
            print(f"  [Phase2] All flagged steps unmappable. Activating Step-0 Failsafe.")
            print(f"\nPrediction for {json_file}: Error found.")
            print(f"Agent Name: {first_exec_agent}")
            print(f"Step Number: {first_exec_step}")
            print(f"Reason: (ALG-Enhance Failsafe) Flagged steps unmappable.")
            continue

        # ════════════════════════════════════════════════════════════════════════════
        # Phase 3: Multi-window DAO — 3 experts parallel
        # ════════════════════════════════════════════════════════════════════════════
        merged_context = _build_dao_context(
            chat_history, step_summaries, top_flagged_tuples, WIN_HALF
        )

        anchor        = top_flagged_tuples[0]
        anchor_hist   = anchor[1]
        anchor_agent  = anchor[2]
        anchor_reason = anchor[4]

        base_context = (
            "[BACKGROUND] Post-mortem investigation of a FAILED multi-agent task. "
            "The agents below did NOT produce the correct final answer.\n\n"
            f"General Task Origin: {problem}\nGround Truth Expected Outcome: {ground_truth}\n"
            f"Active Agents in This Case:\n{roles_desc}\n"
            "SCOPE CONSTRAINT: Only evaluate agents listed above. "
            "Do NOT generate evaluations for any agent not present in the log.\n\n"
            "Note: 'Computer_terminal' is a sandbox environment — it is a passive executor with no reasoning capacity.\n\n"
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
            "executed upstream errors without performing the verifications its role requires, "
            "liability transfers to that downstream agent.\n"
        )

        task_and_format_instruction = (
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

        # ── ALG DAO expert system prompts (empirical cluster taxonomy) ─────────────
        dao_sys_base = (
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

        focus_factual = (
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
        focus_logic = (
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
        focus_execution = (
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

        sys_1 = dao_sys_base.format(focus=focus_factual)
        sys_2 = dao_sys_base.format(focus=focus_logic)
        sys_3 = dao_sys_base.format(focus=focus_execution)


        dao_1 = (base_context
                 + "\n[YOUR DIMENSION] FACTUAL & INFORMATION RETRIEVAL Expert.\n\n"
                 + task_and_format_instruction)
        dao_2 = (base_context
                 + "\n[YOUR DIMENSION] LOGIC, REASONING & PLANNING Expert.\n\n"
                 + task_and_format_instruction)
        dao_3 = (base_context
                 + "\n[YOUR DIMENSION] AGENT EXECUTION & COORDINATION Expert.\n\n"
                 + task_and_format_instruction)

        print(f"  [Phase3] Launching 3 DAO experts (parallel) ...")

        # 统计 Phase 3 输入 token（调用前）
        for _s, _u in [(sys_1, dao_1), (sys_2, dao_2), (sys_3, dao_3)]:
            file_tok["dao_in"] += _msgs_in([{"role": "system", "content": _s},
                                             {"role": "user",   "content": _u}])

        def _call_dao_alg(args):
            i, sys_p, usr_p = args
            msg = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
            if i == 0: return _make_api_DAO_1_call(client, model, msg, max_tokens)
            if i == 1: return _make_api_DAO_2_call(client, model, msg, max_tokens)
            return _make_api_DAO_3_call(client, model, msg, max_tokens)

        with _cf.ThreadPoolExecutor(max_workers=3) as executor:
            futs    = {executor.submit(_call_dao_alg, (i, s, u)): i
                       for i, (s, u) in enumerate([(sys_1, dao_1), (sys_2, dao_2), (sys_3, dao_3)])}
            results = [None] * 3
            for f in _cf.as_completed(futs):
                results[futs[f]] = f.result()

        # 统计 Phase 3 输出 token
        for r in results:
            file_tok["dao_out"] += _est(r)

        # ════════════════════════════════════════════════════════════════════════════
        # Phase 4: Dense Soft Voting (from agent_evaluations only)
        # ════════════════════════════════════════════════════════════════════════════
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

                # Ghost-step protection
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
            first_exec_agent = chat_history[0]["normalized_name"]
            first_exec_step  = chat_history[0]["history_index"]
            print(f"\nPrediction for {json_file}: Error found (Failsafe).")
            print(f"Agent Name: {first_exec_agent}")
            print(f"Step Number: {first_exec_step}")
            print(f"Reason: (ALG-Enhance Failsafe) No confident conviction reached.")

        # ── Per-file token 开销报告 ─────────────────────────────────────────────────
        file_total = sum(file_tok.values())
        print(f"\n[Token Est.] {json_file}")
        print(f"  Phase1 Summary : in={file_tok['summary_in']:,}  out={file_tok['summary_out']:,}")
        print(f"  Phase2 Probe   : in={file_tok['probe_in']:,}  out={file_tok['probe_out']:,}")
        print(f"  Phase3 DAO     : in={file_tok['dao_in']:,}  out={file_tok['dao_out']:,}")
        print(f"  File Total (est): {file_total:,}")
        for _k in global_tok:
            global_tok[_k] += file_tok[_k]

    # ── 全局 token 汇总 ────────────────────────────────────────────────────────────
    grand_total = sum(global_tok.values())
    print("\n" + "=" * 50)
    print("[ALG-Enhance] === Global Token Cost Summary (Est.) ===")
    print(f"  Phase1 Summary : in={global_tok['summary_in']:,}  out={global_tok['summary_out']:,}")
    print(f"  Phase2 Probe   : in={global_tok['probe_in']:,}  out={global_tok['probe_out']:,}")
    print(f"  Phase3 DAO     : in={global_tok['dao_in']:,}  out={global_tok['dao_out']:,}")
    print(f"  Grand Total (est): {grand_total:,} tokens")
    print("=" * 50)
