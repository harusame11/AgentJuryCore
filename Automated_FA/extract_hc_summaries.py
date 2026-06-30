import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from Lib.api_utils import (
    _load_json_data, 
    _get_sorted_json_files, 
    _make_api_DAO_3_call,
    API_MODEL_MAP
)
from Lib.api_utils_4_handcrafted import (
    _normalize_agent_name,
    _parse_team_from_initial_plan,
    _get_effective_agents_hc
)

def extract_summaries(directory_path, model_alias, max_tokens=512):
    """
    Standalone script to perform Phase 1 structured compression (ECHO summaries)
    on Hand-Crafted datasets.
    """
    api_key = os.environ.get("SILICON_API_KEY", "")
    if not api_key:
        raise ValueError("Missing API key. Set SILICON_API_KEY before running this script.")

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("SILICON_BASE_URL", "https://api.siliconflow.cn/v1"),
    )
    
    model = API_MODEL_MAP.get(model_alias, model_alias)
    
    print(f"\n--- [Standalone] Starting ECHO Summary Extraction ---")
    print(f"Directory: {directory_path}")
    print(f"Model: {model_alias} ({model})")
    
    json_files = _get_sorted_json_files(directory_path)
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

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

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data: continue

        history_raw = data.get("history", [])
        chat_history = []
        for h_idx, entry in enumerate(history_raw):
            raw_role = entry.get("role", "")
            normalized = _normalize_agent_name(raw_role)
            if normalized.lower() == "human": continue
            chat_history.append({
                "content": entry.get("content", ""),
                "normalized_name": normalized,
                "history_index": h_idx,
            })

        step_summaries = [""] * len(chat_history)
        
        for s_idx in range(len(chat_history)):
            s_agent = chat_history[s_idx]["normalized_name"]
            s_content = chat_history[s_idx]["content"]

            past_summaries_context = ""
            for i in range(max(0, s_idx - 2), s_idx):
                if step_summaries[i] and "(Failed)" not in step_summaries[i]:
                    past_summaries_context += f"[Step {chat_history[i]['history_index']}] {chat_history[i]['normalized_name']}: {step_summaries[i]}\n"

            summary_user = ""
            if past_summaries_context:
                summary_user += f"--- Previous Steps Context ---\n{past_summaries_context}\n\n"
            
            s_content_truncated = s_content[:3000] + ("...[TRUNCATED]" if len(s_content) > 3000 else "")
            summary_user += f"--- Current Step to Extract ---\nAgent: {s_agent}\nContent: {s_content_truncated}"

            msg = [{"role": "system", "content": summary_sys}, {"role": "user", "content": summary_user}]
            res = _make_api_DAO_3_call(client, model, msg, max_tokens)
            step_summaries[s_idx] = res if res else "(Failed)"

        # Save to JSON
        file_idx = os.path.splitext(json_file)[0]
        abstract_filename = f"structured_abstract_hc_{file_idx}.json"
        abstract_path = os.path.join(output_dir, abstract_filename)

        abstract_list = []
        for s_idx, summary in enumerate(step_summaries):
            abstract_list.append({
                "step_index": chat_history[s_idx]["history_index"],
                "agent": chat_history[s_idx]["normalized_name"],
                "summary": summary
            })

        with open(abstract_path, "w", encoding="utf-8") as f:
            json.dump(abstract_list, f, ensure_ascii=False, indent=2)

    print(f"\n--- [Standalone] Finished! Summaries saved in '{output_dir}/' ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structured summaries from Hand-Crafted dataset logs.")
    parser.add_argument("--directory_path", type=str, required=True, help="Path to the JSON log directory.")
    parser.add_argument("--model", type=str, default="ds-r1", help="Model alias (e.g., ds-r1).")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens for summary.")

    args = parser.parse_args()
    extract_summaries(args.directory_path, args.model, args.max_tokens)
