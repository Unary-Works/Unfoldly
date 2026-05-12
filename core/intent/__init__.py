"""
Multi-agent intent routing package (v3 micro-agent architecture).

Layer -1: QueryPreprocessor    — greeting / capability / translate / zero-scope intercepts
Layer 0 (Experts):
  - SelectionExpert      — "selected/chosen files" binary classifier
  - ContextFollowupExpert — post-count/post-content follow-up detection
  - CountExpert          — "how many files" detection
  - FilenameExpert       — bare filename → search
  - EntitySearchExpert   — bare entity (≤3 words) → search
  - CategoryListExpert   — "find + file-type" → count(category)
Layer 1: ConversationRouter  — lightweight LLM call, decides: continuation | file_op | chat
Layer 2A: ContinuationAgent  — handles translate_response | process_previous | chat
Layer 2B: FileOpAgent        — handles search | count | summarize | summarize_all
Layer 3: IntentValidator     — centralized correction chain
Layer 4: QueryNormalizer     — output normalization pipeline
"""
