from typing import Optional

# ==================== English Prompt Variants ====================

SUMMARIZE_VIDEO_FILE_PROMPT_EN = """Based on the video metadata and keyframe descriptions below, describe the content of this video to the user.

[IMPORTANT] Do NOT wrap your summary in a Markdown code block.
[IMPORTANT] Do NOT say "I cannot watch the video" — you already have keyframe descriptions, answer directly based on them.

[Video metadata]
{metadata}

[Keyframe descriptions (chronological)]
{keyframes}

[Output format]
## 🎬 Video Overview
2-3 sentences summarizing the main content

## 🔑 Key Frames
List important information from each keyframe in chronological order

## 📝 Summary
Briefly state the purpose or value of this video

---
**Do you want me to open the original file?**"""

CLASSIFY_PROMPT_EN = """
Analyze the following file and assign the single best category
(e.g. Resume, Report, Contract, Note, Manual, Paper, Presentation, Data, Email, Image, Book, Code, Invoice, etc.).

<Instructions>
- Prefer common categories: Resume, Report, Contract, Note, Manual, Paper, Presentation, Data, Email, Image, Book.
- If none fits, create a short custom category name.
- Output category name only, with no extra text.
- Output "Other" only when the content is unreadable.
- Distinguish Paper vs Book carefully:
  * Paper: usually shorter (for example PDF pages <= 40) and contains academic structure such as Abstract, Keywords, References, DOI, arXiv, methodology, experiments, conclusion.
  * Book: usually longer (for example PDF pages >= 80) and contains book structure such as ISBN, publisher, copyright page, table of contents, chapters, preface.
- README / quickstart / install documents are usually Manual or Note, not Book.
</Instructions>

File info:
- Name: {file_name}
- Extension: {file_ext}
- PDF pages: {page_count}

File content (first 2000 chars):
{content}

Category:
"""

CLASSIFY_TAXONOMY_PROMPT_EN = """
You are building document taxonomy metadata for a local file retrieval system. Output one JSON object describing the file's stable family, dynamic leaf_category, and semantic role.

<Instructions>
- Split the result into 3 layers:
  1. family: a stable retrieval family. It MUST be one of:
     ["resume", "report", "contract", "note", "manual", "paper", "presentation", "data", "email", "image", "audio", "video", "book", "code", "invoice", "quotation", "document", "other"]
  2. leaf_category: a finer reusable leaf label. It may be freely created and should stay short and stable.
     - Prefer natural type labels such as "paper_summary", "application_form", "architecture_plan", "datasheet", "faq", "meeting_notes"
     - Do not just repeat family unless no finer stable label is available
  3. role: the file's role in the knowledge system. It MUST be one of:
     ["primary_source", "summary", "explainer", "analysis", "transcript", "ocr_result", "generated_doc", "reference", "other"]
- Important distinctions:
  - The original paper / original contract / original resume / original form / original manual => role should usually be primary_source
  - A document explaining, summarizing, or analyzing another document => summary / explainer / analysis
  - OCR extraction results => ocr_result
  - Chat transcripts, AI-generated writeups, or transcript-derived notes => transcript or generated_doc
- family is for stable filtering; leaf_category is where taxonomy can grow dynamically.
- confidence should be a float between 0 and 1.
- Output JSON only, with no extra text.
</Instructions>

File info:
- Name: {file_name}
- Extension: {file_ext}
- PDF pages: {page_count}

File content (first 2000 chars):
{content}

Output format:
{{"family":"report","leaf_category":"industry_report","role":"primary_source","confidence":0.91}}
"""

SUMMARY_PROMPT_INDEXING = """You are a bilingual document summarizer for a semantic file retrieval system that serves English-speaking users.

<Rules>
1. Output EXACTLY ONE paragraph. Write primarily in English.
2. PRESERVE Chinese proper nouns: person names, company names, product names, place names — keep the ORIGINAL Chinese characters AND add Pinyin transliteration in parentheses. However, if the document explicitly mentions an official English name, use that English name instead of Pinyin. DO NOT translate names literally into English words. NEVER invent English names. Also, if the entity is globally well-known, use its established English name. Format as: `Original Name (Official English / Pinyin)`.
   Examples:
   - "<Chinese person name> (<English name or Pinyin>), role at <Chinese company name> (<English company name>)"
   - "<Chinese document title> (<concise English retrieval title>)"
   - "<Chinese market/product phrase> (<English retrieval gloss>)"
3. RETRIEVAL PRIORITY: Ask yourself "What English keywords would an English-speaking user type to find this file?"
   Your summary MUST naturally embed those keywords AND the original Chinese proper nouns.
4. Summary format by document type:
   - Resume/CV: "[Full Name Chinese+English] - [Job Title] at [Company Chinese+English], [X years of experience], [key skills/achievements]"
     * MUST include years of experience if mentioned (e.g. "10 years", "5年经验").
     * MUST include 1-2 quantified achievements if present (e.g. "40% YoY growth", "led team of 20").
   - Report/Analysis: "[Topic English] [report/analysis] (Chinese title preserved) - key findings: [X, Y, Z]"
     * MUST extract and include quantitative metrics (numbers, %, growth rates, amounts, rankings) if present.
   - Invoice/Receipt: "[Seller name] invoice [number], total amount: [¥/$/€ X.XX], date [Y], buyer: [Z]"
      * CRITICAL: You MUST extract the total amount/金额/合计 from invoices. Look for fields like 价税合计, 合计金额, Total, Amount, Grand Total, 小写, ¥, $, etc.
      * If multiple amounts exist, use the LARGEST one (usually 价税合计 or Grand Total).
      * If no amount is found, write "amount: not found in document".
   - Audio/Media: "[Type] of [English description], original name: [Chinese name if any], format: [EXT]"
   - Technical doc/Manual: "[Product/system name] [manual/guide] - covers [topics]"
   - Other: core topic in English + Chinese title preserved + 2-3 key retrievable facts
5. If the document content below is empty, unreadable, or meaningless, output exactly: "Empty or unreadable document."
6. Prefix your output with "Summary: " and nothing else before it.
</Rules>

Document content (first 1500 chars):
{content}

Summary: """

SUMMARY_AND_EXTRACT_PROMPT = """You are a bilingual document summarizer and an expert information extractor.

<Rules>
1. Output EXACTLY ONE paragraph for the summary. Write primarily in English.
2. PRESERVE Chinese proper nouns: person names, company names, product names, place names — keep the ORIGINAL Chinese characters AND add Pinyin transliteration in parentheses. However, if the document explicitly mentions an official English name, use that English name instead of Pinyin. DO NOT translate names literally into English words. NEVER invent English names. Also, if the entity is globally well-known, use its established English name. Format as: `Original Name (Official English / Pinyin)`.
3. RETRIEVAL PRIORITY: Ask yourself "What English keywords would an English-speaking user type to find this file?" Your summary MUST naturally embed those keywords AND the original Chinese proper nouns.
4. Summary format by document type:
   - Resume/CV: "[Full Name Chinese+English] - [Job Title] at [Company Chinese+English], [X years of experience], [key skills/achievements]"
   - Report/Analysis: "[Topic English] [report/analysis] (Chinese title preserved) - key findings: [X, Y, Z]"
   - Invoice/Receipt: "[Seller name] invoice [number], total amount: [¥/$/€ X.XX], date [Y], buyer: [Z]"
      * CRITICAL: You MUST extract the total amount/金额/合计 from invoices. Look for 价税合计, 合计金额, Total, Amount, Grand Total, 小写, ¥, $, etc.
      * If multiple amounts exist, use the LARGEST one (usually 价税合计 or Grand Total).
      * If no amount is found, write "amount: not found in document".
   - Audio/Media: "[Type] of [English description], original name: [Chinese name if any], format: [EXT]"
   - Technical doc/Manual: "[Product/system name] [manual/guide] - covers [topics]"
   - Other: core topic in English + Chinese title preserved + 2-3 key retrievable facts
5. If the document content below is empty, unreadable, or meaningless, summary should be exactly: "Empty or unreadable document."
6. EXACT JSON OUTPUT REQUIRED. Do NOT wrap in markdown block (e.g. ```json). Your response must be purely a JSON object.
7. If the caller explicitly requests filename translation, you MUST output a concise English filename translation in the "file_name_en" field.
   - The value must be a short label only, not a sentence.
   - Translate the filename LITERALLY from the filename text itself. Do NOT infer a better title from the document content.
   - Preserve filename distinctions such as version markers, interview/integrated labels, and business-card/profile style wording when present.
   - If no filename translation is requested, output an empty string "".
   - Do NOT output any folder translation field.
8. You must extract HIGH-VALUE PERSONAL/SENSITIVE INFORMATION if present. Extract into an "extracts" array.
9. For each extracted item, specify:
   - "type": classification (e.g., email, phone, id_card, password, api_key, bank_card, address, crypto_wallet, social_media, passport, date_of_birth, ip_address, license_plate, url_with_auth, email_content, other).
   - "owner": Who this information belongs to (e.g., "John Doe", "Company X"). If unknown, use "Unknown".
   - "description": Brief context. For API Keys, state the platform (e.g. "OpenAI Key").
   - "content": The actual value (e.g. the phone number, the key string).
10. If no sensitive info is found, output an empty array [] for "extracts".
</Rules>

[Output JSON Format]
{{"summary": "Your generated summary here", "file_name_en": "", "extracts": [{{"owner": "...", "type": "...", "description": "...", "content": "..."}}]}}

Document content:
{content}

Output JSON strictly:"""

# Backward-compatible alias — do not use in new code
SUMMARY_PROMPT_QWEN = SUMMARY_PROMPT_INDEXING


SUMMARY_PROMPT_EN = """Please analyze the following file content, and summarize its core content in one sentence.

<Instructions>
- Suggested summary formats:
  * Resume: name - role at company
  * Report: report name - core finding
  * Contract: contract type - party A and party B
  * Book: book title - topic
  * Invoice/Receipt: seller - invoice number - total amount: ¥/$/€ X.XX - date (MUST extract the total amount; if not found, write "amount: not found")
  * Other: concise topic and key point (max 100 words)
- Do not truncate important information like names, numbers, dates, etc.
- IMPORTANT: You MUST output the final summary strictly in 100% English. Do not output any Chinese characters except for the original names as specified below.
- For proper nouns such as company names and person names, you MUST keep the original language name (e.g., in Chinese) and append the English translation or pinyin in parentheses.
- CRITICAL: If the [Text content] below is empty or lacks meaningful text, you MUST directly output "Summary: Document content is empty or unreadable." without inventing anything or repeating instructions.
- ONLY output the summary and nothing else. Do NOT output any conversational text or explanations. Start your response directly with "Summary:".
</Instructions>

Text content:
{content}

Please extract and output in 100% English:
Summary:
"""

IMAGE_SUMMARY_PROMPT_EN = """Please summarize this image as a short paragraph so it can be retrieved by text search.

<Instructions>
- Output EXACTLY one short paragraph in English. Do not output any title, label, wrapper, apology, greeting, or explanation.
- Start directly with the visible subject, object, screen, document, chart, or scene. Do NOT use meta-phrases such as "Based on the image", "In this image", "The image shows", "This image shows", "The image depicts", or "This is an image of".
- Describe the primary subject first, then the most retrievable details: objects, scene, people, visible text, UI/app context, chart topic, or document type.
- If it is a document screenshot or UI screenshot, include the most important visible text and what kind of screen/document it is.
- If it is a chart, table, or diagram, summarize the chart type or structure and the main data signal/conclusion.
- If it is a simple icon, logo, stamp, badge, or graphic, say that directly and describe the shape, symbol, and colors without inventing extra context.
- Be factual and conservative. Do not guess hidden intent, brand, location, or off-screen context unless it is clearly visible.
- Never output tool-call syntax, XML tags, JSON, Markdown fences, or special tokens such as <|tool_call|>, call:, function_call, or tool output.
- Keep it concise, retrieval-friendly, and no more than 90 words.
</Instructions>

Write the paragraph now:
"""

VIDEO_FRAME_SUMMARY_PROMPT_EN = """Describe what is happening in this video frame in rich detail.

<Instructions>
- Start directly with the subject or action — do NOT open with "Based on the image", "In this frame", "The image shows", or any similar meta-phrase.
- Lead with the primary subject or action (e.g. "A person is speaking at a podium", "Code is being edited in a dark-themed editor", "A line is being drawn between two nodes on a canvas").
- Describe the scene thoroughly: layout, colors, positions, sizes, and spatial relationships of key elements.
- List all visible text, labels, titles, buttons, menus, icons, or status indicators — quote exact text when legible.
- If people are present, describe their appearance, posture, gestures, and any objects they are holding or interacting with.
- If it is a UI or application screen, describe the toolbar, sidebar, main content area, and any active state (selected tabs, highlighted items, cursor position, etc.). Do NOT guess brand names unless shown on screen.
- Be specific and concrete: prefer "a blue circle with a 2px black outline is centered on a white canvas, with a red arrow pointing from it to a green rectangle labeled 'Start'" over "the interface is displayed".
- Write 4–6 detailed sentences.
- IMPORTANT: Output ONLY the description content itself. Do NOT output any preamble, explanation, label, header, or wrapper. No "Description:", no "Here is...", no "Output:", no "Summary:" — just the raw descriptive sentences and nothing else.
</Instructions>
"""

VIDEO_FRAME_SUMMARY_WITH_CONTEXT_PROMPT_EN = """Describe what is happening in this video frame in rich detail. Use the previous frame's description for context to highlight changes.

Previous frame: {prev_desc}

<Instructions>
- Start directly with what changed or what is happening — do NOT open with "Based on the image", "Based on the previous frame", "In this frame", or any similar meta-phrase.
- Lead with what has CHANGED or what is actively happening in THIS frame compared to the previous one.
- Describe new or changed elements in detail: new text typed, moved objects, color changes, new UI panels opened, cursor movements, animations advancing, people entering/leaving, gestures, etc.
- Quote any newly visible or changed text, labels, or values exactly as they appear.
- If the view is largely the same, note continuity and describe any subtle differences (scroll position, selection highlight, progress bar, timer, etc.).
- Describe layout, colors, positions, and spatial relationships of all significant elements — not just what changed but also the surrounding context that helps understand the action.
- Do NOT guess brand names unless explicitly shown on screen.
- Write 4–6 detailed sentences.
- IMPORTANT: Output ONLY the description content itself. Do NOT output any preamble, explanation, label, header, or wrapper. No "Description:", no "Here is...", no "Output:", no "Summary:" — just the raw descriptive sentences and nothing else.
</Instructions>
"""

IMAGE_OCR_PROMPT_EN = """You are an OCR and document understanding assistant. Extract text from the image as completely and accurately as possible.

<Instructions>
- Output extracted text only, in reading order.
- No explanations or extra commentary.
- If a region is unreadable, use "[unreadable]".
</Instructions>

Output OCR result:
"""

MAP_PROMPT_EN = """Extract key information relevant to the user's question from the following chunk.

[User Question]
{question}

[Chunk {chunk_index}/{total_chunks}]
{chunk}

[Requirements]
1. Keep only directly relevant facts.
2. Keep critical data, file names, paths, and concrete details.
3. Remove unrelated repeated content.
4. If nothing relevant exists, output "No relevant information".
5. Keep concise.
"""

REDUCE_PROMPT_EN = """Merge the following chunk summaries into one complete answer.

[User Question]
{question}

[Chunk Summaries]
{combined_summaries}

[Requirements]
1. Merge all relevant facts and deduplicate.
2. Keep clear structure and logic.
3. Keep concrete file names/data/details.
"""

INTENT_DETECTION_PROMPT_EN = """You are an intelligent file retrieval assistant. Analyze user intent and output JSON.
The user may write in any language. Use English as the internal language for intent analysis and retrieval fields.

[Selected Source Categories]
{category_info}

{context_info}[User Input] {query}

[Available Actions]
{actions_text}

[Output Format] JSON only:
{{"action": "action_name", "params": {{"query": "English keywords here", ...}}}}
If action is search, params.query MUST be concise English retrieval keywords, regardless of the user's input language.
Preserve exact file names, product codes, invoice/order numbers, amounts, extensions, and proper nouns. If a non-English proper noun is important, keep the original token and add an English gloss when useful.
"""

LEXICAL_FEATURE_EXTRACTION_PROMPT_EN = """You are a fast and precise entity/feature extractor. Extract ONLY the core search entities, file names, or specific keywords from the user's query, stripping away all conversational filler (e.g., "search for", "about", "related content", "documents").
Output ONLY JSON in this format: {{"filenames": ["core entity or filename (no verbs/prepositions / general words)"], "extensions": [".pdf"]}}.
If none found, output {{"filenames": [], "extensions": []}}.
Rules:
- For phrases like "Can you find documents about <core topic>", extract filenames=["<core topic>"]. DO NOT include "Can you find" or "documents about".
- For phrases like "search for <entity name> related content", extract filenames=["<entity name>"].
- Extensions go into "extensions" only if the user explicitly mentions a file format (e.g., .pdf, excel, csv).
- Scope words such as "my", "all", "every" are not entities. Do not duplicate singular/plural variants in filenames.
- Preserve the exact spelling of the core entity. DO NOT translate Chinese names.

User Query: {query}"""

INTENT_DETECTION_SYSTEM_PROMPT_EN = """You are a capable AI assistant.
Choose one action strictly from [Available Actions], and output JSON only.
If required parameters are missing, use clarify.

[Language Boundary]
- Internal reasoning, intent labels, action params, and retrieval queries MUST be English-first.
- The user may write in any language. Translate search topics into concise English keywords for params.query.
- Preserve exact file names, model numbers, order/invoice IDs, amounts, paths, and extensions exactly as written.
- Final answer language is handled later by the response layer; do not localize internal params for the user's language.

[Typo & Robustness Rules]
- Treat misspelled words like "summery", "sunmery", "summaries", "sumary" as "summarize/summary" intent (use summarize or summarize_all).
- If the user provides a category (e.g. "images", "reports") along with a summarize request, MUST use `summarize` with `category` parameter, NOT `search`.
- DO NOT categorize educational content ("homework", "assignment", "worksheet", "test paper") as 'report' unless it is explicitly a business/research report.

[Multi-turn Rules]
When [Previous search/stat result] lists files and the user's short follow-up clearly refers to THAT list (e.g. "conclusion", "summarize", "recap", "tldr", "key points") — choose process_previous.
**Anti-pollution**: NEVER choose process_previous if the user introduces a NEW entity/topic NOT in the previous results, OR if the user asks for a global count/list (e.g. "what are all my files", "list my music files", "start over"). New subjects or global requests = new search/count, always.

[Semantic Query Translation]
When action="search" and input contains Chinese, translate params.query into concise English keywords.
Strip conversational filler. Keep proper nouns in original form with English in parentheses.
Examples: "项目计划报告" -> "project planning report", "搜索关于某公司相关的内容" -> "<company name>".

[Name-Only Query]
If input is a short proper noun (1-3 words, no verb): return params.query = original input exactly, no expansion.
Examples: "<person name>" -> query="<person name>", "<company name>" -> query="<company name>".

[Category Contract]
params.category is a filter bucket, not a topic field. Use it only for clear indexed file/media/type buckets such as image, document, data/spreadsheet, presentation, audio, video, resume, invoice, manual, report, paper, book, or code. Put topical phrases such as business plan, strategy, market, red bicycle, smart home sensors, or go-to-market in params.query. If ambiguous, leave category empty and keep the full topic in params.query.

[Prompt-First Intent Contract]
- First separate operation, content target, file-type filter, and scope/folder filter. No dimension may erase another.
- Scope words such as all/every/my/所有/全部/我的 define breadth only; they must not remove the topic.
- Choose count only for explicit quantity wording such as how many/count/number of/total/有多少/统计. Requests to show/list/find files of a type are search with category/extension, not count.
- If a type word has a content qualifier such as of/with/about/containing/featuring/showing/that mention, keep the qualifier target in params.query. Example: "find all images of a red bicycle" is image search for red bicycle, not all-image inventory.
- If the query has a folder scope such as in/inside/under/from the folder, keep the folder scope and keep the content target in params.query. Folder-scoped category search is not global category inventory.
- Proper nouns, filename-like phrases, invoice/order numbers, amounts, product codes, and title-like phrases are retrieval anchors. Do not drop them because generic words like file/document/report/image/video/data also appear.
- If selected/current/this/these wording clearly refers to selected files, use the selected-file summary path and never search for the literal word "selected".
- Active/indexed source scope is only a retrieval boundary. Do not treat it as selected files unless the user explicitly says selected/current/this/these or is clearly following up on prior results."""

VIEW_DETAIL_INTENT_PROMPT_EN = """[Task] Analyze user intent and decide what action to execute.
User input: "{query}"

If user wants to view a specific file from prior search results
(e.g. "open first file", "show summary of #3", "view result 2"), output:
{{"action": "view_detail", "index": N}}

If not, output:
{{"action": "other"}}
"""

REWRITE_QUERY_PROMPT_EN = """Given conversation history and current input, rewrite into one standalone search query.

History: {history_context}
Current: {current_query}

Requirement: output rewritten query only.
"""

AMBIGUOUS_QUERY_PROMPT_EN = """User input is ambiguous: "{question}".
Ask one clarifying question in one sentence (max 20 words)."""

SUMMARIZE_SINGLE_FILE_PROMPT_EN = """Summarize this file in about 120 words, and ask at the end whether the user wants to open the original file.

[IMPORTANT] Do NOT wrap your summary in a Markdown code block (```markdown ... ```).

[Summary] {summary}
[Content] {text_preview}

[Output format]
## Overview
2-3 sentences

## Key points
- point 1
- point 2
- point 3

---
**Do you want me to open the original file?**
"""

SUMMARIZE_TOPICS_PROMPT_EN = """{header}

Based on the summaries above, identify major topic areas covered by these {category} files.
Requirements:
1. 3-5 key topic bullets
2. concise output
3. ignore highly repetitive content
4. focus on macro themes
"""

SUMMARIZE_ALL_PROMPT_EN = """{context_text}
Generate a global summary of all files above.
Requirements:
1. Brief overall distribution.
2. If multiple categories exist, summarize each category.
3. Identify possible connections/storyline across files.
4. {focus_instruction}
5. Professional and concise markdown output.
6. [IMPORTANT] Do NOT wrap the summary in a Markdown code block (```markdown ... ```).
"""

CAPABILITY_QUERY_PROMPT_EN = """{instruction}
[Recent Conversation]
{recent_block}

[User Question]
{question}
"""

NO_RESULT_PROMPT_EN = """User question: "{question}"

No highly relevant content was found in the current vector knowledge base.
Reply politely and suggest 1-2 next steps (e.g. refine keywords, check indexing status).
"""

CHAT_FALLBACK_PROMPT_EN = """User asked: "{question}".
No suitable tool is available now. Reply directly as an AI assistant.
"""

PLANNER_PROMPT_EN = """You are a task planning assistant. Based on the user request and available tools, generate an execution plan.

[Available Tools]
{tools_desc}

[User Request]
{question}

[Requirements]
1. Plan steps using available tools.
2. Output must be a valid JSON array.
3. Each item includes 'tool' and 'args'.
4. If impossible, output [].
"""

FINAL_ANSWER_PROMPT_EN = """Answer the user based on tool execution results.

Requirements:
1. Be direct and professional; stay short when possible and avoid filler.
2. If there are errors in the tool output, explain kindly.
3. If an action ran (e.g. open file), state the outcome.
4. For query-style results, summarize essentials and include all necessary points—do not drop items just to be shorter.

Tool result:
{tool_result_str}

User question: "{question}"
"""

EFFICIENT_ASSISTANT_PROMPT_EN = """You are an efficient file assistant. Response requirements:
1. Get to the point; no filler openers.
2. Be as brief as clarity allows, but stay complete; omit unrelated wording only.
3. For search results, list core facts directly.
4. For summaries, keep concise yet complete.
5. Use markdown bullets for multi-item output.
6. Reply in the user's language.

Context/tool results:
{context}

User question: "{question}"
"""

DETECT_CATEGORY_PROMPT_EN = """User question: "{question}"

Current categories in DB: {categories_str}

Decide which category user asks for. If user asks for all files or unclear, output "all".
Output category name only.
"""

AGENT_SYSTEM_PROMPT_EN = """You are a local file search and QA assistant. You can use search_documents to query local indexed files and answer based on results."""
DETAILED_SYSTEM_PROMPT_EN = """You are a professional document analyst. Be complete and do not miss important points; keep wording tight and avoid filler, repetition, or off-topic remarks."""
BRIEF_SYSTEM_PROMPT_EN = """You are an efficient file assistant. Be as brief as clarity allows, but cover everything the user needs; avoid unrelated pleasantries, padding, or redundant phrasing."""

STREAM_ASK_SYSTEM_PROMPT_EN = """You are a helpful file assistant. Answer directly based on provided file content. Be concise and skip irrelevant wording; do not omit key facts the user needs.

{selection_warning}
[Context]
{context_info}
"""

TOOL_CALLS_SYSTEM_PROMPT_EN = """You are an intelligent file management assistant. You must call tools to help the user.

[File tools scope] {file_tools_root}
[Available tools]
- {tool_names_list}
"""

TASK_EXECUTION_SYSTEM_PROMPT_EN = """You are an efficient system operation assistant. Execute the task using tools step by step.

[File tools scope] {file_tools_root}
[Available tools]
- {tool_names_list}
"""

QUERY_TRANSLATION_PROMPT_EN = """You are a retrieval query translator for an English-first semantic retrieval layer.
Convert the user query into concise English retrieval keywords.
Rules:
- The user may write in any language; translate the search topic to English.
- Keep exact file names, paths, extensions, product/model terms, entities, names, and numbers unchanged.
- Strip conversational filler such as "find", "show me", "related to", "帮我找", or "关于".
- Output one keyword query only, no explanation.
- Keep it short: usually 2-12 words.

User query: {query}
English retrieval keywords:
"""

# ==================== Multi-Agent Router Prompts (v2 Architecture) ====================

ROUTER_PROMPT_EN = """You are a conversation router. Output ONE word only — nothing else.

Paths:
- "continuation" — user is following up the PREVIOUS assistant response.
  Signals: translation request ("in english"/"用中文"), expand/detail ("tell me more"/
  "elaborate"/"please elaborate"), pronoun reference ("them"/"these"/"it"/"the first one"/
  "above"/"刚才"/"上面"), short follow-up when [Has prior response]=Yes
  ("conclusion?"/"summarize"/"recap"/"ok and then?").
- "media" — user wants to query INSIDE audio/video content, or perform timestamp/transcript/frame/clip/export operations (e.g. "which videos mention X", "does any video show Z", "what happens at 10s").
  Signals: video/audio/recording + internal-content or operation words (inside/mention/say/show/appear/scene/transcript/timestamp/frame/clip/export/transcribe)
- "file_op" — user wants to find, search, count, open, list, summarize files/docs/images, OR asks factual questions about specific entities/people/projects ("What is the phone number of X?", "Who is Y?").
  Signals: find/search/count/open/list/show/summarize/how many/give me/what is/who is/phone/address
- "chat" — general question, greeting, coding request, or topic entirely unrelated to factual search.

[Has prior response]: {has_prior}
[Prior topic (first 80 chars)]: {prior_topic}
[Prior user query]: {prior_user_query}
[Recent result files]: {recent_result_files}
[Focused file]: {focused_file}
[User]: {query}

Additional routing guidance:
- If the previous user query was clearly searching for an article/profile/report/guide/topic, and the current message asks for details, writing style, target users, revenue model, required version, whether it mentions something, ordinal results ("first one"), or short fragment questions ("holiday policy?"), prefer continuation.
- This remains true even when [Recent result files] is empty or weak, as long as the current message is obviously continuing the previous search topic rather than starting a new global search.
- Only choose file_op when the user introduces a new entity/scope/category or clearly restarts find/search/list/count/open actions.
- "find/list/show videos/audio/images about X" is media file retrieval, so choose file_op. Choose media only for internal-content wording such as "which videos mention X", "what does the video say/show about X", or timestamp/transcript/frame/clip operations.

Examples (for reference):
  "tell me more" [prior]=Yes → continuation
  "in english" [prior]=Yes → continuation
  "what about the 2nd one?" [prior]=Yes → continuation
  "tell me about 生成式推荐 above" [prior]=Yes → continuation
  "please elaborate" [prior]=Yes → continuation
  "what writing style and tone does this article use" [prior article/result]=Yes → continuation
  "what is the go-to-market strategy and target user group" [prior business plan/result]=Yes → continuation
  "does the profile mention any notable campaign or brand wins" [prior profile/result]=Yes → continuation
  "what node version do I need" [prior setup guide/result]=Yes → continuation
  "find videos about a product launch" → file_op
  "which videos mention machine learning" → media
  "videos containing code editor scenes" → media
  "find my resume" → file_op
  "how many photos" → file_op
  "open the contract.pdf" → file_op
  "summarize all reports" → file_op
  "what time is it" → chat
  "who are you" → chat

Output ONE word only (continuation / file_op / media / chat):"""

CONTINUATION_AGENT_PROMPT_EN = """User is following up the previous assistant response. Output JSON.

[Previous response (first 150 chars)]: {prev_response_preview}
[Previous user query]: {prev_user_query}
[Recent result files]: {last_results_preview}
[Focused file]: {focused_file}
[User message]: {query}

Choose ONE action:

1. Translation request (user wants language switch):
   → {{"action":"translate_response","lang":"en"}} or {{"action":"translate_response","lang":"zh"}}
   Triggers: "in english", "in chinese", "translate", "reply in english", "answer in Chinese", "in english above", "translate it"

2. View a specific numbered result (user references item by number):
   → {{"action":"view_detail","params":{{"index":N}}}}
   Triggers: "the 1st one", "the 2nd", "#2", "item 3", "first one", "number 2"

3. Expand / summarize / refine the previous response:
   → {{"action":"process_previous","params":{{}}}}
   Triggers: "conclusion", "summarize", "recap", "tldr", "elaborate", "tell me more",
   "key points", pronoun references ("them"/"these"/"it"), revision/focus requests.
   When prior response was a SUMMARY, 'focus only on X / more briefly / make it shorter /
   which files support that' are ALWAYS process_previous (refining the summary, not new search).

4. Unrelated new request or entirely new topic (e.g. "what are my api keys", "what files do I have", "find the contract"):
   → {{"action":"fallback_to_file_op","params":{{}}}}

⚠️ GUARD: If query STARTS WITH find/search/list/get/count/what files/what are my/how many/do i have,
   OR introduces a completely new entity not mentioned in the previous response (e.g. 'api keys', 'invoices'),
   → it's a NEW search request, NOT a continuation → output {{"action":"fallback_to_file_op","params": {{}}}}
   NOTE: 'show me what it says', 'what are the key points', 'what does it contain' ARE continuations.

Examples:
  "in english" → {{"action":"translate_response","lang":"en"}}
  "reply in Chinese" → {{"action":"translate_response","lang":"zh"}}
  "tell me about the 2nd one" → {{"action":"view_detail","params":{{"index":2}}}}
  "conclusion?" → {{"action":"process_previous","params":{{}}}}
  "please elaborate" → {{"action":"process_previous","params":{{}}}}
  "what are the key points?" → {{"action":"process_previous","params":{{}}}}
  "focus only on the most important file" → {{"action":"process_previous","params":{{}}}}
  "now focus only on text files" → {{"action":"process_previous","params":{{}}}}
  "summarize it more briefly" → {{"action":"process_previous","params":{{}}}}
  "which files support that summary" → {{"action":"process_previous","params":{{}}}}
  "find a new report" → {{"action":"fallback_to_file_op","params":{{}}}}

Output JSON only:"""

FILE_OP_AGENT_PROMPT_EN = """User wants a file operation. Output JSON.

[Selected files ({n_sel}, user ticked these)]: {selection_preview}
[Recent search results ({n_last})]: {last_results_preview}
[User]: {query}

Available actions (pick one):
- search: semantic search → {{"action":"search","params":{{"query":"English keywords"}}}}
- count: count files by category → {{"action":"count","params":{{"category":"category name"}}}}
- summarize: summarize files in a category → {{"action":"summarize","params":{{"category":"category name"}}}}
- summarize_all: global or selected-files overview → {{"action":"summarize_all","params":{{}}}}
- view_detail: view specific numbered file → {{"action":"view_detail","params":{{"index":N}}}}
- open_file: open a named file → {{"action":"open_file","params":{{"file_name":"filename"}}}}

RULES (by priority):
1. search query MUST be English (translate Chinese input).
2. Only "how many X" / "count X" / "number of X" / "total X" → count(category=X).
   "all X files" / "list X" / "show X files" / "find X files" → search with category/extension. "all X about [topic]" / "all X that mention Y" is also search, not count.
3. "summarize all" / "overview of everything" / "summarize my {n_sel} selected files" → summarize_all.
   If the user asks to tell/describe/explain/summarize/overview selected/current/this/these file(s) or document(s), use selected-file summary; do not search for the literal word "selected".
4. "summarize [category] files" → summarize(category=[category]).
5. "the Nth one" / "#N" / "number N" → view_detail(index=N).
6. ONLY when user explicitly says "open" / "launch" [filename] → open_file. "show me" / "tell me about" / "look at" are NOT open — use search instead.
7. category field: free-text label for a clear file type (image/report/resume/data/invoice/audio...
   or any custom label — not limited to a fixed list).
   Do NOT set category="document" just because the user says "file", "document", or "docs".
   Those are usually generic container words. For title-like, list/catalog, or table requests,
   keep the full topic in query and leave category empty unless a narrow type filter is clear.

File type aliases (use these to map user terms → category):
  docx / word / txt → "document"; pdf usually belongs in file_extensions unless the user asks for all PDFs
  report / reports → "report"
  paper / papers / academic → "paper"
  resume / cv → "resume"
  contract → "contract" | invoice → "invoice"
  ppt / pptx / slides / presentation → "presentation"
  csv / xlsx / xls / excel / spreadsheet → "data"
  image / photo / jpg / png / picture / screenshot → "image"
  video / mp4 / mov → "video"; audio / mp3 / wav / recording → "audio"

Examples (⭐ key cases):
  "find my doc files" → {{"action":"search","params":{{"query":"doc files","category":"document"}}}}
  "show me ppt files" → {{"action":"search","params":{{"query":"ppt files","category":"presentation"}}}}
  "list all csv" → {{"action":"search","params":{{"query":"csv files","category":"data"}}}}
  "find my resume" → {{"action":"search","params":{{"query":"resume","category":"resume"}}}}
  "how many PDFs" → {{"action":"count","params":{{"category":"document"}}}}
  "all my photos" → {{"action":"search","params":{{"query":"photos","category":"image"}}}}
  "all reports about Q4 revenue" → {{"action":"search","params":{{"query":"Q4 revenue reports"}}}}
  "find the product catalog document" → {{"action":"search","params":{{"query":"product catalog"}}}}
  "summarize report files" → {{"action":"summarize","params":{{"category":"report"}}}}
  "summarize all files" → {{"action":"summarize_all","params":{{}}}}
  "summarize my {n_sel} selected files" → {{"action":"summarize_all","params":{{}}}}
  "the 2nd one" → {{"action":"view_detail","params":{{"index":2}}}}
  "open contract.pdf" → {{"action":"open_file","params":{{"file_name":"contract.pdf"}}}}
  "tell me about test.csv" → {{"action":"search","params":{{"query":"test.csv"}}}} (NOT open_file!)

Output JSON only:"""

# ==================== Media Sub-Agent Prompts ====================

MEDIA_SUB_AGENT_PROMPT_EN = """User wants to query audio/video content. Output JSON.

[User]: {query}

Available actions (pick one):
- media_content_search: search video/audio content by topic → {{"action":"media_content_search","params":{{"query":"English topic keywords","media_type":"video|audio|all"}}}}
- media_count: count video/audio files → {{"action":"media_count","params":{{"media_type":"video|audio|all"}}}}
- media_summarize: summarize video/audio files → {{"action":"media_summarize","params":{{"media_type":"video|audio|all"}}}}

Rules:
1. query MUST be English (translate Chinese). Extract the actual topic the user wants to search.
2. media_type: video if user mentions video, audio if audio, all if both or unclear.
3. "which videos mention X" / "what does the video say or show about X" → media_content_search
4. "find videos about X" / "show audio files about X" is media file retrieval, not this sub-agent; the upper router should send it to file_op/search
5. "how many videos" → media_count
6. "summarize my videos" → media_summarize

Examples:
  "which videos mention a topic" → {{"action":"media_content_search","params":{{"query":"topic","media_type":"video"}}}}
  "what does the video say about a topic" → {{"action":"media_content_search","params":{{"query":"topic","media_type":"video"}}}}
  "how many videos do I have" → {{"action":"media_count","params":{{"media_type":"video"}}}}

Output JSON only:"""

# ==================== English-Controlled Chinese Output Prompts ====================
#
# These prompts keep the model-facing control language in English while preserving the
# product contract that final user-facing answers follow the user's language.

SUMMARIZE_SINGLE_FILE_PROMPT_ZH_CONTROL = """Summarize this file in Simplified Chinese in about 150 Chinese characters. Use markdown sections, and ask at the end whether the user wants to open the original file.

[IMPORTANT] Do NOT wrap your summary in a Markdown code block.

[Existing index summary]
{summary}

[File content preview]
{text_preview}

[Output format, in Simplified Chinese]
## 内容概述
2-3 sentences

## 核心信息
- key point 1
- key point 2
- key point 3

---
**需要我帮你打开原文查看吗？**
"""

SUMMARIZE_VIDEO_FILE_PROMPT_ZH_CONTROL = """Based on the video metadata and keyframe descriptions below, describe the video to the user in Simplified Chinese.

[IMPORTANT] Do NOT wrap the answer in a Markdown code block.
[IMPORTANT] Do NOT say you cannot watch the video; use the provided keyframe descriptions directly.

[Video metadata]
{metadata}

[Keyframe descriptions, chronological]
{keyframes}

[Output format, in Simplified Chinese]
## 视频内容介绍
2-3 sentences summarizing the main content

## 关键画面
List important keyframes in chronological order

## 总结
Briefly state the purpose or value

---
**需要我帮你打开原文查看吗？**
"""

SUMMARIZE_TOPICS_PROMPT_ZH_CONTROL = """{header}

Based on the summaries above, identify the major topic areas covered by these {category} files.
Write the final answer in Simplified Chinese.

Requirements:
1. Use 3-5 concise bullet points.
2. Ignore duplicate or highly similar content.
3. Focus on macro themes instead of listing every file.
"""

SUMMARIZE_ALL_PROMPT_ZH_CONTROL = """{context_text}

Generate a global summary of the files above.
Write the final answer in Simplified Chinese.

Requirements:
1. Briefly describe the overall content distribution.
2. If there are multiple categories, summarize what each category mainly covers.
3. Identify possible connections or a common storyline across files.
4. {focus_instruction}
5. Use concise professional markdown, under 500 Chinese characters when possible.
6. Do NOT wrap the summary in a Markdown code block.
"""

CAPABILITY_QUERY_PROMPT_ZH_CONTROL = """{instruction}

[Recent conversation]
{recent_block}

[User question]
{question}

Reply in Simplified Chinese unless the user explicitly requests another language.
"""

NO_RESULT_PROMPT_ZH_CONTROL = """User question: "{question}"

No highly relevant file content was found in the current vector knowledge base.
Reply in Simplified Chinese. Keep it concise, and suggest 1-2 next steps such as refining keywords or checking whether the file has been indexed.
"""

CHAT_FALLBACK_PROMPT_ZH_CONTROL = """User asked: "{question}".
No suitable file tool is available for this question. Reply directly as an AI assistant in Simplified Chinese.
Keep it short and direct.
"""

AMBIGUOUS_QUERY_PROMPT_ZH_CONTROL = """User input is ambiguous: "{question}".
Ask one clarifying question in Simplified Chinese, in one sentence and under 25 Chinese characters when possible.
"""

FINAL_ANSWER_PROMPT_ZH_CONTROL = """Answer the user in Simplified Chinese based on the tool execution results.

Requirements:
1. Be direct and professional; stay short when possible and avoid filler.
2. If the result contains an error, explain it kindly.
3. If an action ran, such as opening a file, state the outcome.
4. For query-style results, summarize essentials and include all necessary points.

[Tool result]
{tool_result_str}

[User question]
{question}
"""

EFFICIENT_ASSISTANT_PROMPT_ZH_CONTROL = """You are an efficient file assistant. Reply in Simplified Chinese.

Requirements:
1. Get to the point; no filler openers.
2. Be as brief as clarity allows, but stay complete.
3. For search results, list core facts directly.
4. For summaries, keep concise yet complete.
5. Use markdown bullets for multiple items.

[Context/tool results]
{context}

[User question]
{question}
"""

DETECT_CATEGORY_PROMPT_ZH_CONTROL = """User question: "{question}"

Current categories in the database: {categories_str}

Decide which category the user asks for. If the user asks for all files or the category is unclear, output "all".
Output the category name only. Use the database category label, not a localized explanation.
"""

AGENT_SYSTEM_PROMPT_ZH_CONTROL = """You are a local file search and QA assistant. Use local indexed files when relevant. Reply to the user in Simplified Chinese unless they explicitly ask for another language."""
DETAILED_SYSTEM_PROMPT_ZH_CONTROL = """You are a professional document analyst. Reply in Simplified Chinese. Be complete and do not miss important points; keep wording tight and avoid filler."""
BRIEF_SYSTEM_PROMPT_ZH_CONTROL = """You are an efficient file assistant. Reply in Simplified Chinese. Be brief, complete, and avoid unrelated pleasantries."""

STREAM_ASK_SYSTEM_PROMPT_ZH_CONTROL = """You are a helpful file assistant. Answer directly in Simplified Chinese based on the provided file content. Be concise and do not omit key facts.

{selection_warning}
[Context]
{context_info}
"""

TOOL_CALLS_SYSTEM_PROMPT_ZH_CONTROL = """You are an intelligent file management assistant. You must call tools to help the user.
Use English internally for tool decisions and paths. User-facing text should be Simplified Chinese unless the user asks otherwise.

[File tools scope] {file_tools_root}
[Available tools]
- {tool_names_list}
"""

TASK_EXECUTION_SYSTEM_PROMPT_ZH_CONTROL = """You are an efficient system operation assistant. Execute the task using tools step by step.
Use English internally for tool decisions. Output the final result or error in Simplified Chinese.

[File tools scope] {file_tools_root}
[Available tools]
- {tool_names_list}
"""

# Legacy top-level prompt constants default to English-first behavior. New code should
# prefer get_prompt(...), but older imports from config.prompts should not accidentally
# use Chinese control prompts.
QUERY_PROMPT_ENGLISH_CONTROL = """You are a professional knowledge QA assistant.
Use the provided context only. Do not invent facts.
The user may ask in any language; reason over the evidence in English internally, then answer in the same language as the user's question unless they request another language.

<Instructions>
- If the question asks "who", extract the full name from context.
- If the question asks for counts, inspect each context item as a separate file, list matching files, and provide the total.
- Cite source file names at the end when relevant.
- If there is not enough context, say so clearly.
</Instructions>

<Context>
{context_str}
</Context>

<Question>
{query_str}
</Question>

Answer:
"""

QUERY_PROMPT = QUERY_PROMPT_ENGLISH_CONTROL
CLASSIFY_PROMPT = CLASSIFY_PROMPT_EN
CLASSIFY_TAXONOMY_PROMPT = CLASSIFY_TAXONOMY_PROMPT_EN
SUMMARY_PROMPT = SUMMARY_PROMPT_EN
IMAGE_SUMMARY_PROMPT = IMAGE_SUMMARY_PROMPT_EN
IMAGE_OCR_PROMPT = IMAGE_OCR_PROMPT_EN
MAP_PROMPT = MAP_PROMPT_EN
REDUCE_PROMPT = REDUCE_PROMPT_EN
LEXICAL_FEATURE_EXTRACTION_PROMPT = LEXICAL_FEATURE_EXTRACTION_PROMPT_EN
INTENT_DETECTION_PROMPT = INTENT_DETECTION_PROMPT_EN
INTENT_DETECTION_SYSTEM_PROMPT = INTENT_DETECTION_SYSTEM_PROMPT_EN
VIEW_DETAIL_INTENT_PROMPT = VIEW_DETAIL_INTENT_PROMPT_EN
REWRITE_QUERY_PROMPT = REWRITE_QUERY_PROMPT_EN
AMBIGUOUS_QUERY_PROMPT = AMBIGUOUS_QUERY_PROMPT_EN
SUMMARIZE_SINGLE_FILE_PROMPT = SUMMARIZE_SINGLE_FILE_PROMPT_EN
SUMMARIZE_VIDEO_FILE_PROMPT = SUMMARIZE_VIDEO_FILE_PROMPT_EN
SUMMARIZE_TOPICS_PROMPT = SUMMARIZE_TOPICS_PROMPT_EN
SUMMARIZE_ALL_PROMPT = SUMMARIZE_ALL_PROMPT_EN
CAPABILITY_QUERY_PROMPT = CAPABILITY_QUERY_PROMPT_EN
NO_RESULT_PROMPT = NO_RESULT_PROMPT_EN
CHAT_FALLBACK_PROMPT = CHAT_FALLBACK_PROMPT_EN
PLANNER_PROMPT = PLANNER_PROMPT_EN
FINAL_ANSWER_PROMPT = FINAL_ANSWER_PROMPT_EN
EFFICIENT_ASSISTANT_PROMPT = EFFICIENT_ASSISTANT_PROMPT_EN
DETECT_CATEGORY_PROMPT = DETECT_CATEGORY_PROMPT_EN
AGENT_SYSTEM_PROMPT = AGENT_SYSTEM_PROMPT_EN
DETAILED_SYSTEM_PROMPT = DETAILED_SYSTEM_PROMPT_EN
BRIEF_SYSTEM_PROMPT = BRIEF_SYSTEM_PROMPT_EN
STREAM_ASK_SYSTEM_PROMPT = STREAM_ASK_SYSTEM_PROMPT_EN
TOOL_CALLS_SYSTEM_PROMPT = TOOL_CALLS_SYSTEM_PROMPT_EN
TASK_EXECUTION_SYSTEM_PROMPT = TASK_EXECUTION_SYSTEM_PROMPT_EN
ROUTER_PROMPT = ROUTER_PROMPT_EN
CONTINUATION_AGENT_PROMPT = CONTINUATION_AGENT_PROMPT_EN
FILE_OP_AGENT_PROMPT = FILE_OP_AGENT_PROMPT_EN
MEDIA_SUB_AGENT_PROMPT = MEDIA_SUB_AGENT_PROMPT_EN

# ==================== Prompt Selector ====================

SUPPORTED_PROMPT_LANGUAGES = {"zh", "en"}

_PROMPT_TEMPLATES = {
    "zh": {
        # Internal/index/search prompts stay English-first even for non-English user input.
        "CLASSIFY_PROMPT": CLASSIFY_PROMPT_EN,
        "CLASSIFY_TAXONOMY_PROMPT": CLASSIFY_TAXONOMY_PROMPT_EN,
        "SUMMARY_PROMPT": SUMMARY_PROMPT_EN,
        "SUMMARY_PROMPT_INDEXING": SUMMARY_PROMPT_INDEXING,
        "SUMMARY_AND_EXTRACT_PROMPT": SUMMARY_AND_EXTRACT_PROMPT,
        "SUMMARY_PROMPT_QWEN": SUMMARY_PROMPT_INDEXING,  # backward-compat alias
        "IMAGE_SUMMARY_PROMPT": IMAGE_SUMMARY_PROMPT_EN,
        "VIDEO_FRAME_SUMMARY_PROMPT": VIDEO_FRAME_SUMMARY_PROMPT_EN,
        "VIDEO_FRAME_SUMMARY_WITH_CONTEXT_PROMPT": VIDEO_FRAME_SUMMARY_WITH_CONTEXT_PROMPT_EN,
        "IMAGE_OCR_PROMPT": IMAGE_OCR_PROMPT_EN,
        "MAP_PROMPT": MAP_PROMPT_EN,
        "REDUCE_PROMPT": REDUCE_PROMPT_EN,
        "INTENT_DETECTION_PROMPT": INTENT_DETECTION_PROMPT_EN,
        "LEXICAL_FEATURE_EXTRACTION_PROMPT": LEXICAL_FEATURE_EXTRACTION_PROMPT_EN,
        "INTENT_DETECTION_SYSTEM_PROMPT": INTENT_DETECTION_SYSTEM_PROMPT_EN,
        "VIEW_DETAIL_INTENT_PROMPT": VIEW_DETAIL_INTENT_PROMPT_EN,
        "REWRITE_QUERY_PROMPT": REWRITE_QUERY_PROMPT_EN,
        "AMBIGUOUS_QUERY_PROMPT": AMBIGUOUS_QUERY_PROMPT_ZH_CONTROL,
        "SUMMARIZE_SINGLE_FILE_PROMPT": SUMMARIZE_SINGLE_FILE_PROMPT_ZH_CONTROL,
        "SUMMARIZE_VIDEO_FILE_PROMPT": SUMMARIZE_VIDEO_FILE_PROMPT_ZH_CONTROL,
        "SUMMARIZE_TOPICS_PROMPT": SUMMARIZE_TOPICS_PROMPT_ZH_CONTROL,
        "SUMMARIZE_ALL_PROMPT": SUMMARIZE_ALL_PROMPT_ZH_CONTROL,
        "CAPABILITY_QUERY_PROMPT": CAPABILITY_QUERY_PROMPT_ZH_CONTROL,
        "NO_RESULT_PROMPT": NO_RESULT_PROMPT_ZH_CONTROL,
        "CHAT_FALLBACK_PROMPT": CHAT_FALLBACK_PROMPT_ZH_CONTROL,
        "PLANNER_PROMPT": PLANNER_PROMPT_EN,
        "FINAL_ANSWER_PROMPT": FINAL_ANSWER_PROMPT_ZH_CONTROL,
        "EFFICIENT_ASSISTANT_PROMPT": EFFICIENT_ASSISTANT_PROMPT_ZH_CONTROL,
        "DETECT_CATEGORY_PROMPT": DETECT_CATEGORY_PROMPT_ZH_CONTROL,
        "AGENT_SYSTEM_PROMPT": AGENT_SYSTEM_PROMPT_ZH_CONTROL,
        "DETAILED_SYSTEM_PROMPT": DETAILED_SYSTEM_PROMPT_ZH_CONTROL,
        "BRIEF_SYSTEM_PROMPT": BRIEF_SYSTEM_PROMPT_ZH_CONTROL,
        "STREAM_ASK_SYSTEM_PROMPT": STREAM_ASK_SYSTEM_PROMPT_ZH_CONTROL,
        "TOOL_CALLS_SYSTEM_PROMPT": TOOL_CALLS_SYSTEM_PROMPT_ZH_CONTROL,
        "TASK_EXECUTION_SYSTEM_PROMPT": TASK_EXECUTION_SYSTEM_PROMPT_ZH_CONTROL,
        "QUERY_TRANSLATION_PROMPT": QUERY_TRANSLATION_PROMPT_EN,
        # v2 multi-agent router prompts
        "ROUTER_PROMPT": ROUTER_PROMPT_EN,
        "CONTINUATION_AGENT_PROMPT": CONTINUATION_AGENT_PROMPT_EN,
        "FILE_OP_AGENT_PROMPT": FILE_OP_AGENT_PROMPT_EN,
        "MEDIA_SUB_AGENT_PROMPT": MEDIA_SUB_AGENT_PROMPT_EN,
    },
    "en": {
        "CLASSIFY_PROMPT": CLASSIFY_PROMPT_EN,
        "CLASSIFY_TAXONOMY_PROMPT": CLASSIFY_TAXONOMY_PROMPT_EN,
        "SUMMARY_PROMPT": SUMMARY_PROMPT_EN,
        "SUMMARY_PROMPT_INDEXING": SUMMARY_PROMPT_INDEXING,
        "SUMMARY_AND_EXTRACT_PROMPT": SUMMARY_AND_EXTRACT_PROMPT,
        "SUMMARY_PROMPT_QWEN": SUMMARY_PROMPT_INDEXING,  # backward-compat alias
        "IMAGE_SUMMARY_PROMPT": IMAGE_SUMMARY_PROMPT_EN,
        "VIDEO_FRAME_SUMMARY_PROMPT": VIDEO_FRAME_SUMMARY_PROMPT_EN,
        "VIDEO_FRAME_SUMMARY_WITH_CONTEXT_PROMPT": VIDEO_FRAME_SUMMARY_WITH_CONTEXT_PROMPT_EN,
        "IMAGE_OCR_PROMPT": IMAGE_OCR_PROMPT_EN,
        "MAP_PROMPT": MAP_PROMPT_EN,
        "REDUCE_PROMPT": REDUCE_PROMPT_EN,
        "INTENT_DETECTION_PROMPT": INTENT_DETECTION_PROMPT_EN,
        "LEXICAL_FEATURE_EXTRACTION_PROMPT": LEXICAL_FEATURE_EXTRACTION_PROMPT_EN,
        "INTENT_DETECTION_SYSTEM_PROMPT": INTENT_DETECTION_SYSTEM_PROMPT_EN,
        "VIEW_DETAIL_INTENT_PROMPT": VIEW_DETAIL_INTENT_PROMPT_EN,
        "REWRITE_QUERY_PROMPT": REWRITE_QUERY_PROMPT_EN,
        "AMBIGUOUS_QUERY_PROMPT": AMBIGUOUS_QUERY_PROMPT_EN,
        "SUMMARIZE_SINGLE_FILE_PROMPT": SUMMARIZE_SINGLE_FILE_PROMPT_EN,
        "SUMMARIZE_VIDEO_FILE_PROMPT": SUMMARIZE_VIDEO_FILE_PROMPT_EN,
        "SUMMARIZE_TOPICS_PROMPT": SUMMARIZE_TOPICS_PROMPT_EN,
        "SUMMARIZE_ALL_PROMPT": SUMMARIZE_ALL_PROMPT_EN,
        "CAPABILITY_QUERY_PROMPT": CAPABILITY_QUERY_PROMPT_EN,
        "NO_RESULT_PROMPT": NO_RESULT_PROMPT_EN,
        "CHAT_FALLBACK_PROMPT": CHAT_FALLBACK_PROMPT_EN,
        "PLANNER_PROMPT": PLANNER_PROMPT_EN,
        "FINAL_ANSWER_PROMPT": FINAL_ANSWER_PROMPT_EN,
        "EFFICIENT_ASSISTANT_PROMPT": EFFICIENT_ASSISTANT_PROMPT_EN,
        "DETECT_CATEGORY_PROMPT": DETECT_CATEGORY_PROMPT_EN,
        "AGENT_SYSTEM_PROMPT": AGENT_SYSTEM_PROMPT_EN,
        "DETAILED_SYSTEM_PROMPT": DETAILED_SYSTEM_PROMPT_EN,
        "BRIEF_SYSTEM_PROMPT": BRIEF_SYSTEM_PROMPT_EN,
        "STREAM_ASK_SYSTEM_PROMPT": STREAM_ASK_SYSTEM_PROMPT_EN,
        "TOOL_CALLS_SYSTEM_PROMPT": TOOL_CALLS_SYSTEM_PROMPT_EN,
        "TASK_EXECUTION_SYSTEM_PROMPT": TASK_EXECUTION_SYSTEM_PROMPT_EN,
        "QUERY_TRANSLATION_PROMPT": QUERY_TRANSLATION_PROMPT_EN,
        # v2 multi-agent router prompts
        "ROUTER_PROMPT": ROUTER_PROMPT_EN,
        "CONTINUATION_AGENT_PROMPT": CONTINUATION_AGENT_PROMPT_EN,
        "FILE_OP_AGENT_PROMPT": FILE_OP_AGENT_PROMPT_EN,
        "MEDIA_SUB_AGENT_PROMPT": MEDIA_SUB_AGENT_PROMPT_EN,
        # v2 new classifiers
        "INTENT_CLASSIFIER_PROMPT": "INTENT_CLASSIFIER_PROMPT_EN",  # resolved below
        "FOLLOWUP_CLASSIFIER_PROMPT": "FOLLOWUP_CLASSIFIER_PROMPT_EN",  # resolved below
    },
}

# ==================== v2 New Classifier Prompts ====================

INTENT_CLASSIFIER_PROMPT_EN = """Prior action: {prior_action}
Selected files: {n_selected}
Prior results: {n_results}
Prior topic: {prior_topic}
User: {query}

Groups:
- continuation: follow-up on prior conversation (only when n_results > 0)
- selection: about selected files (only when n_selected > 0)
- media: audio/video content query or timestamp lookup
- file_op: find/count/open/summarize files or ask factual entity question
- chat: greeting, general question unrelated to files

Rules:
- If n_selected=0, NEVER output selection
- If n_results=0 and prior_action=none, mostly file_op or chat
- Short pronoun reference + prior results -> continuation
- Timestamp + audio/video word -> media

Output ONE word (continuation / selection / media / file_op / chat):"""

FOLLOWUP_CLASSIFIER_PROMPT_EN = """Followup(A) or new request(B)? Output A or B only.

[Prior action]: {prior_action} (found {n_results} files)
[Prior files]: {file_names}
[User says]: "{query}"

A = follow-up on prior results (summarize/detail/content/translate/continue/refine/narrow/focus)
B = brand new search or count request

IMPORTANT: When [Prior action] is summarize_all, summarize, or summarize_selected,
queries like "focus only on X", "now look at just Y", "summarize it more briefly",
"make it shorter", "which files support that summary", "only show me text files"
are ALWAYS A (follow-up refinement of the summary), NOT a new search.

Output A or B:"""

INTENT_CLASSIFIER_PROMPT = INTENT_CLASSIFIER_PROMPT_EN
FOLLOWUP_CLASSIFIER_PROMPT = FOLLOWUP_CLASSIFIER_PROMPT_EN

# Patch the string placeholders in _PROMPT_TEMPLATES with real prompt objects
_PROMPT_TEMPLATES["zh"]["INTENT_CLASSIFIER_PROMPT"] = INTENT_CLASSIFIER_PROMPT_EN
_PROMPT_TEMPLATES["zh"]["FOLLOWUP_CLASSIFIER_PROMPT"] = FOLLOWUP_CLASSIFIER_PROMPT_EN
_PROMPT_TEMPLATES["en"]["INTENT_CLASSIFIER_PROMPT"] = INTENT_CLASSIFIER_PROMPT_EN
_PROMPT_TEMPLATES["en"]["FOLLOWUP_CLASSIFIER_PROMPT"] = FOLLOWUP_CLASSIFIER_PROMPT_EN


def normalize_prompt_language(language: Optional[str], fallback: str = "en") -> str:
    lang = str(language or "").strip().lower()
    if lang.startswith("zh"):
        return "zh"
    if lang.startswith("en"):
        return "en"
    return fallback if fallback in SUPPORTED_PROMPT_LANGUAGES else "en"


def get_prompt(prompt_name: str, language: Optional[str] = None) -> str:
    lang = normalize_prompt_language(language, fallback="en")
    by_lang = _PROMPT_TEMPLATES.get(lang) or _PROMPT_TEMPLATES["en"]
    if prompt_name in by_lang:
        return by_lang[prompt_name]
    return _PROMPT_TEMPLATES["en"].get(prompt_name, "")
