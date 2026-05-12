from __future__ import annotations

import re
import logging
import os
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class ContextFollowupExpert:
    """
    Detects follow-up queries that depend on the previous exchange context.
    
    This is a stateless classifier: it receives the prior action context
    and current query, and determines whether the query is a follow-up.
    """

    # ── Post-count content request patterns ───────────────────────────────
    _POST_COUNT_CONTENT_RE = re.compile(
        r'\b(detail|details|content|contents|explain|describe|'
        r'what.{0,8}(in|say|about|contain|discuss)|'
        r'tell.{0,5}(me|us)|more.{0,5}(about|on|detail)|inside)\b'
        r'|看看|看下|详细|内容|总结|说说|介绍|里面|讲什么|有什么|说什么|看一下|相关内容',
        re.IGNORECASE,
    )

    # ── Guard: new scope search ───────────────────────────────────────────
    # NOTE: Python's \b is ASCII-only; it silently fails at CJK char boundaries.
    # ASCII parts use \b; CJK parts have NO outer \b.
    _NEW_SCOPE_RE = re.compile(
        # ASCII part — \b works correctly
        # "find X" matches multi-word entities but excludes pronoun/demonstrative targets
        # (find it / find them / find the document → these are follow-up references, not new searches)
        r'\b(find\s+(?!it\b|its\b|them\b|this\b|that\b|those\b|these\b|him\b|her\b|the\s+(file|document|result|same|above|previous|last)\b).{2,}|search\s+(for|the|all|again)|look\s+for|'
        r'another|different|new\s+(query|search)|start\s+over|ignore\s+previous|from\s+scratch|global\s+search|'
        r'all\s+(of\s+)?my|every\s+file|everything\s+(I|i)\s+have|what\s+are\s+my|'
        # inventory expansion around the previous result should become a NEW scoped/global search,
        # not a continuation on the same file's content.
        r'(what|which)\s+other\b.*\b(do\s+(I|i)\s+have|are\s+there)\b|'
        r'\b(do\s+(I|i)\s+have|are\s+there)\b.*\b(other|more)\b.*\b(like|similar\s+to)\b|'
        r'\b(like|similar\s+to)\s+(this|that|it|them)\b.*\b(do\s+(I|i)\s+have|are\s+there|(what|which)\s+other)\b)\b'
        r'|'
        # CJK part — NO outer \b
        # Generic reset / new-scope phrases in Chinese:
        r'其他的|新的搜索|另外搜|帮我找|帮我搜|帮我重新|重新(查找|寻找|搜索|搜)|全局(查找|搜索|搜)|之前.*不要|所有.*(文件|文档|资料|照片|简历)'
        r'|还有(哪些|什么).*(类似|相似)|我还有(哪些|什么).*(类似|相似)|还有什么.*(像这|像它|类似)'
        r'|'
        # Uses non-greedy lookahead: at least 2 CJK chars after action verb (= entity name)
        r'^(找|搜|查|找下|找一下|搜一下|搜下|查一下|查下)[^\s，,。！!？?]{2,}'
        r'|'
        r'(总结|概括|归纳|统计|梳理|盘点).{0,6}(我的?|我所有的?|全部|所有|每个|每份|整体|全局|所有文件|全部文件)',
        re.IGNORECASE,
    )

    # ── Guard: global scope signal after summarize verbs ─────────────────────
    # not a follow-up on the previous search results.
    _GLOBAL_SCOPE_SUMMARIZE_RE = re.compile(
        r'(总结|概括|归纳|统计|梳理|盘点|summarize|recap|overview).{0,15}'
        r'(所有|全部|全体|整体|全局|我(的)?所有|我(的)?全部|每个|每份|all\s+my|all\s+of\s+my|every)',
        re.IGNORECASE,
    )

    # Fires when the query starts with a search verb followed by a named entity
    # that is NOT a pronoun/demonstrative. This prevents named-entity searches and
    # "find John's resume" from being absorbed as process_previous follow-ups.
    _SEARCH_VERB_ENTITY_RE = re.compile(
        r'^(找|搜|查|帮我找|帮我搜|帮我查|找找|找下|搜下|查下|找一下|搜一下|查一下)'
        r'(?!他|她|它|这|那|这个|那个|上面|上述|以上|之前)'
        r'|'
        # English: find/search for/look for/get me/show me at sentence start,
        # NOT followed by pronouns/demonstratives referring to previous results.
        r'^\b(find|search\s+for|look\s+for|get\s+me|show\s+me|fetch|retrieve)\b'
        r'(?!\s+(it\b|its\b|this\b|that\b|them\b|those\b|these\b|him\b|her\b|his\b|their\b'
        r'|the\s+(file|document|result|above|previous|last|same)))',
        re.IGNORECASE,
    )

    # ── Guard: re-count ───────────────────────────────────────────────────
    # NOTE: Python's \b is ASCII-only and silently fails at CJK char boundaries.
    # Split into ASCII part (with \b) + CJK part (without \b) joined by |.
    _RECOUNT_RE = re.compile(
        # ASCII part — \b works correctly here
        r'\b(how\s+many\s*(files|documents|docs|resumes|photos|papers|products|items|of\s+them|of\s+these|are)|'
        r'list\s+(all|me|the|these|those)\s*(files|documents|items)|count\b|'
        # Concrete file/item nouns only — avoids catching 'what experience does he have'
        r'what\s+(files|documents|docs|items|resumes|photos|invoices|reports|papers|contracts|api\s*keys?|passwords?)\s*do\s+(I|i)\s+have|'
        r'which\s+(files|documents|docs|items|resumes|photos|invoices|reports|papers|contracts)\s*do\s+(I|i)\s+have|'
        r'what\s+are\s+all\s+my|among.*how\s+many|of\s+them.*how\s+many|of\s+these.*how\s+many)\b'
        r'|'
        # CJK part — NO outer \b (CJK chars have no ASCII word boundaries)
        r'(?!这几个|那几个|这些|那些)(这|其中)?(有)(多少|多少份|几个|几份)(文件|文档|简历|资料|照片|的)|'
        r'其中.*(多少|几个)|(一共有|我共有).*(哪些|几)|'
        r'有哪些(文件|内容|文档)|我(的)?.{0,5}有哪些',
        re.IGNORECASE,
    )

    # ── Guard: personal attribute lookups MUST go to search, never continuation ──
    # Matches queries that ask for a specific person's contact/profile attribute.
    # Works for ANY name in ANY language — the pattern is based on attribute words,
    # NOT on hardcoded names. Supports:
    _ATTR_LOOKUP_RE = re.compile(
        # English: possessive + attribute noun
        r"'s\s*(phone|email|e-mail|address|home\s+address|residence|contact|mobile|tel|salary|birthday|number|password|id)|"
        r'\b(his|her|their|its)\s+(phone|email|e-mail|mobile|telephone|contact|address|home|residence)\b|'
        r'\b(phone\s*(number)?|email(\s+address)?|e-mail|mailing\s+address|street\s+address|work\s+address|office\s+address|'
        r'home\s+address|residence|residential\s+address|'
        r'mobile\s+(number|phone)|telephone|contact\s*(info|details?)|salary|birthday|date\s+of\s+birth|'
        r'school|university|college|alma\s+mater|education|degree|major|graduat(?:e|ed|ion))\b|'
        r'\b(his|her|their|its)\s+(home|residence|residential\s+address)\b|'
        r'\b(?:where\s+(?:does|do|is|are)|where).{0,24}\b(?:he|she|they|his|her|their).{0,24}'
        r'(?:live|lives|reside|resides|home|address|location)\b|'
        r'的\s*(电话|手机|邮箱|地址|家庭住址|住址|居住地|居住地址|住所|住宅|联系方式|联系电话|邮件地址|生日|工资|薪资|薪水|密码|账号|id|工号|'
        r'毕业院校|毕业学校|学校|大学|高校|学历|学位|专业|公司|单位|雇主|职位|职务|岗位)|'
        # Use lookahead to ensure it follows at least one CJK char (= a name)
        r'(?<=[一-鿿])(电话|手机|邮箱|地址|家庭住址|住址|居住地|联系方式|毕业院校|毕业学校|学历|学位|专业)|'
        # Chinese standalone attribute questions (works regardless of preceding name)
        r'(电话|手机号|邮箱|地址|家庭住址|住址|居住地|联系方式|电话号码|位置|所在地)(是多少|是什么|是哪个|号码|是哪|在哪里|在哪)|'
        r'(?:他|她|这个人|这位|该候选人|候选人).{0,12}(哪个学校|什么学校|毕业于|毕业院校|毕业学校|学历|学位|专业|哪家公司|什么公司|职位|职务|岗位)|'
        r'(?:哪个学校|什么学校|毕业院校|毕业学校|学历|学位|专业|哪家公司|什么公司|职位|职务|岗位).{0,12}(他|她|这个人|这位|候选人)|'
        r'(?:他|她|这个人|这位|该候选人|候选人).{0,12}(住哪|住在哪里|住址|家庭住址|居住地|地址|家在哪|家在哪里)|'
        r'(?:住哪|住在哪里|住址|家庭住址|居住地|地址|家在哪|家在哪里).{0,12}(他|她|这个人|这位|候选人)|'
        r'\b(location|work\s+location|office\s+location)\b',
        re.IGNORECASE,
    )

    # ── Tier-1: Strong pronoun / demonstrative references ─────────────────
    # High-confidence continuation signal: direct pronoun or demonstrative reference
    # to a previous entity. Triggers even without a topic verb.
    _PRONOUN_FOLLOWUP_RE = re.compile(
        r'\b(he|she|it|they|his|her|its|their|him|them|this person|these|those)\b|'
        r'\b(the|this|that)\s+(company|candidate|author|paper|report|document|file|resume|person|brand|product)\b|'
        r'(?<![其吉维排利])(他|她|它|他们|她们|它们|这些|那些|这批|那批|这几个|这几份|这几张|这篇|这份|那篇|那份|这几|那个|这个人|这人)|'
        r'(这个|这家|该|那家|那个|这位|那位|该名)(候选人|公司|人|作者|论文|报告|文档|文件|简历|先生|女士|经理|总监|品牌|产品)',
        re.IGNORECASE,
    )

    _ORDINAL_FOLLOWUP_RE = re.compile(
        r'\b(?:the\s+)?(?:first|second|third|fourth|fifth|last|\d+(?:st|nd|rd|th)|#\s*\d+|number\s+\d+|item\s+\d+)\s*'
        r'(?:one|file|document|doc|result|video|audio|image|photo|paper|report|item)?\b'
        r'|第\s*(?:一|二|三|四|五|六|七|八|九|十|\d+)\s*(?:个|份|篇|张|条)?\s*'
        r'(?:文件|文档|结果|视频|音频|图片|照片|论文|报告|项目)?',
        re.IGNORECASE,
    )

    _COMPARATIVE_FOLLOWUP_RE = re.compile(
        r'^\s*(?:compared\s+(?:to|with)|relative\s+to|in\s+comparison\s+(?:to|with)|versus|vs\.?)\b'
        r'|相比|对比|比较一下|比起',
        re.IGNORECASE,
    )

    _COMPOUND_COUNT_QA_RE = re.compile(
        r'\bhow\s+many\b.{0,80}\b(?:listed|shown|included|mentioned|covered)\b.{0,80}\b(?:and|,)\s*(?:what|which|who|where|how)\b'
        r'|\bhow\s+many\b.{0,80}\b(?:and|,)\s*(?:what|which)\b.{0,80}\b(?:cover|categories|types|topics|fields|columns)\b'
        r'|(?:列出|提到|包含|覆盖).{0,30}(?:多少|几个|几类).{0,30}(?:以及|和|并且).{0,30}(?:哪些|什么)',
        re.IGNORECASE,
    )
    _SCOPED_CONTENT_COUNT_RE = re.compile(
        r'\bhow\s+many\s+(?!(?:files?|documents?|docs?|folders?|images?|photos?|pictures?|videos?|audios?|'
        r'recordings?|resumes?|reports?|papers?|contracts?|invoices?)\b)'
        r'[a-z][a-z0-9_-]{1,40}\b.{0,60}\b(?:remain(?:ing)?|left|listed|mentioned|included|covered|available|are\s+there)\b'
        r'|(?:里面|其中|上面|上述|前面|这些|那些).{0,24}'
        r'(?:一共|总共|共有|有|包含|列出|提到).{0,12}'
        r'(?:多少|几个|几条|几项|几种|几类).{0,16}'
        r'(?:岗位|职位|角色|模型|型号|产品|芯片|公司|候选人|记录|条目|行|列|字段|指标)',
        re.IGNORECASE,
    )
    _FOCUS_REFINE_RE = re.compile(
        r'^\s*(?:now\s+)?(?:focus|narrow|filter|limit)\s+'
        r'(?:only\s+)?(?:on|to|down\s+to)\b'
        r'|^\s*(?:only|just)\s+(?:show|use|keep|include)\b'
        r'|只看|只关注|聚焦|缩小到|限定到|筛选到',
        re.IGNORECASE,
    )
    _SUMMARY_SUPPORT_FOLLOWUP_RE = re.compile(
        r'\bwhich\s+files?\b.{0,80}\b(?:support|back\s+up|justify|evidence)\b.{0,80}\b(?:that|this|the)?\s*(?:summary|answer|conclusion|analysis)\b'
        r'|\b(?:supporting|source|evidence)\s+files?\b.{0,80}\b(?:that|this|the)?\s*(?:summary|answer|conclusion|analysis)\b'
        r'|(?:哪些|哪个).{0,20}(?:文件|文档|资料).{0,40}(?:支持|证明|支撑).{0,20}(?:总结|结论|回答|分析)',
        re.IGNORECASE,
    )
    _SCOPED_METADATA_QA_RE = re.compile(
        r"^\s*(?:who\s+(?:is|was)\s+(?:the\s+)?(?:author|writer|creator|owner)|"
        r"who\s+(?:wrote|created|authored)\b|"
        r"what\s+(?:is|was)\s+(?:the\s+)?(?:author|writer|creator|owner|title|date|version|value)|"
        r"(?:author|writer|creator|owner|title|date|version|value)\s*(?:is|was|:)?).{0,48}[?？]?\s*$"
        r"|^\s*(?:作者|撰写者|编写者|创建者|创建人|负责人|标题|题目|日期|时间|版本|数值|值)"
        r".{0,16}(?:是谁|是什么|是多少|在哪|是哪个|哪个|多少)?[?？]?\s*$"
        r"|^\s*(?:谁是|谁写的|谁创建的|谁编写的).{0,24}[?？]?\s*$"
        r"|^\s*(?:表格|sheet|工作表)?.{0,12}(?:第\s*\d+\s*(?:行|row)|row\s*\d+).{0,24}"
        r"(?:第\s*\d+\s*(?:列|col|column)|col(?:umn)?\s*\d+).{0,16}(?:是什么|是多少|value|数据)?[?？]?\s*$",
        re.IGNORECASE,
    )
    _SCOPED_MENTION_QA_RE = re.compile(
        r"\b(?:was|were|is|are|did|does|do)\b.{0,80}\b(?:mention(?:ed|s)?|list(?:ed|s)?|include(?:d|s)?|cover(?:ed|s)?|state(?:d|s)?|note(?:d|s)?|say|said|report(?:ed|s)?)\b"
        r"|\b(?:mention(?:ed|s)?|list(?:ed|s)?|include(?:d|s)?|cover(?:ed|s)?|state(?:d|s)?|note(?:d|s)?|reported)\b.{0,80}[?？]\s*$"
        r"|(?:是否|有没有|有无|哪里|哪儿|什么|哪些|哪个|哪一个).{0,40}(?:提到|提及|写到|说到|讲到|列出|包含|包括|覆盖|说明|显示)",
        re.IGNORECASE,
    )
    _SCOPED_COMPARISON_RE = re.compile(
        r"\b(?:which\s+(?:one|candidate|resume|report|file|document|item)|"
        r"which\s+of\s+(?:these|those|them|the\s+above)|who)\b"
        r".{0,80}\b(?:best|better|most|least)\b.{0,40}"
        r"\b(?:fit|fits|match(?:es|ing)?|suit(?:s|ed)?|align(?:s|ed)?|relevant)\b"
        r"|(?:这几份|这些|那几份|那些|其中|上面|上述|前面|这批|那批).{0,24}"
        r"(?:哪一份|哪份|哪个|谁).{0,24}(?:最|更).{0,8}(?:匹配|适合|符合|契合|对应)"
        r"|(?:哪一份|哪份|哪个|谁).{0,24}(?:最|更).{0,8}(?:匹配|适合|符合|契合|对应)",
        re.IGNORECASE,
    )
    _CONTEXT_GENERATION_RE = re.compile(
        r"\b(?:based\s+on|from|using|with|according\s+to)\s+(?:this|that|these|those|it|them|the\s+above|previous|prior|last|results?|files?|documents?)\b"
        r".{0,80}\b(?:write|draft|compose|create|generate|prepare|make|draw|plot|chart|visuali[sz]e|turn)\b"
        r"|\b(?:write|draft|compose|create|generate|prepare|make|draw|plot|chart|visuali[sz]e|turn)\b"
        r".{0,80}\b(?:email|e-mail|letter|message|reply|speech|talking\s+points?|script|ppt|presentation|slides?|deck|memo|outline|report|chart|pie\s+chart|bar\s+chart|graph|visuali[sz]ation)\b"
        r".{0,80}\b(?:this|that|these|those|it|them|above|previous|prior|results?|files?|documents?)?\b"
        r"|(?:基于|根据|参考|按照|按|用).{0,24}(?:这个|这份|这些|这几份|它|它们|上述|上面|前面|之前|结果|文件|内容|报告|文档).{0,60}(?:写|撰写|起草|生成|整理|制作|做|画|绘制|输出)"
        r"|(?:写|撰写|起草|生成|整理|制作|做|画|绘制|输出).{0,60}(?:邮件|信|回复|讲稿|演讲稿|ppt|PPT|幻灯片|汇报|报告|提纲|大纲|饼图|柱状图|图表|可视化)",
        re.IGNORECASE,
    )
    _CONTINUE_FOLLOWUP_RE = re.compile(
        r"^\s*(?:and\s+then|then\s+what|what\s+next|next|go\s+on|continue|continue\s+that)\s*[?？。.!]*\s*$"
        r"|^\s*(?:然后呢|接着呢|后续呢|然后|接着|继续|继续说|再说说|往下说)\s*[?？。.!]*\s*$",
        re.IGNORECASE,
    )

    _SHORT_FRAGMENT_FOLLOWUP_RE = re.compile(
        r'^\s*(?:any|which|what|who|where|when|why|how|does|do|did|is|are|can|could|should)\b.{0,80}\?\s*$'
        r'|^\s*[A-Za-z][A-Za-z0-9 /&,+-]{1,48}\?\s*$'
        r'|^\s*(?:还有|是否|有没有|哪个|哪些|什么|怎么|如何).{0,40}[？?]?\s*$',
        re.IGNORECASE,
    )
    _GENERAL_CHAT_SHORT_RE = re.compile(
        r'^\s*(?:what\s+time\s+is\s+it|who\s+are\s+you|how\s+are\s+you|hello|hi|hey|thanks?|thank\s+you)\b'
        r'|^\s*(?:现在几点|你是谁|你好|谢谢)\s*[？?。.!]*\s*$',
        re.IGNORECASE,
    )

    # ── Tier-2: Short topic continuation markers ──────────────────────────
    # Lower-confidence: these words alone do NOT trigger continuation.
    # Only fire when BOTH: (a) word_count ≤ 8, AND (b) no _ATTR_LOOKUP_RE hit.
    _TOPIC_FOLLOWUP_SHORT_RE = re.compile(
        r'\b(mainly|specifically|how|where|when|why|who|which|'
        r'compared|between|regarding|responsible\s+for|experience|background|difference|'
        r'elaborate|expand|further|more\s+about|key\s+point|takeaway)\b|'
        r'(主要|具体|详细|核心|关键|重点|怎么样|怎样|如何|大概|大约|是什么|'
        r'掌握|负责|从事|用了什么|是不是|哪位|哪些|哪个|怎么|区别|对比|相比)',
        re.IGNORECASE,
    )

    # ── Post-content deeper dive keywords ─────────────────────────────────
    _DEEPER_KWS = [
        "more", "deeper", "elaborate", "expand", "further", "continue", "go on",
        "详细", "展开", "更多", "继续", "深入", "补充", "再多说", "继续说",
    ]
    _META_SUMMARY_RE = re.compile(
        r'\b(summary|summarize|overview|recap|conclusion|key\s+takeaways?|takeaways?)\b'
        r'|总结|概括|归纳|汇总|结论|要点',
        re.IGNORECASE,
    )

    # ── Implicit file reference patterns ──────────────────────────────────
    _IMPLICIT_PATTERNS = [
        # English: pronoun + content verb
        re.compile(r"what.{0,12}(it say|it contain|it about|it discuss|in it|does it|it cover)", re.IGNORECASE),
        re.compile(r"\btell me (about|more about) (it|this|that|them)\b", re.IGNORECASE),
        re.compile(r"\bshow me (the\s+)?(content|what.{0,5}(in|inside)|what it (say|contain))", re.IGNORECASE),
        re.compile(r"\bsummarize (this|it|them|that)\b", re.IGNORECASE),
        re.compile(r"what.{0,6}(this|the) (file|document|doc).{0,10}(about|say|contain|cover)", re.IGNORECASE),
        re.compile(r"\b(it|its|this|that).{0,20}\b(about|contain|say|discuss|cover|describe)\b", re.IGNORECASE),
        re.compile(r"what('s|\s+is)\s+in (it|this|the file|the document)", re.IGNORECASE),
        re.compile(r"\bwhat does (it|this|the file|the document) (say|contain|cover|describe|discuss)", re.IGNORECASE),
        # Chinese: implicit pronouns
        re.compile(r"(它|这个|这份|该)(文件|文档|资料)?(里面|内容|讲|说|是什么|是关于|有什么)"),
        re.compile(r"(里面|内容)(说了什么|讲了什么|是什么|有什么|有哪些)"),
        re.compile(r"(这|那)(篇|个|份).{0,6}(讲|说|写|介绍|关于|是)"),
        re.compile(r"说了什么|讲了什么|是关于什么"),
    ]

    _COUNT_GUARD_RE = _RECOUNT_RE
    _CONTENT_QUESTION_RE = re.compile(
        r"^(what|how|which|why|when|where|does|do|did|is|are|can|could|would|will)\b|"
        r"^(这|那|它|该|这个|那个|这份|那份|怎么|如何|为什么|为何|是否|有没有|能否|会不会|"
        r"哪个|哪一个|哪款|哪一款|哪些|什么|谁|讲了什么|说了什么|内容是啥|内容是什么)",
        re.IGNORECASE,
    )

    _FILE_TYPE_HINTS = (
        (
            "sheet",
            re.compile(r"\b(spreadsheet|sheet|worksheet|excel|xlsx|xls|csv)\b|表格|工作表", re.IGNORECASE),
            {".xls", ".xlsx", ".csv", ".tsv"},
        ),
        (
            "pdf",
            re.compile(r"\bpdfs?\b|PDF", re.IGNORECASE),
            {".pdf"},
        ),
        (
            "image",
            re.compile(r"\b(images?|photos?|pictures?|screenshots?|png|jpe?g|webp)\b|图片|照片|截图|图像", re.IGNORECASE),
            {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"},
        ),
        (
            "video",
            re.compile(r"\b(videos?|clips?|footage|mp4|mov|mkv|webm)\b|视频|录像|片段", re.IGNORECASE),
            {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"},
        ),
        (
            "audio",
            re.compile(r"\b(audios?|recordings?|voice\s+notes?|mp3|wav|m4a|flac)\b|音频|录音|语音", re.IGNORECASE),
            {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"},
        ),
        (
            "text",
            re.compile(r"\b(markdown|md|text\s+file|txt|plain\s+text)\b|文本文件|markdown", re.IGNORECASE),
            {".md", ".txt", ".rtf"},
        ),
    )

    # Media prior context should only switch to media_content_search when the
    # current follow-up is itself about media evidence. Ordinary topic questions
    # after a media-looking search result are safer as scoped process_previous.
    _MEDIA_FOLLOWUP_SIGNAL_RE = re.compile(
        r"\b(transcript|asr|subtitle|caption|keyframe|frame|scene|timestamp|timecode|"
        r"footage|speaker|voice|heard|hear|said|say|says|saying|"
        r"mention(?:s|ed|ing)?|discuss(?:es|ed|ing)?|talking|speaking|"
        r"visual|screen|shown|appears?|happens?|watch|see)\b"
        r"|第\s*\d+\s*(?:秒|分|分钟)|(?<![A-Za-z])\d+\s*(?:seconds?|secs?|s|minutes?|mins?|m)\b"
        r"|字幕|转写|帧|关键帧|画面|场景|镜头|"
        r"说了|在说|讲了|在讲|提到|讲到|说到|提及|听到|看到|出现|第几秒|第几分",
        re.IGNORECASE,
    )

    @classmethod
    def _query_file_type_hints(cls, query: str) -> set:
        text = str(query or "")
        hints = set()
        for label, pattern, _extensions in cls._FILE_TYPE_HINTS:
            if pattern.search(text):
                hints.add(label)
        return hints

    @classmethod
    def _result_file_type_hints(cls, last_results: Optional[List[dict]]) -> set:
        hints = set()
        for item in list(last_results or []):
            if not isinstance(item, dict):
                continue
            path = str(
                item.get("file_path")
                or item.get("path")
                or item.get("file_name")
                or item.get("name")
                or ""
            ).strip()
            icon_type = str(item.get("iconType") or item.get("type") or "").strip().lower()
            if icon_type in {"sheet", "pdf", "image", "video", "audio"}:
                hints.add(icon_type)
            ext = os.path.splitext(path.lower())[1]
            if not ext:
                continue
            for label, _pattern, extensions in cls._FILE_TYPE_HINTS:
                if ext in extensions:
                    hints.add(label)
        return hints

    @classmethod
    def _media_or_process_followup(
        cls,
        query: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        confidence: float = 0.9,
        log_prefix: str = "media followup",
    ) -> Dict[str, Any]:
        from core.intent.media_query_expert import MediaQueryExpert

        normalized = MediaQueryExpert._normalize_time_query(str(query or "")).lower()
        has_time_signal = bool(MediaQueryExpert._HAS_TIME_SIGNAL.search(normalized))
        if has_time_signal and MediaQueryExpert._looks_like_calendar_year_reference(normalized):
            has_time_signal = False

        if has_time_signal:
            media_params = dict(params or {})
            detected = MediaQueryExpert.analyze(query, llm_service=None) or {}
            if str(detected.get("action") or "") == "media_export":
                media_params.update(dict(detected.get("params") or {}))
            else:
                time_sec = MediaQueryExpert._extract_time(normalized)
                time_end = MediaQueryExpert._extract_time_range_end(normalized)
                if time_sec is not None:
                    media_params.setdefault("time_sec", float(time_sec))
                if time_end is not None:
                    media_params.setdefault("time_end_sec", float(time_end))
                media_params.setdefault("target_type", MediaQueryExpert._detect_target_type(normalized))
                media_params.setdefault(
                    "sub_intent",
                    "range_summary" if time_end is not None else "point_lookup",
                )
            media_params.setdefault("query", query)
            logger.debug(f"[context_followup] {log_prefix} time signal -> media_export query_chars={len(query or '')}")
            return {"action": "media_export", "params": media_params, "confidence": max(confidence, 0.95)}

        if MediaQueryExpert.looks_like_duration_query(query):
            media_params = dict(params or {})
            file_hint = str(media_params.get("file_hint") or "").strip()
            augmented_query = query
            if file_hint and file_hint.lower() not in str(query or "").lower():
                augmented_query = f"{query} in {file_hint}"
            detected = MediaQueryExpert.analyze(augmented_query, llm_service=None) or {}
            if str(detected.get("action") or "") == "media_export":
                media_params.update(dict(detected.get("params") or {}))
                media_params["query"] = query
                if file_hint:
                    media_params.setdefault("file_hint", file_hint)
                logger.debug(f"[context_followup] {log_prefix} duration -> media_export query_chars={len(query or '')}")
                return {"action": "media_export", "params": media_params, "confidence": max(confidence, 0.95)}

        if cls._MEDIA_FOLLOWUP_SIGNAL_RE.search(str(query or "")):
            media_params = dict(params or {})
            media_params.setdefault("query", query)
            logger.debug(f"[context_followup] {log_prefix} -> media_content_search query_chars={len(query or '')}")
            return {"action": "media_content_search", "params": media_params, "confidence": confidence}
        logger.debug(f"[context_followup] {log_prefix} without media evidence -> process_previous query_chars={len(query or '')}")
        return {"action": "process_previous", "params": {}, "confidence": min(confidence, 0.9)}

    @classmethod
    def analyze_context_followup(
        cls,
        query: str,
        prior_context: dict,
        *,
        last_results: Optional[List[dict]] = None,
        active_paths: Optional[List[str]] = None,
    ) -> Optional[dict]:
        """
        Check if the query is a context-dependent follow-up.
        
        Args:
            query: Current user query
            prior_context: Output from IntentAnalyzer._extract_prior_action_context()
            last_results: Previous search/count results
            active_paths: Currently selected file paths
            
        Returns:
            Intent dict if follow-up detected, None otherwise.
        """
        ql = (query or "").lower()
        word_count = len([w for w in ql.split() if w])
        if cls._GENERAL_CHAT_SHORT_RE.search(ql):
            return None
        has_prior_context = bool(last_results) or bool(
            prior_context.get("prior_was_content")
            or prior_context.get("prior_was_search")
            or prior_context.get("prior_was_count")
            or prior_context.get("focused_file")
        )
        has_any_history_context = has_prior_context or bool(prior_context.get("prior_user_query"))

        query_file_type_hints = cls._query_file_type_hints(query)
        if last_results and query_file_type_hints:
            prior_file_type_hints = cls._result_file_type_hints(last_results)
            if prior_file_type_hints and not query_file_type_hints.intersection(prior_file_type_hints):
                logger.debug(
                    "[context_followup] explicit file-type switch guard → search "
                    "query_hints=%s prior_hints=%s query_chars=%d",
                    sorted(query_file_type_hints),
                    sorted(prior_file_type_hints),
                    len(query or ""),
                )
                return None

        # ── Compound count+QA over prior results: keep it scoped ─────────────
        if last_results and cls._COMPOUND_COUNT_QA_RE.search(ql):
            logger.debug(f"[context_followup] compound count+QA over prior results → process_previous query_chars={len(query or '')}")
            return {"action": "process_previous", "params": {}, "confidence": 0.88}

        # ── Scoped content counts over prior results ─────────────────────────
        # Scoped entity-count follow-ups ask about rows/entities inside the
        # previous file set, not about the number of files in the library.
        if last_results and cls._SCOPED_CONTENT_COUNT_RE.search(ql):
            logger.debug(f"[context_followup] scoped content count → process_previous query_chars={len(query or '')}")
            return {"action": "process_previous", "params": {}, "confidence": 0.88}

        # ── Guard 1: Re-count check MUST happen before pronoun check etc. ─────
        if cls._RECOUNT_RE.search(ql):
            return None  # Not a process_previous followup if it's explicitly recounting

        if has_prior_context and cls._CONTEXT_GENERATION_RE.search(query.strip()):
            if prior_context.get("prior_was_media"):
                return cls._media_or_process_followup(
                    query,
                    confidence=0.88,
                    log_prefix="context-bound generation (media prior)",
                )
            logger.debug(f"[context_followup] context-bound generation → process_previous query_chars={len(query or '')}")
            return {"action": "process_previous", "params": {}, "confidence": 0.88}

        if has_prior_context and cls._CONTINUE_FOLLOWUP_RE.search(query.strip()):
            logger.debug(f"[context_followup] continuation prompt → process_previous query_chars={len(query or '')}")
            return {"action": "process_previous", "params": {}, "confidence": 0.86}

        if has_prior_context and cls._SCOPED_MENTION_QA_RE.search(query.strip()):
            if prior_context.get("prior_was_media"):
                return cls._media_or_process_followup(
                    query,
                    confidence=0.86,
                    log_prefix="scoped mentioned/listed question (media prior)",
                )
            logger.debug(f"[context_followup] scoped mentioned/listed question → process_previous query_chars={len(query or '')}")
            return {"action": "process_previous", "params": {}, "confidence": 0.84}

        if has_prior_context and cls._SCOPED_COMPARISON_RE.search(query.strip()):
            if prior_context.get("prior_was_media"):
                return cls._media_or_process_followup(
                    query,
                    confidence=0.87,
                    log_prefix="scoped comparison/ranking question (media prior)",
                )
            logger.debug(f"[context_followup] scoped comparison/ranking question → process_previous query_chars={len(query or '')}")
            return {"action": "process_previous", "params": {}, "confidence": 0.86}

        # ── Guard 0: Personal attribute lookup → ALWAYS search/PersonalInfoDB, never plain continuation ──
        # NOT to process_previous which only scans previous file summaries.
        # correctly sent to PersonalInfoDB rather than absorbed as a follow-up answer.
        if cls._ATTR_LOOKUP_RE.search(ql):
            logger.debug(f"[context_followup] personal attr lookup guard → PersonalInfoDB path query_chars={len(query or '')}")
            return None

        # ── Refinement of the previous answer/result set ────────────────────
        # Phrases like "focus only on..." and "which files support that summary"
        # are not new corpus searches; they are scoped refinements of the
        # immediately previous answer or file set.
        if has_any_history_context and (
            cls._FOCUS_REFINE_RE.search(ql)
            or cls._SUMMARY_SUPPORT_FOLLOWUP_RE.search(ql)
        ):
            logger.debug(f"[context_followup] prior-answer refinement → process_previous query_chars={len(query or '')}")
            return {"action": "process_previous", "params": {}, "confidence": 0.86}

        if has_prior_context and cls._SCOPED_METADATA_QA_RE.search(query.strip()):
            if prior_context.get("prior_was_media"):
                return cls._media_or_process_followup(
                    query,
                    confidence=0.86,
                    log_prefix="scoped metadata/table question (media prior)",
                )
            logger.debug(f"[context_followup] scoped metadata/table question → process_previous query_chars={len(query or '')}")
            return {"action": "process_previous", "params": {}, "confidence": 0.84}

        # ── Tier-1 Pronoun follow-up (highest confidence) ─────────────────
        # A clear pronoun/demonstrative reference to a previous entity, WITHOUT
        # an attribute lookup (handled above), is a strong continuation signal.
        if last_results and cls._PRONOUN_FOLLOWUP_RE.search(ql):
            if prior_context.get("prior_was_media"):
                return cls._media_or_process_followup(
                    query,
                    confidence=0.95,
                    log_prefix="Tier-1 pronoun (media)",
                )
            logger.debug(f"[context_followup] Tier-1 pronoun followup → process_previous query_chars={len(query or '')}")
            return {"action": "process_previous", "params": {}, "confidence": 0.9}

        if last_results and cls._ORDINAL_FOLLOWUP_RE.search(ql):
            if prior_context.get("prior_was_media"):
                return cls._media_or_process_followup(
                    query,
                    confidence=0.9,
                    log_prefix="ordinal result followup (media)",
                )
            logger.debug(f"[context_followup] ordinal result followup → process_previous query_chars={len(query or '')}")
            return {"action": "process_previous", "params": {}, "confidence": 0.88}

        # ── Guard 2: Explicit new scope ───────────────────────────────────────
        if cls._NEW_SCOPE_RE.search(ql):
            logger.debug(f"[context_followup] new scope guard → search query_chars={len(query or '')}")
            return None

        # ── Guard 2.3: Summarize verb + global scope → new summarize/search ──
        if cls._GLOBAL_SCOPE_SUMMARIZE_RE.search(ql):
            logger.debug(f"[context_followup] global-scope summarize guard → search query_chars={len(query or '')}")
            return None

        # ── Guard 2.5: Search verb + distinct entity name → always new search ──
        # even when last_results exist and word_count is small.
        # Rationale: user is clearly naming a NEW target entity, not referring back.
        if cls._SEARCH_VERB_ENTITY_RE.search(ql):
            logger.debug(f"[context_followup] search-verb+entity guard → search query_chars={len(query or '')}")
            return None

        if last_results and cls._COMPARATIVE_FOLLOWUP_RE.search(ql):
            if prior_context.get("prior_was_media"):
                return cls._media_or_process_followup(
                    query,
                    confidence=0.86,
                    log_prefix="comparative followup (media)",
                )
            logger.debug(f"[context_followup] comparative followup → process_previous query_chars={len(query or '')}")
            return {"action": "process_previous", "params": {}, "confidence": 0.84}

        # ── Content question on prior results ─────────────────────────────
        # A full sentence like "how did the team grow from 2015 to now" may
        # contain no pronoun, but after a successful file search it is still a
        # scoped QA follow-up unless a stronger new-scope/search guard fired.
        if (
            has_prior_context
            and cls._CONTENT_QUESTION_RE.search(query.strip())
        ):
            if prior_context.get("prior_was_media"):
                return cls._media_or_process_followup(
                    query,
                    confidence=0.86,
                    log_prefix="content question (media prior)",
                )
            logger.debug(f"[context_followup] content question on prior results → process_previous query_chars={len(query or '')}")
            return {"action": "process_previous", "params": {}, "confidence": 0.84}

        if (
            last_results
            and cls._POST_COUNT_CONTENT_RE.search(ql)
        ):
            if prior_context.get("prior_was_media"):
                return cls._media_or_process_followup(
                    query,
                    confidence=0.86,
                    log_prefix="content request (media prior)",
                )
            logger.debug(f"[context_followup] content request on prior results → process_previous query_chars={len(query or '')}")
            return {"action": "process_previous", "params": {}, "confidence": 0.84}

        if last_results and word_count <= 8 and cls._SHORT_FRAGMENT_FOLLOWUP_RE.search(query.strip()):
            if prior_context.get("prior_was_media"):
                return cls._media_or_process_followup(
                    query,
                    confidence=0.82,
                    log_prefix="short-fragment followup (media)",
                )
            logger.debug(f"[context_followup] short-fragment followup → process_previous query_chars={len(query or '')}")
            return {"action": "process_previous", "params": {}, "confidence": 0.8}

        # ── Weak-result topic follow-up ───────────────────────────────────
        # If the previous turn was a search attempt that failed to keep any
        # scoped files, we still want clearly anchored detail questions to stay
        # on that topic instead of broadening to a brand-new global search.
        if prior_context.get("prior_was_search") and prior_context.get("prior_search_failed") and not last_results:
            has_topic_followup_signal = (
                bool(cls._PRONOUN_FOLLOWUP_RE.search(ql))
                or bool(cls._POST_COUNT_CONTENT_RE.search(ql))
                or any(p.search(ql) for p in cls._IMPLICIT_PATTERNS)
                or bool(cls._CONTENT_QUESTION_RE.search(query.strip()))
                or (word_count <= 8 and bool(cls._TOPIC_FOLLOWUP_SHORT_RE.search(ql)))
            )
            if has_topic_followup_signal:
                logger.debug(f"[context_followup] weak-result topic followup → process_previous query_chars={len(query or '')}")
                return {"action": "process_previous", "params": {}, "confidence": 0.78}

        # ── Tier-2 Short topic marker follow-up (medium confidence) ──────────
        # Only fires for SHORT queries (≤ 8 words) that don't look like attribute
        if last_results and word_count <= 8 and cls._TOPIC_FOLLOWUP_SHORT_RE.search(ql):
            if prior_context.get("prior_was_media"):
                if cls._META_SUMMARY_RE.search(ql):
                    logger.debug(f"[context_followup] Tier-2 topic (media summary) → process_previous query_chars={len(query or '')}")
                    return {"action": "process_previous", "params": {}, "confidence": 0.88}
                return cls._media_or_process_followup(
                    query,
                    confidence=0.9,
                    log_prefix="Tier-2 topic (media)",
                )
            logger.debug(f"[context_followup] Tier-2 topic followup → process_previous query_chars={len(query or '')}")
            return {"action": "process_previous", "params": {}, "confidence": 0.8}

        # ── Post-count follow-up ──────────────────────────────────────────
        if prior_context.get("prior_was_count") and last_results:
            if (word_count <= 10
                    and cls._POST_COUNT_CONTENT_RE.search(ql)):
                if prior_context.get("prior_was_media"):
                    if cls._META_SUMMARY_RE.search(ql):
                        logger.debug(f"[context_followup] post-count (media summary) → process_previous query_chars={len(query or '')}")
                        return {"action": "process_previous", "params": {}, "confidence": 0.92}
                    return cls._media_or_process_followup(
                        query,
                        confidence=0.9,
                        log_prefix="post-count (media)",
                    )
                logger.debug(f"[context_followup] post-count → process_previous query_chars={len(query or '')}")
                return {"action": "process_previous", "params": {}, "confidence": 0.9}

        # ── Post-content deeper dive ──────────────────────────────────────
        if prior_context.get("prior_was_content") and last_results:
            if word_count <= 6 and any(kw in ql for kw in cls._DEEPER_KWS):
                if prior_context.get("prior_was_media"):
                    if cls._META_SUMMARY_RE.search(ql):
                        logger.debug(f"[context_followup] post-media deeper summary → process_previous query_chars={len(query or '')}")
                        return {"action": "process_previous", "params": {}, "confidence": 0.95}
                    return cls._media_or_process_followup(
                        query,
                        confidence=0.95,
                        log_prefix="post-media deeper",
                    )
                logger.debug(f"[context_followup] post-content deeper → process_previous query_chars={len(query or '')}")
                return {"action": "process_previous", "params": {}, "confidence": 0.9}

        # ── Focused File follow-up ────────────────────────────────────────
        # When a single file was the previous result, only treat as follow-up
        # if the query contains a clear content signal — not just "any query".
        if prior_context.get("focused_file"):
            from core.intent.media_query_expert import MediaQueryExpert
            has_content_signal = (
                cls._POST_COUNT_CONTENT_RE.search(ql)
                or cls._PRONOUN_FOLLOWUP_RE.search(ql)
                or any(p.search(ql) for p in cls._IMPLICIT_PATTERNS)
                or MediaQueryExpert.looks_like_duration_query(query)
            )
            if has_content_signal and not cls._COUNT_GUARD_RE.search(ql):
                if prior_context.get("prior_was_media"):
                    return cls._media_or_process_followup(
                        query,
                        params={"file_hint": prior_context["focused_file"]},
                        confidence=0.95,
                        log_prefix=f"focused_file ({prior_context['focused_file']}) (media)",
                    )
                logger.info(
                    f"[context_followup] focused_file ({os.path.basename(str(prior_context['focused_file']))}) "
                    f"+ content signal → process_previous query_chars={len(query or '')}"
                )
                return {"action": "process_previous", "params": {}, "confidence": 0.85}

        # ── Implicit file reference with active_paths ─────────────────────
        if active_paths:
            if (any(p.search(ql) for p in cls._IMPLICIT_PATTERNS)
                    and not cls._COUNT_GUARD_RE.search(ql)):
                logger.info(
                    f"[context_followup] implicit file ref → summarize_all"
                    f" ({len(active_paths)} files) query_chars={len(query or '')}"
                )
                return {"action": "summarize_all", "params": {}, "confidence": 0.85}

        return None
