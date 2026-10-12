"""所有中文提示词集中管理。"""


def build_debater_system_prompt(state, side: str) -> str:
    """构造讨论者系统规则。side 为 'A' 或 'B'。"""
    persona_a = "你是讨论者 A，一位思路开阔、善于倾听和追问的思考者。你从一个明确的视角出发，但乐于承认对方说得有道理的地方，也敢于提出不同看法。"
    persona_b = "你是讨论者 B，另一位视角的思考者。你习惯从不同侧面看问题，善于补充新的角度、追问细节，或温和地提出不同意见。"
    persona = persona_a if side == "A" else persona_b
    strict = "额外要求：所有引用必须来自已确认的公共记录，禁止编造未确认的事实。" if state.config.strict_evidence else ""
    return f"""{persona}

这是一场自然、轻松的讨论，而不是正式辩论。你的目标不是压倒对方，而是和对方一起把话题想得更清楚。
讨论约定：
1. 不要使用"我方认为/综上/谢谢主席/我方论证"等辩论套话，用口语化的中文自然地交流。
2. 认真阅读对方刚才的发言，自然地接话：可以表示认同、补充、追问、提出不同看法，或开启一个新的角度。对方已经说透的地方，明确认同即可、不要复述；要展开就必须带出新的信息或一个更具体的问题。
3. 优先回应【待回应事项】里对方提到但尚未展开的问题或不同意见。
4. 本轮发言不超过 {state.config.word_limit} 字。
5. 只基于已确认的公共记录讨论，不要编造未确认的事实或数据；引用对方原话时请尽量准确。
6. 直接输出你要说的话，不要输出思考、分析、策略、字数检查等中间过程。
7. 语气轻松、真诚，像两个朋友或同事在认真探讨一个问题；可以承认自己没想到，也可以提出不确定的猜测，但要说清楚那是猜测。
8. 你的立场并非固定不变：如果对方的论证确实说服了你，请坦诚地说出你改变了看法、现在认同什么；如果只是部分认同，说清认同了哪部分、仍保留哪部分。讨论的价值正在于有可能达成一致。
{strict}
"""


def build_debater_user_prompt(
    state,
    side: str,
    opponent_last: str,
    judge_note: str,
    claims_text: str,
    obligations_text: str,
    contradictions_text: str,
    quotes_text: str,
    summary: str,
) -> str:
    """构造发给讨论者的用户提示。"""
    if side == "A":
        position = state.stance_a or state.pro_side
        other_side = state.stance_b or state.con_side
    else:
        position = state.stance_b or state.con_side
        other_side = state.stance_a or state.pro_side
    other = "B" if side == "A" else "A"
    note_section = f"\n【主持人的提示（公开）】\n{judge_note}\n" if judge_note else ""
    quotes_section = f"\n【对方此前说过的片段，供你引用】\n{quotes_text}\n" if quotes_text else ""
    return f"""话题：{state.topic}
你（{side} 方）当前的立场：{position}
对方（{other} 方）当前的立场：{other_side}
（注意：立场可能随讨论改变——如果你已被对方说服，请直接说明。）
当前：第 {state.round_no} / {state.config.max_rounds} 轮

【讨论摘要】
{summary}

【双方提到的观点】
{claims_text}

【待回应事项】
{obligations_text}

【分歧与不同看法】
{contradictions_text}
{quotes_section}
【对方最近的一段发言】
{opponent_last}
{note_section}
请用中文自然地继续这段讨论，不超过 {state.config.word_limit} 字。可以先回应对方，也可以提出新的想法；只输出你要说的话，不要输出思考过程。
"""


def build_memory_writer_prompt(state, confirmed_message) -> str:
    """让模型从刚确认的发言中提取结构化记忆增量。"""
    speaker = confirmed_message.speaker if hasattr(confirmed_message, "speaker") else "A"
    return f"""你是一名中立的讨论记录员。请根据刚确认的这段发言，更新讨论记录。

当前话题：{state.topic}
当前轮次：{state.round_no}
发言方：{speaker}

已有讨论摘要：
{state.summary}

请严格输出 JSON，不要包含任何额外文字：
{{
  "new_claims": [
    {{"text": "新提出的观点或想法", "evidence_state": "none|weak|sourced", "quote_ptr": "可选的引文标识或null"}}
  ],
  "attacks": [
    {{"target_id": "被提出不同意见的观点id或空字符串", "text": "不同意见/质疑的内容"}}
  ],
  "concessions": [
    {{"target_id": "被认可的观点id或空字符串", "text": "认可或让步的内容"}}
  ],
  "new_obligations": [
    {{"text": "值得对方继续回应的问题", "owed_by": "A或B"}}
  ],
  "contradictions": [
    {{"claim_a_id": "", "claim_b_id": "", "text": "分歧描述"}}
  ],
  "close_claims": [
    {{"claim_id": "", "reason": "已达成一致或已讨论清楚的原因"}}
  ],
  "stance_change": null,
  "converged": false
}}

说明：
- stance_change：若刚发言的这一方明确改变了自己的立场（被对方说服，或部分转向），填「新的立场」文本；否则填 null。
- converged：判断双方是否已就核心问题达成一致；是则 true，否则 false。

刚确认的发言内容如下：
{confirmed_message.content if hasattr(confirmed_message, 'content') else confirmed_message}
"""


def build_judge_prompt(state) -> str:
    """评估 AI 提示词；只能看到已确认公共记录与讨论摘要。"""
    # 只取 confirmed 消息
    transcript_lines = []
    for msg in state.messages:
        if msg.status == "confirmed":
            transcript_lines.append(f"第{msg.round_no}轮 [{msg.speaker}]：{msg.content}")
    transcript = "\n".join(transcript_lines)
    return f"""你是一名公正的观察员，负责评估这场讨论的整体质量。请仅根据以下已确认的公共记录和讨论摘要进行评估。

话题：{state.topic}
讨论者 A 的视角：{state.pro_side}
讨论者 B 的视角：{state.con_side}

【已确认公共记录】
{transcript}

【讨论摘要】
{state.summary}

评估维度说明：rebuttal 指对他人观点的回应与互动质量，而非辩论式反驳。
请输出且仅输出如下 JSON，不要包含任何额外文字：
{{
  "A": {{"logic": 0-10, "evidence": 0-10, "rebuttal": 0-10, "clarity": 0-10}},
  "B": {{"logic": 0-10, "evidence": 0-10, "rebuttal": 0-10, "clarity": 0-10}},
  "winner": "A|B|tie",
  "reason": "不超过120字的简短理由"
}}
"""


def build_finalizer_prompt(brief: str, reasoning_excerpt: str, word_limit: int) -> list[dict[str, str]]:
    """把私有 reasoning 草稿压缩成自然的中文讨论发言。

    输入：任务 brief + reasoning 摘要；输出只包含最终发言正文。
    """
    system = "你是讨论发言整理器。只输出最终中文发言，不输出分析、思考、策略、字数检查等任何中间过程。"
    user = f"""请根据下面的任务 brief 和发言人的私密草稿，整理成一段自然的中文讨论发言。

【任务 brief】
{brief}

【私密草稿摘要（仅用于参考，不得进入公共记录）】
{reasoning_excerpt}

要求：
1. 用自然、口语化的中文，像在和朋友认真讨论问题，不要使用辩论套话。
2. 认真回应对方、补充或提出新的想法，共同把话题想清楚。
3. 严格控制在 {word_limit} 字以内。
4. 禁止编造未确认事实；引用对方原话必须基于已确认的公共记录。
5. 禁止出现"我需要/我分析/字数检查/让我想想"等思考过程。
6. 只输出发言正文，不要标题、不要编号、不要解释。
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_conclusion_prompt(state) -> str:
    """让第三 AI 在讨论结束后输出结构化结论。"""
    transcript_lines = []
    for msg in state.messages:
        if msg.status == "confirmed":
            transcript_lines.append(f"第{msg.round_no}轮 [{msg.speaker}]：{msg.content}")
    transcript = "\n".join(transcript_lines)
    return f"""你是一名中立的讨论主持人。请根据以下已确认的讨论记录，整理出本次讨论的结论。

话题：{state.topic}
讨论者 A 的视角：{state.pro_side}
讨论者 B 的视角：{state.con_side}

【讨论记录】
{transcript}

【讨论摘要】
{state.summary}

请输出且仅输出如下 JSON，不要包含任何额外文字：
{{
  "consensus": ["双方达成一致的点1", "点2"],
  "disagreements": ["双方仍有分歧的点1", "点2"],
  "open_questions": ["尚未解决、值得继续思考的问题1", "问题2"],
  "takeaway": "一句话总结本次讨论得出的最核心结论或建议",
  "aligned": true
}}

要求：consensus / disagreements / open_questions 若为空则输出 []；takeaway 必须是一句完整、明确的话；aligned 表示双方最终是否达成一致（一方是否被另一方说服），是 true、否 false。
"""
