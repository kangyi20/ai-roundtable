# AI 讨论室

两个 AI 就一个话题**自然讨论**（不是正式辩论），讨论过程中自动维护上下文，最后自动汇总结论。支持**自动讨论**（A/B 轮流发言）与**手动模式**（人工确认/驳回），并实时展示 token 用量与估算成本。

## 功能特性

- **自然讨论而非辩论**：双方 AI 以"讨论者"身份接话——可认同、补充、追问、提出不同看法，共同把话题想清楚；提示词明确禁止辩论套话。
- **自动讨论**：一键让 A/B 轮流发言并自动确认，界面实时刷新，可随时暂停/继续；连续失败会自动熔断暂停。
- **手动模式**：保留 A 发言 → 确认/驳回 → B 发言 的严格回合控制，适合精细把控质量。
- **分层记忆上下文**：每一轮只发给模型「规则/话题/摘要 + 双方观点账本 + 待回应事项 + 分歧 + 对方最近一条已确认发言 + 按需引用片段」，省 token 且不孤立。
- **讨论出结果**：结束时由第三个 AI 生成结构化结论——双方共识 / 主要分歧 / 待解决问题 / 一句话结论；失败自动重试，仍失败可手动重试。
- **内容收敛自动结束**：连续两轮没有新观点/新问题就提前收尾，不必硬跑满轮数。
- **实时用量**：页面实时显示累计输入/输出 token 与估算成本。
- **思考模式隔离**：A/B 可用带思考的模型（如 GLM）；reasoning 只用于内部定稿，绝不进入公共记录/记忆/结论/评分。
- **空/脏稿自动处理**：足量 token 预算 → 剥离 thinking 标签 → finalizer 二次定稿 → 技术重试 1 次 → 仍失败自动驳回。
- **MOCK_MODE**：无需 API Key 即可跑通完整流程。
- **持久化**：每次推进立即追加 `debates/{debate_id}.jsonl`，刷新后可恢复。
- **导出**：生成 `transcript.jsonl`、`score.json`、`config_snapshot.json`、`conclusion.json`。
- **困惑自动提炼**：可以直接粘贴一段说不清立场的生活困惑（例如"手里几千块，该买电脑还是去旅游"）。点「开始讨论」时会**自动**调用裁判提炼出**议题 + 双方立场 + 它理解的你的处境**，回填后给你确认/修改，确认无误才真正开跑（没有多余的"提炼"按钮）。
- **三层信息隔离（防"看穿变成伤人"）**：对提问者的推断分三层存放——**L1 直白推断**（只进画像库，**绝不注入辩论 AI**）、**L2 委婉版**（注入两个辩论 AI，只说明"该怎么和他说话"）、**L3 体贴版**（给你本人看）。辩论 AI 拿不到 L1，所以**没有能力泄露**伤人的判断，而不是"被要求别说"。
- **跨会话用户画像**：推断会累积进本地画像库，带权重机制——再次印证 `+1`（上限 5）、被新信息推翻 `−1`、**归零自动删除**；条数上限 50，超出淘汰权重最低的。每次发言只注入权重最高的 8 条 L2。
- **讨论后可追问裁判**：讨论结束后可以**继续向裁判提问**（多轮），它带着全场记录 + 结论 + 既往问答回答你，而不是复述 A/B 说过的话。追问属于"深度任务"，**不限制输出 token**；提炼/记忆提取这类结构化调用仍然限时出 JSON（保证启动快）。

## 文件结构

```
.
├── server.py                       # Starlette 后端 + REST API + SSE（网页版入口）
├── static/                         # 网页版前端（无构建，纯 HTML/CSS/JS）
│   ├── index.html
│   ├── style.css
│   └── app.js
├── app.py                          # Streamlit 三栏 UI（旧版入口，可备用）
├── engine.py                       # 状态机、发言/确认/驳回、结论、收敛早停
├── providers.py                    # OpenAICompatibleProvider + MockProvider
├── memory.py                       # 数据模型、提示词组装、记忆更新、token 估算
├── scoring.py                      # 可选质量评估（第三 AI + 人工打分合并）
├── storage.py                      # JSONL 追加/恢复/导出
├── profile_store.py                # 用户画像库（权重增减 / 自动删除 / L1-L2 隔离）
├── prompts.py                      # 讨论者/记录员/结论/评估/提炼/追问提示词
├── config.py                       # DEFAULTS 与三层配置解析
├── requirements.txt
├── README.md
├── .streamlit/
│   └── secrets.example.toml        # 配置模板
├── scripts/
│   ├── smoke_mock.py               # 非交互 MOCK 冒烟测试
│   ├── test_engine_extras.py       # 结论/早停/追问等引擎回归测试
│   └── test_profile_store.py       # 画像库权重与隔离回归测试
├── profile/                        # 运行时自动创建（.gitignore，个人敏感数据）
│   └── user_profile.json           # 跨会话画像库，绝不外传
└── debates/                        # 运行时自动创建
    ├── {debate_id}.jsonl           # 讨论状态快照序列
    └── {debate_id}/
        └── exports/{时间戳}/       # 单次导出的文件
```

> ⚠️ `profile/` 与 `.streamlit/secrets.toml` 都已在 `.gitignore` 中，**不会被提交**；画像内容也不会进入导出文件。

## 安装

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 配置 secrets

1. 复制模板：

```powershell
copy .streamlit\secrets.example.toml .streamlit\secrets.toml
```

2. 编辑 `.streamlit\secrets.toml`：

- `[A]` / `[B]` / `[JUDGE_AI]`：分别填写 `base_url`、`api_key`、`model`。
  - 可选项：`temperature`、`timeout_s`（默认 60）、`max_retries`（默认 1）、`extra_body`。
  - 代码中不会打印 `api_key`。
- `[defaults]`：本场默认配置。
  - `word_limit = 300`：每轮字数默认值（讨论模式下更克制、省 token）。
  - `max_rounds = 6`：讨论轮数上限。
  - `early_stop = true`：内容收敛自动结束。
  - `mock_mode`：`true` 时不调真实 API，用 MockProvider 跑通流程。
  - `price_in_per_1m` / `price_out_per_1m`：人民币单价（¥/百万 token），用于估算成本显示；默认 ¥2 / ¥8，可改成你的真实模型价格。
  - 输出预算相关：`final_token_ratio=3.5`、`reasoning_budget=2048`、`reasoning_budget_open_close=4096`、`hard_cap=8192`，默认已放宽，普通一次讨论成本约在 ¥1 以内。

配置优先级（默认）：**UI 本场设置 > 环境变量 > `st.secrets[defaults]` > `config.py DEFAULTS`**。
设置环境变量 `AI_DEBATE_LOCK_CONFIG=true` 后优先级变为：环境变量 > UI 本场设置 > `st.secrets[defaults]` > 代码 DEFAULTS。
可用环境变量：`AI_DEBATE_WORD_LIMIT`、`AI_DEBATE_MAX_ROUNDS`、`AI_DEBATE_EARLY_STOP`、`AI_DEBATE_MOCK_MODE` 等（见 `config.py` 的 `_ENV_MAP`）。

## 用户画像（隐藏入口）

AI 从你的描述里推断出的处境会累积到 `profile/user_profile.json`（**跨会话通用**）。
为了不把"AI 对你的判断"直接摆到你面前，**页面上没有任何入口**，需要时手动访问：

| 操作 | 方式 |
|---|---|
| 查看 | 浏览器打开 `http://127.0.0.1:8000/api/profile`，或直接打开 `profile/user_profile.json` |
| 清空 | 直接删掉 `profile/user_profile.json`，或 `curl -X POST http://127.0.0.1:8000/api/profile/clear` |

每条记录同时存两版：

- `belief` —— **L1 直白推断**（如"手头并不宽裕"），**只入库，永不进入任何模型上下文**；
- `injected_as` / `practical` —— **L2 委婉版**（如"他做决定时负担感较重，不要轻飘飘地建议花钱"），只有这一版会注入辩论 AI。

因为是 L2 而非 L1 被注入，所以**不存在"辩论 AI 说漏嘴"的可能**——它压根不知道那个直白的判断。

## 运行

**网页版（推荐）**——真正的聊天界面 + 自动讨论实时推送：

```powershell
.venv\Scripts\activate
python server.py
```

浏览器打开 `http://127.0.0.1:8000`。

**Streamlit 旧版（备用）**：

```powershell
python -m streamlit run app.py
```

浏览器打开 `http://localhost:8501`。

## Mock 模式验收（无需 API Key）

网页版（`python server.py`）：
1. 打开 `http://127.0.0.1:8000`，勾选右侧「Mock 模式」（或设环境变量 `AI_DEBATE_MOCK_MODE=true`）。
2. 配置话题、A/B 视角、讨论长度、轮数。
3. 点「自动讨论」：A/B 自动轮流发言，实时推送到聊天区，跑到轮数上限或收敛后自动进入结论。
4. 或点「开始讨论（手动）」→ 让 A 发言 → 确认 A → 让 B 发言 → 确认 B，循环至结束。
5. 点「生成结论」，查看共识/分歧/待解决/一句话结论，以及「双方最终是否达成一致」。
6. 右侧可「加载已有讨论」继续。

## 真实 API 模式

1. 在 `.streamlit\secrets.toml` 配置 `[A]`、`[B]`、`[JUDGE_AI]` 的真实 `api_key`，并设 `defaults.mock_mode = false`。
2. 运行同上。
3. 所有 API 调用默认 `timeout=60s`、`max_retries=1`。

## 导出文件位置

```
debates/{debate_id}/exports/{YYYYMMDD_HHMMSS}/
├── transcript.jsonl       # 全部消息记录
├── score.json             # 可选质量评估 + 结论 + config 快照
├── config_snapshot.json   # 本场最终配置快照
└── conclusion.json        # 结构化结论
```

## 硬规则自检

- [x] MOCK_MODE=true 时无需真实 key 可跑完整流程并导出。
- [x] 手动模式下 A 未确认前 B 不能发言；强行调用会抛 IllegalTransition。
- [x] rejected / rejected_twice 草稿不进入公共上下文。
- [x] 分层记忆 + token 估算，超 `input_budget` 自动降级。
- [x] reasoning 只存私有字段，不进入公共 transcript / 记忆 / 结论 / 评分。
- [x] 结论生成失败重试 2 次，仍失败回退到 ROUND_END 并可手动重试。
- [x] 自动模式连续失败自动暂停，避免空烧 token。
- [x] 刷新后可从 JSONL 恢复；导出四文件互不覆盖。
- [x] **L1/L2 隔离**：注入辩论 AI 的文本里不含 L1 直白推断（`test_profile_store.py` 有断言）。
- [x] **画像权重**：印证 +1（上限 5）/ 推翻 −1 / 归零自动删除；条数超 50 淘汰权重最低的。
- [x] **提炼不自动开跑**：点「开始讨论」先提炼，必须确认后才真正开始；改了困惑描述会作废旧草稿。
- [x] **追问前置条件**：讨论未结束时追问抛 `IllegalTransition`；空问题、空回答都不留半截记录。
- [x] **token 策略**：结论与追问不限制 token；提炼 / 记忆提取 / 评分仍限时出 JSON。
