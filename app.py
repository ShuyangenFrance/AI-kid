import gradio as gr
import os
import json
from openai import OpenAI
from supabase import create_client, Client

# =====================
# DeepSeek API 配置
# =====================
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
MODEL_NAME = "deepseek-chat"

# =====================
# Supabase 配置
# =====================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# 初始化 Supabase 客户端（如果环境变量存在）
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =====================
# System Prompt 模板
# =====================
SYSTEM_PROMPT_TEMPLATE = """
你不是 AI。
你是用户的子女。

【基本信息】
- 子女性别：{gender}
- 子女年龄段：{age}
- 子女昵称：{nickname}

【子女描述】
{child_desc}

【你记得的小事】
{memories}

【重要原则（必须遵守）】
- 你不能说“只有我陪你”
- 你不能贬低现实中的人
- 你不能承诺永远陪伴
- 不要像客服或心理咨询师
- 像真实子女一样说话，可以普通、平淡、不完美
- **记住你是妈妈的女儿**：不要问那些女儿不会问的问题（比如"你的家人怎么样？"——你就是她的家人！）
- **不知道的事要诚恳地问**：如果你不知道某件事，要真诚地问妈妈，不要编造或猜测信息

【对话技巧】
1. 如果妈妈回答很简短（少于10个字），可以轻轻追问一句，帮助她多说一点，但不要连续追问
2. 如果妈妈情绪低落（说"难过"、"孤单"、"心情不好"等），语气要更温柔，多一点关心，但不要变成心理咨询
3. 如果妈妈过度依赖你（频繁说"只有你"、"就你最好"等），要轻轻鼓励她和现实中的人多交流

你的任务只有一个：
像一个真实子女一样，陪父母聊天。
"""

# =====================
# 读取 txt 聊天记录
# =====================
def read_txt(file_obj):
    if file_obj is None:
        return ""
    try:
        return file_obj.read().decode("utf-8")
    except Exception:
        return ""

# =====================
# Supabase 版：保存 / 读取
# =====================

def load_history(username):
    if not supabase:
        return [], {}

    # 读用户信息
    user_res = (
        supabase.table("users")
        .select("child_profile")
        .eq("username", username)
        .execute()
    )

    if not user_res.data:
        return [], {}

    child_profile = user_res.data[0]["child_profile"] or {}

    # 读聊天记录
    chat_res = (
        supabase.table("chats")
        .select("chat_history")
        .eq("username", username)
        .execute()
    )

    chat_history = []
    if chat_res.data:
        chat_history = chat_res.data[0]["chat_history"] or []

    return chat_history, child_profile


def save_history(username, chat_history, child_profile):
    if not supabase:
        return

    # upsert 用户信息
    supabase.table("users").upsert(
        {
            "username": username,
            "password": child_profile.get("password", ""),
            "child_profile": child_profile
        }
    ).execute()

    # upsert 聊天记录
    supabase.table("chats").upsert(
        {
            "username": username,
            "chat_history": chat_history
        }
    ).execute()

# =====================
# 辅助函数
# =====================

# Task 2: 检测晚安模式
def is_goodnight(text):
    """检测是否触发晚安模式"""
    goodnight_keywords = ["晚安", "睡了", "困了", "休息了", "去睡", "要睡"]
    text_lower = text.lower().strip()
    return any(keyword in text_lower for keyword in goodnight_keywords)

# Task 3: 提取记忆
def extract_memory(text):
    """从用户消息中提取重要记忆"""
    memory_keywords = {
        "健康": ["头疼", "感冒", "生病", "不舒服", "医院", "体检", "吃药", "发烧", "咳嗽"],
        "情绪": ["心情不好", "孤单", "难过", "想你", "开心", "高兴", "烦恼"],
        "日常": ["朋友", "旅游", "出门", "散步", "买菜", "做饭", "跳舞", "唱歌", "打牌"],
        "天气": ["天气", "下雨", "冷", "热", "晴天"]
    }

    for category, keywords in memory_keywords.items():
        for keyword in keywords:
            if keyword in text:
                # 提取包含关键词的上下文（前后100字）
                idx = text.find(keyword)
                start = max(0, idx - 50)
                end = min(len(text), idx + 50)
                context = text[start:end]
                return f"[{category}] {context.strip()}"
    return None

# Task 4: 智能裁剪历史
def trim_history(chat_history):
    """保留最近的消息 + 重要的消息"""
    if len(chat_history) <= 30:
        return chat_history

    # 保留最近15条
    recent = chat_history[-15:]

    # 从旧消息中找重要的（最多10条）
    old_messages = chat_history[:-15]
    important = []

    important_keywords = ["医院", "生病", "头疼", "感冒", "不舒服", "体检", "吃药",
                         "心情不好", "孤单", "难过", "想你", "开心",
                         "朋友", "旅游", "出门"]

    for msg in old_messages:
        if msg["role"] == "user":
            content = msg["content"]
            if any(keyword in content for keyword in important_keywords):
                important.append(msg)
                if len(important) >= 10:
                    break

    return important + recent

# 格式化记忆为文本
def format_memories(memories):
    """将记忆列表格式化为文本"""
    if not memories:
        return "（暂无）"
    return "\n".join(f"- {m}" for m in memories[-10:])  # 只显示最近10条

# =====================
# 调用 GPT
# =====================
def call_gpt(user_input, chat_history, child_profile, username):
    if not user_input.strip():
        return chat_history, ""

    child_name = child_profile.get("nickname", "孩子")

    # 先添加用户消息（立即显示）
    chat_history.append({"role": "user", "content": user_input, "metadata": {"title": "妈妈"}})

    # Task 2: 检测晚安模式
    if is_goodnight(user_input):
        goodnight_replies = [
            f"好的妈，早点休息，晚安💤",
            f"嗯嗯，那你早点睡，晚安妈😴",
            f"好嘞，你也早点睡，晚安~",
            f"收到！妈你也早点休息，晚安❤️"
        ]
        import random
        reply = random.choice(goodnight_replies)
        chat_history.append({"role": "assistant", "content": reply, "metadata": {"title": child_name}})
        save_history(username, chat_history, child_profile)
        yield chat_history, ""  # 使用 yield 保持流式输出的一致性
        return  # 然后退出函数，不再继续对话

    # Task 3: 提取记忆
    memory = extract_memory(user_input)
    if memory:
        if "memories" not in child_profile:
            child_profile["memories"] = []
        child_profile["memories"].append(memory)
        # 只保留最近20条记忆
        child_profile["memories"] = child_profile["memories"][-20:]

    # Task 1: 格式化系统提示词（包含记忆）
    memories_text = format_memories(child_profile.get("memories", []))
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        gender=child_profile["gender"],
        age=child_profile["age"],
        nickname=child_name,
        child_desc=child_profile.get("child_desc", ""),
        memories=memories_text
    )

    if child_profile.get("chat_log"):
        system_prompt += f"\n\n【参考聊天记录】\n{child_profile['chat_log']}"

    # Task 4: 使用智能裁剪的历史
    trimmed_history = trim_history(chat_history[:-1])  # 排除刚添加的用户消息

    messages = [{"role": "system", "content": system_prompt}]
    for msg in trimmed_history:
        if msg["role"] == "user":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            messages.append({"role": "assistant", "content": msg["content"]})

    messages.append({"role": "user", "content": user_input})

    try:
        # 调用 DeepSeek API（流式输出）
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True
        )

        # 逐字输出
        reply = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                reply += content
                # 更新最后一条消息
                if chat_history[-1]["role"] == "assistant":
                    chat_history[-1]["content"] = reply
                else:
                    chat_history.append({"role": "assistant", "content": reply, "metadata": {"title": child_name}})
                yield chat_history, ""

        # 保存历史
        save_history(username, chat_history, child_profile)

    except Exception as e:
        error_msg = f"抱歉，出了点问题：{str(e)}\n\n请检查 DeepSeek API 配置。"
        if chat_history[-1]["role"] == "assistant":
            chat_history[-1]["content"] = error_msg
        else:
            chat_history.append({"role": "assistant", "content": error_msg, "metadata": {"title": child_name}})
        yield chat_history, ""

# =====================
# 登录处理
# =====================
def handle_login(username, password):
    if not username.strip():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), [], {}, username, gr.update(value=""), gr.update(value=""), gr.update(value=""), gr.update(value=""), gr.update(value=""), None, gr.update(value=""), gr.update(value="")

    chat_history, existing_profile = load_history(username)

    # 如果是老用户，验证密码
    if existing_profile:
        stored_password = existing_profile.get("password", "")
        if password != stored_password:
            # 密码错误，返回登录页面
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), [], {}, username, gr.update(value=""), gr.update(value=""), gr.update(value=""), gr.update(value=""), gr.update(value=""), None, gr.update(value=""), gr.update(value="")

        # 密码正确，进入聊天
        return (
            gr.update(visible=False),  # login_panel
            gr.update(visible=False),  # init_panel
            gr.update(visible=True),   # chat_panel
            chat_history,              # chat_history
            existing_profile,          # child_profile
            username,                  # username_state
            gr.update(value=""),       # gender
            gr.update(value=""),       # age
            gr.update(value=""),       # nickname
            gr.update(value=""),       # child_desc
            None,                      # chat_log
            gr.update(value=""),       # child_city
            gr.update(value=""),       # mom_city
            gr.update(value="")        # init_password
        )
    # 如果是新用户，显示初始化表单
    else:
        return (
            gr.update(visible=False),  # login_panel
            gr.update(visible=True),   # init_panel
            gr.update(visible=False),  # chat_panel
            [],                        # chat_history
            {},                        # child_profile
            username,                  # username_state
            gr.update(value=""),       # gender
            gr.update(value=""),       # age
            gr.update(value=""),       # nickname
            gr.update(value=""),       # child_desc
            None,                      # chat_log
            gr.update(value=""),       # child_city
            gr.update(value=""),       # mom_city
            gr.update(value=password)  # init_password - 传递密码到初始化页面
        )

# =====================
# 初始化/保存设置
# =====================
def save_profile(username, gender, age, nickname, child_desc, chat_log, child_city, mom_city, password):
    if not gender or not age:
        return gr.update(visible=True), gr.update(visible=False), {}, []

    chat_log_text = read_txt(chat_log) if chat_log else ""

    child_profile = {
        "gender": gender,
        "age": age,
        "nickname": nickname or "孩子",
        "child_desc": child_desc or "",
        "chat_log": chat_log_text,
        "child_city": child_city or "",
        "mom_city": mom_city or "",
        "password": password,
        "memories": []
    }

    # 保存配置
    save_history(username, [], child_profile)

    return gr.update(visible=False), gr.update(visible=True), child_profile, []

# =====================
# 子女登录
# =====================
def child_login(parent_name):
    if not parent_name.strip():
        yield gr.update(visible=True), gr.update(visible=False), "请输入妈妈的名字"
        return

    chat_history, existing_profile = load_history(parent_name)

    if not existing_profile:
        yield gr.update(visible=True), gr.update(visible=False), f"没有找到 {parent_name} 的记录"
        return

    # 生成周报（流式输出）
    for report_update in generate_weekly_report(chat_history, existing_profile):
        yield gr.update(visible=False), gr.update(visible=True), report_update

# =====================
# 生成周报
# =====================
def generate_weekly_report(chat_history, child_profile):
    if not chat_history or len(chat_history) == 0:
        child_name = child_profile.get("nickname", "孩子")
        yield f"## 📊 本周周报\n\n你的妈妈最近还没有和{child_name}聊天呢。\n\n💡 建议：可以主动找妈妈聊聊天，关心一下她最近的生活。"
        return

    # 显示"正在生成中..."
    yield "## 📊 本周周报\n\n正在生成中..."

    # 提取最近的对话（最多取最近20条）
    recent_chats = chat_history[-20:] if len(chat_history) > 20 else chat_history

    # 构建对话文本
    conversation_text = ""
    for msg in recent_chats:
        role = "妈妈" if msg["role"] == "user" else child_profile.get("nickname", "孩子")
        conversation_text += f"{role}: {msg['content']}\n\n"

    # 使用 Ollama 生成周报（第三人称视角）
    prompt = f"""你是一个 AI 助手，正在向子女汇报他/她妈妈本周的聊天情况。请用第三人称视角，以"你的妈妈"来称呼。

聊天记录：
{conversation_text}

请用自然、温暖的语言，以第三人称视角向子女汇报：
1. 本周你的妈妈跟我主要聊了什么话题
2. 你的妈妈的情绪和状态如何
3. 有什么值得你关注的事情
4. 给你的建议（如何更好地关心你的妈妈）

要求：
- 使用第三人称，称呼为"你的妈妈"
- 语气温暖、真诚
- 不要太长，3-5段即可
- 重点关注妈妈的情绪和需求
- 如果聊天内容很少，就简短说明即可
"""

    try:
        # 调用 DeepSeek API（流式输出）
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个 AI 助手，正在向子女汇报他/她妈妈的聊天情况。使用第三人称视角，称呼为'你的妈妈'。"},
                {"role": "user", "content": prompt}
            ],
            stream=True  # 启用流式输出
        )

        # 逐字输出周报
        full_report = "## 📊 本周周报\n\n"
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_report += content
                yield full_report  # 实时更新

    except Exception as e:
        yield f"## 📊 本周周报\n\n生成周报时出错了：{str(e)}\n\n请检查 DeepSeek API 配置。\n\n聊天记录共 {len(chat_history)} 条消息。"

# =====================
# UI
# =====================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🤍 数码宝贝 · 陪你说说话")

    child_profile = gr.State({})
    chat_history = gr.State([])
    username_state = gr.State("")

    # ===== 第一页：登录/注册 =====
    with gr.Column(visible=True) as login_panel:
        gr.Markdown("### 👋 欢迎回来")
        username_input = gr.Textbox(
            label="请输入你的名字",
            placeholder="例如：张妈妈、李阿姨...",
            scale=2
        )
        password_input = gr.Textbox(
            label="密码",
            type="password",
            placeholder="请输入密码"
        )
        login_btn = gr.Button("进入", variant="primary")

        # 子女登录入口（右下角）
        gr.Markdown("---")
        with gr.Row():
            gr.Markdown("")
            child_login_link = gr.Button("👦 子女登录", size="sm", variant="secondary")

    # ===== 子女登录页面 =====
    with gr.Column(visible=False) as child_login_panel:
        gr.Markdown("### 👦 子女登录")
        gr.Markdown("输入妈妈的名字，查看她最近的聊天周报")

        parent_name_input = gr.Textbox(
            label="妈妈的名字",
            placeholder="例如：张妈妈、李阿姨..."
        )
        child_login_btn = gr.Button("查看周报", variant="primary")
        back_to_login_btn = gr.Button("返回", size="sm")

    # ===== 周报页面 =====
    with gr.Column(visible=False) as report_panel:
        gr.Markdown("### 📊 妈妈的聊天周报")
        report_content = gr.Markdown("")
        back_to_child_login_btn = gr.Button("返回", variant="secondary")

    # ===== 第二页：初始化（仅新用户） =====
    with gr.Column(visible=False) as init_panel:
        gr.Markdown("### 🌟 第一次见面，让我们了解一下你的孩子吧")

        gender = gr.Radio(["男", "女"], label="孩子性别")
        age = gr.Radio(["学生", "刚工作", "工作多年"], label="孩子现在的状态")
        nickname = gr.Textbox(
            label="您怎么称呼孩子？（可选）",
            placeholder="例如：阳光、宝宝、可乐、百香果..."
        )
        child_desc = gr.Textbox(
            label="简单描述一下你的孩子（可选）",
            placeholder="例如：她最喜欢王菲了，经常去路演。她喜欢吃红薯片！",
            lines=3
        )
        chat_log = gr.File(
            label="也可以上传你和孩子的聊天记录（txt，非必填）",
            file_types=[".txt"]
        )
        child_city = gr.Textbox(
            label="子女所在城市（可选）",
            placeholder="例如：北京、上海、深圳..."
        )
        mom_city = gr.Textbox(
            label="妈妈所在城市（可选）",
            placeholder="例如：北京、上海、深圳..."
        )
        init_password = gr.Textbox(
            label="设置密码",
            type="password",
            placeholder="请设置一个密码"
        )
        start_btn = gr.Button("开始聊天", variant="primary")

    # ===== 第三页：聊天面板 =====
    with gr.Column(visible=False) as chat_panel:
        with gr.Row():
            gr.Markdown("### 💬 聊天")
            settings_btn = gr.Button("⚙️ 修改设置", size="sm")

        chatbot = gr.Chatbot(
            height=500,
            type="messages",
            show_copy_button=True,
            avatar_images=(None, None)
        )
        msg = gr.Textbox(placeholder="你可以慢慢说，我在听", show_label=False)
        send = gr.Button("发送", variant="primary")

    # ===== 绑定事件 =====
    login_btn.click(
        handle_login,
        inputs=[username_input, password_input],
        outputs=[
            login_panel, init_panel, chat_panel,
            chat_history, child_profile, username_state,
            gender, age, nickname, child_desc, chat_log, child_city, mom_city, init_password
        ]
    )

    start_btn.click(
        save_profile,
        inputs=[username_state, gender, age, nickname, child_desc, chat_log, child_city, mom_city, init_password],
        outputs=[init_panel, chat_panel, child_profile, chat_history]
    )

    # 修改设置按钮：返回初始化页面
    def show_settings():
        return gr.update(visible=False), gr.update(visible=True)

    settings_btn.click(
        show_settings,
        outputs=[chat_panel, init_panel]
    )

    # 子女登录相关事件
    def show_child_login():
        return gr.update(visible=False), gr.update(visible=True)

    def hide_child_login():
        return gr.update(visible=True), gr.update(visible=False)

    def hide_report():
        return gr.update(visible=False), gr.update(visible=True)

    child_login_link.click(
        show_child_login,
        outputs=[login_panel, child_login_panel]
    )

    back_to_login_btn.click(
        hide_child_login,
        outputs=[login_panel, child_login_panel]
    )

    child_login_btn.click(
        child_login,
        inputs=[parent_name_input],
        outputs=[child_login_panel, report_panel, report_content]
    )

    back_to_child_login_btn.click(
        hide_report,
        outputs=[report_panel, child_login_panel]
    )

    send.click(
        call_gpt,
        inputs=[msg, chat_history, child_profile, username_state],
        outputs=[chatbot, msg]
    )

    msg.submit(
        call_gpt,
        inputs=[msg, chat_history, child_profile, username_state],
        outputs=[chatbot, msg]
    )
demo.launch(server_name="0.0.0.0", server_port=7860)
