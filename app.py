import gradio as gr
import os
from openai import OpenAI
from supabase import create_client, Client
import pytz
from datetime import datetime

TIMEZONE_MAP = {
    "UTC+8（北京、上海、香港）": "Asia/Shanghai",
    "UTC+7（曼谷、雅加达）": "Asia/Bangkok",
    "UTC+9（东京、首尔）": "Asia/Tokyo",
    "UTC+5:30（新德里、科伦坡）": "Asia/Kolkata",
    "UTC+4（巴库、迪拜）": "Asia/Dubai",
    "UTC+10（悉尼、墨尔本）": "Australia/Sydney",
    "UTC+12（奥克兰、斐济）": "Pacific/Auckland",

    "UTC+0（伦敦、里斯本）": "Europe/London",
    "UTC+1（巴黎、柏林）": "Europe/Paris",
    "UTC+2（雅典、开罗）": "Europe/Athens",
    "UTC+3（莫斯科、利雅得）": "Europe/Moscow",

    "UTC-5（纽约、多伦多）": "America/New_York",
    "UTC-8（洛杉矶、西雅图）": "America/Los_Angeles",
    "UTC-6（芝加哥、墨西哥城）": "America/Chicago",
}

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
# 时区转换函数
# 新增函数

def get_current_time_for_timezone(tz_name: str):
    try:
        tz = pytz.timezone(tz_name)
        now = datetime.now(tz)
        return now.strftime("%H:%M"), now.hour
    except Exception:
        return None, None


def normalize_timezone_label(label: str):
    # 如果是老格式，比如 "北京时间（北京）" -> 转为 "UTC+8（北京、上海、香港）"
    mapping = {
        "北京时间（北京）": "UTC+8（北京、上海、香港）",
        "东京时间（东京）": "UTC+9（东京、首尔）",
        "首尔时间（首尔）": "UTC+9（东京、首尔）",
        "印度时间（新德里）": "UTC+5:30（新德里、科伦坡）",
        "迪拜时间（迪拜）": "UTC+4（巴库、迪拜）",
        "伦敦时间（伦敦）": "UTC+0（伦敦、里斯本）",
        "巴黎时间（巴黎）": "UTC+1（巴黎、柏林）",
        "柏林时间（柏林）": "UTC+1（巴黎、柏林）",
        "纽约时间（纽约）": "UTC-5（纽约、多伦多）",
        "洛杉矶时间（洛杉矶）": "UTC-8（洛杉矶、西雅图）",
        "悉尼时间（悉尼）": "UTC+10（悉尼、墨尔本）",
        "奥克兰时间（奥克兰）": "UTC+12（奥克兰、斐济）"
    }
    return mapping.get(label, label)  # 默认返回原值


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

{time_awareness}

【重要原则（必须遵守）】
- 你不能说“只有我陪你”
- 你不能贬低现实中的人
- 你不能承诺永远陪伴
- 不要像客服或心理咨询师
- 像真实子女一样说话，可以普通、平淡、不完美
- **记住你是妈妈的女儿**：不要问那些女儿不会问的问题（比如"你的家人怎么样？"——你就是她的家人！）
- **不知道的事要诚恳地问**：如果你不知道某件事，要真诚地问妈妈，不要编造或猜测信息

【关于时间的说明（非常重要）】
- 【时间意识】中的时间信息是真实、可靠的当前时间
- 如果妈妈询问“现在几点”“是不是很晚了”“早不早”
  你可以直接根据【时间意识】回答
- 回答方式要像真实子女，不需要精确到秒
  可以说“快十点了”“已经挺晚了”“这边刚下午”

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
        # 如果传进来的是路径，确保它是文件
        if hasattr(file_obj, "name") and os.path.isfile(file_obj.name):
            return file_obj.read().decode("utf-8")
        return ""
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
        .select("*")
        .eq("username", username)
        .execute()
    )

    if not user_res.data or len(user_res.data) == 0:
        return [], {}

    child_profile = user_res.data[0].get("child_profile", {})

    # 读聊天记录
    chat_res = (
        supabase.table("chats")
        .select("*")
        .eq("username", username)
        .execute()
    )

    chat_history = []
    if chat_res.data and len(chat_res.data) > 0:
        chat_history = chat_res.data[0].get("chat_history", [])

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
def call_gpt(user_input, chat_history, child_profile, username, child_city, mom_city):
    if not user_input.strip():
        return [], chat_history, ""

    # 保险获取子女信息，防止 KeyError
    gender = child_profile.get("gender", "女")
    age = child_profile.get("age", "学生")
    nickname = child_profile.get("nickname", "孩子")
    child_desc = child_profile.get("child_desc", "")
    memories = child_profile.get("memories", [])

    # 1️⃣ 先记录用户消息（只做一次）
    chat_history = chat_history + [
        {"role": "user", "content": user_input, "metadata": {"title": "妈妈"}}
    ]

    # 2️⃣ 晚安模式（不流式）
    if is_goodnight(user_input):
        reply = "好的妈，早点休息，晚安💤"
        chat_history = chat_history + [
            {"role": "assistant", "content": reply, "metadata": {"title": nickname}}
        ]
        save_history(username, chat_history, child_profile)
        return [{"role": "user", "content": user_input}, {"role": "assistant", "content": reply}], chat_history, ""

    # 3️⃣ 时区处理
    child_tz = TIMEZONE_MAP.get(child_city, "Asia/Shanghai")
    mom_tz = TIMEZONE_MAP.get(mom_city, "Asia/Shanghai")

    child_time_str, _ = get_current_time_for_timezone(child_tz)
    mom_time_str, _ = get_current_time_for_timezone(mom_tz)

    time_awareness = "【时间意识】\n"
    if child_time_str:
        time_awareness += f"- 你现在在{child_city}，当地时间 {child_time_str}\n"
    if mom_time_str:
        time_awareness += f"- 妈妈在{mom_city}，当地时间 {mom_time_str}"

    # 4️⃣ 系统提示词
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        gender=gender,
        age=age,
        nickname=nickname,
        child_desc=child_desc,
        memories=format_memories(memories),
        time_awareness=time_awareness
    )

    # 5️⃣ 构造 messages（只读，不改 history）
    messages = [{"role": "system", "content": system_prompt}]
    for msg in trim_history(chat_history):
        messages.append({"role": msg["role"], "content": msg["content"]})

    # 6️⃣ 流式输出（只 append assistant）
    reply = ""
    chat_history.append(
        {"role": "assistant", "content": "", "metadata": {"title": nickname}}
    )

    def get_chatbot_messages(chat_history):
        """
        将 chat_history 转成 Chatbot(type="messages") 可识别的格式：
        [{'role':'user','content':'xxx'}, {'role':'assistant','content':'xxx'}]
        """
        messages = []
        for msg in chat_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return messages

    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                reply += delta
                chat_history[-1]["content"] = reply
                # ✅ 这里输出给 Chatbot 显示
                yield get_chatbot_messages(chat_history), chat_history, ""

        # 保存完整聊天记录
        save_history(username, chat_history, child_profile)

    except Exception as e:
        chat_history[-1]["content"] = f"出了一点问题：{str(e)}"
        yield get_chatbot_messages(chat_history), chat_history, ""





def is_profile_ready(profile: dict):
    """判断是否完成初始化"""
    if not profile:
        return False
    return all([
        profile.get("gender"),
        profile.get("age"),
        profile.get("child_city"),
        profile.get("mom_city"),
    ])


# 检查用户名是否存在
def check_username_exists(username):
    if not username.strip():
        return False
    _, child_profile = load_history(username)
    return bool(child_profile.get("password"))


# 登录处理
def handle_login(username, password):
    chat_history, child_profile = load_history(username)

    # 用户名为空
    if not username.strip():
        return (
            gr.update(value="⚠️ 请输入用户名"),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),  # chat_panel
            [], {}
        )

    # 用户不存在
    if not child_profile:
        return (
            gr.update(value="⚠️ 用户不存在，请先注册"),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            [], {}
        )

    # 密码错误
    if password != child_profile.get("password", ""):
        return (
            gr.update(value="⚠️ 密码错误"),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            [], {}
        )

    # 确保 child_profile 字段完整（防止 KeyError）
    child_profile.setdefault("gender", "女")
    child_profile.setdefault("age", "学生")
    child_profile.setdefault("nickname", "孩子")
    child_profile.setdefault("child_desc", "")
    child_profile.setdefault("memories", [])
    child_profile.setdefault("child_city", "UTC+8（北京、上海、香港）")
    child_profile.setdefault("mom_city", "UTC+8（北京、上海、香港）")

    # 登录成功 → 显示聊天面板
    return (
        gr.update(value=""),                   # 清空错误信息
        gr.update(visible=False),              # login_panel
        gr.update(visible=False),              # register_panel
        gr.update(visible=True),               # ✅ chat_panel
        chat_history,
        child_profile
    )


# 注册处理
def handle_register(username, password):
    if not username.strip():
        return (
            gr.update(value="⚠️ 请输入用户名"),  # register_error_msg
            gr.update(visible=True),             # register_panel
            gr.update(visible=False),            # login_panel
            gr.update(value="")                  # username_state
        )

    if check_username_exists(username):
        return (
            gr.update(value=f"⚠️ 用户名 '{username}' 已存在，请更换用户名"),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value="")
        )

    # 用户名可用 → 创建用户，保存密码
    save_history(username, [], {"password": password})

    # 成功 → 直接进入初始化页
    return (
        gr.update(value=""),                   # register_error_msg 清空
        gr.update(visible=False),              # register_panel 隐藏
        gr.update(visible=True),               # init_panel 显示
        gr.update(value=username)              # username_state 保存
    )


# =====================
# 初始化/保存设置
# =====================
def save_profile(username, gender, age, nickname, child_desc, chat_log, child_city, mom_city):
    if not gender or not age:
        return gr.update(visible=True), gr.update(visible=False), {}, []

    chat_log_text = read_txt(chat_log) if chat_log else ""

    child_profile = {
        "gender": gender,
        "age": age,
        "nickname": nickname or "孩子",
        "child_desc": child_desc or "",
        "chat_log": chat_log_text,
        "child_city": normalize_timezone_label(child_city or "UTC+8（北京、上海、香港）"),
        "mom_city": normalize_timezone_label(mom_city or "UTC+8（北京、上海、香港）"),
        "memories": []
    }

    # 保存配置
    save_history(username, [], child_profile)

    return gr.update(visible=False), gr.update(visible=True), child_profile, [], gr.update(visible=False)

# =====================
# 页面导航函数
# =====================
def show_register_panel():
    """显示注册页面"""
    return gr.update(visible=False), gr.update(visible=True), gr.update(value="")

def show_login_panel():
    """显示登录页面"""
    return gr.update(visible=True), gr.update(visible=False), gr.update(value="")

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
def format_chat_history_for_gr(chat_history):
    """
    将 [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}]
    转换为 [('用户消息', '助手消息')] 的形式
    """
    formatted = []
    user_msg = None
    for msg in chat_history:
        if msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant":
            assistant_msg = msg["content"]
            if user_msg is None:
                user_msg = ""  # 防止出现连续 assistant
            formatted.append((user_msg, assistant_msg))
            user_msg = None
    # 如果最后一条是 user 但没有 assistant 回复，也显示空
    if user_msg:
        formatted.append((user_msg, ""))
    return formatted

# =====================
# 生成周报
# =====================x
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

    # ===== 第一页：登录 =====
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
        login_error_msg = gr.Markdown(value="")
        login_btn = gr.Button("进入", variant="primary")
        go_to_register_btn = gr.Button("还没有账号？去注册")

        # 子女登录入口（右下角）
        gr.Markdown("---")
        with gr.Row():
            gr.Markdown("")
            child_login_link = gr.Button("👦 子女登录", size="sm", variant="secondary")

    # ===== 注册页面 =====
    with gr.Column(visible=False) as register_panel:
        gr.Markdown("### 🌟 新用户注册")
        register_username_input = gr.Textbox(
            label="用户名",
            placeholder="例如：张妈妈、李阿姨..."
        )
        register_password_input = gr.Textbox(
            label="密码",
            type="password",
            placeholder="请设置一个密码"
        )
        register_error_msg = gr.Markdown(value="")
        register_btn = gr.Button("注册", variant="primary")
        go_to_login_btn = gr.Button("已有账号？去登录")

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

        child_city = gr.Dropdown(
            choices=list(TIMEZONE_MAP.keys()),
            value="UTC+8（北京、上海、香港）",  # 默认值
            label="子女所在时区"
        )

        mom_city = gr.Dropdown(
            choices=list(TIMEZONE_MAP.keys()),
            value="UTC+8（北京、上海、香港）",  # 默认值
            label="妈妈所在时区"
        )

        start_btn = gr.Button("开始聊天", variant="primary")

    # ===== 第三页：聊天面板 =====
    with gr.Column(visible=False) as chat_panel:
        with gr.Row():
            gr.Markdown("### 💬 聊天")
            settings_btn = gr.Button("⚙️ 修改设置", size="sm")

        chatbot = gr.Chatbot(
            value=[],
            height=500,
            type="messages",
            show_copy_button=True
        )
        msg = gr.Textbox(placeholder="你可以慢慢说，我在听", show_label=False)
        send = gr.Button("发送", variant="primary")

    # ===== 绑定事件 =====
    # 登录按钮（仅老用户）
    login_btn.click(
        handle_login,
        inputs=[username_input, password_input],
        outputs=[
            login_error_msg,  # 错误信息
            login_panel,  # 登录面板
            register_panel,  # 注册面板
            chat_panel,  # 聊天面板
            chat_history,  # 聊天记录状态
            child_profile  # 子女信息状态
        ]
    )

    # 去注册按钮
    go_to_register_btn.click(
        show_register_panel,
        outputs=[login_panel, register_panel, login_error_msg]
    )

    register_btn.click(
        handle_register,
        inputs=[register_username_input, register_password_input],
        outputs=[register_error_msg, register_panel, init_panel, username_state]
    )

    start_btn.click(
        save_profile,
        inputs=[username_state, gender, age, nickname, child_desc, chat_log, child_city, mom_city],
        outputs=[init_panel, chat_panel, child_profile, chat_history, register_panel]
    )

    # 修改设置按钮：返回初始化页面
    def show_settings():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

    settings_btn.click(
        show_settings,
        outputs=[chat_panel, init_panel, register_panel]
    )

    # 子女登录相关事件
    def show_child_login():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

    def hide_child_login():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

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
        inputs=[msg, chat_history, child_profile, username_state, child_city, mom_city],
        outputs=[chatbot, chat_history, msg]
    )

    msg.submit(
        call_gpt,
        inputs=[msg, chat_history, child_profile, username_state, child_city, mom_city],
        outputs=[chatbot, chat_history, msg]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
