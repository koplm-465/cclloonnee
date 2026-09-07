import os
import re
import sys
import logging
import random
import numpy as np
import gradio as gr
from typing import Optional, Tuple
from funasr import AutoModel
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import voxcpm
from voxcpm.model.utils import resolve_runtime_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- Inline i18n (en + zh-CN only) ----------

_USAGE_INSTRUCTIONS_EN = (
    "**KP VOICE CLONE — အသံထုတ်လုပ်မှု နည်းလမ်း ၃ မျိုး**\n\n"
    "🎨 **အသံဒီဇိုင်း** — အသံအသစ်တစ်မျိုးကို ဖန်တီးပါ  \n"
    "အညွှန်းအသံ မလိုပါ။ အသံရဲ့ လိင်၊ အသက်၊ အသံအနိမ့်အမြင့်၊ ခံစားချက်နဲ့ ပြောနှုန်းတို့ကို **အသံညွှန်ကြားချက်** ထဲမှာ ရေးပေးပါ။ KP VOICE CLONE က သင့်ဖော်ပြချက်အတိုင်း အသံအသစ်ကို ဖန်တီးပေးပါမယ်။\n\n"
    "🎛️ **ထိန်းချုပ်နိုင်သော အသံကူးယူမှု** — စတိုင်ကို ထိန်းချုပ်ပြီး အသံကူးယူပါ  \n"
    "အညွှန်းအသံဖိုင်ကို တင်ပြီး **အသံညွှန်ကြားချက်** ထဲမှာ ခံစားချက်၊ ပြောနှုန်းနဲ့ ပြောဟန်ကို သတ်မှတ်နိုင်ပါတယ်။ မူရင်းအသံရဲ့ သွင်ပြင်ကို ထိန်းသိမ်းပေးပါမယ်။\n\n"
    "🎙️ **အသေးစိတ်အသံကူးယူမှု** — အသံရဲ့ အသေးစိတ်အချက်အလက်အားလုံးကို ပြန်လည်ဖန်တီးပါ  \n"
    "**အသေးစိတ်အသံကူးယူမှု** ကို ဖွင့်ပြီး အညွှန်းအသံရဲ့ စာသားကို ထည့်ပါ (သို့မဟုတ် အလိုအလျောက် စာသားပြောင်းပါ)။ မော်ဒယ်က အညွှန်းအသံကို ရှေ့ပြောပြီးသားအပိုင်းအဖြစ် သတ်မှတ်ကာ အသံအသေးစိတ်များကို ထိန်းသိမ်းပြီး ဆက်လက်ထုတ်လုပ်ပေးပါမယ်။"
    "မှတ်ချက် — ဒီနည်းလမ်းကို သုံးတဲ့အခါ အသံညွှန်ကြားချက်ကို ပိတ်ထားပါမယ်။"
)

_EXAMPLES_FOOTER_EN = (
    "---\n"
    "**💡 အသံဖော်ပြချက် နမူနာများ**  \n"
    "အောက်ပါ အသံညွှန်ကြားချက်များကို စမ်းသုံးပြီး အသံအမျိုးမျိုးကို ဖန်တီးကြည့်ပါ။  \n\n"
    "**နမူနာ ၁ — နူးညံ့ပြီး ဝမ်းနည်းသံပါသော မိန်းကလေး**  \n"
    '`အသံညွှန်ကြားချက်`: *"အသံနူးညံ့ချိုသာသော မိန်းကလေးငယ်၊ ဖြည်းဖြည်းပြောပြီး ဝမ်းနည်းသံ အနည်းငယ်ပါစေ။"*  \n'
    '`ပြောမည့်စာသား`: *"မင်းနေခဲ့ဖို့ ငါဘယ်တုန်းကမှ မတောင်းဆိုခဲ့ပါဘူး… ဒါပေမယ့် မင်းမရှိတော့တဲ့အခါ ဘာကြောင့် ဒီလောက်နာကျင်နေရတာလဲ။"*  \n\n'
    "**နမူနာ ၂ — အေးဆေးသော လူငယ်အမျိုးသား**  \n"
    '`အသံညွှန်ကြားချက်`: *"အေးဆေးပြီး သက်သောင့်သက်သာရှိသော လူငယ်အမျိုးသားအသံ၊ ပြောနှုန်းအနည်းငယ် နှေးပါစေ။"*  \n'
    '`ပြောမည့်စာသား`: *"ဒီနေ့ အလုပ်တွေ အဆင်ပြေရဲ့လား။ အေးအေးဆေးဆေးနဲ့ တစ်ခုချင်းစီ ဆက်လုပ်သွားကြရအောင်။"*'
)

_USAGE_INSTRUCTIONS_ZH = (
    "**VoxCPM2 — 三种语音生成方式：**\n\n"
    "🎨 **声音设计（Voice Design）**  \n"
    "无需参考音频。在 **Control Instruction** 中描述目标音色特征"
    "（性别、年龄、语气、情绪、语速等），VoxCPM2 即可为你从零创造独一无二的声音。\n\n"
    "🎛️ **可控克隆（Controllable Cloning）**  \n"
    "上传参考音频，同时可选地使用 **Control Instruction** 来指定情绪、语速、风格等表达方式，"
    "在保留原始音色的基础上灵活控制说话风格。\n\n"
    "🎙️ **极致克隆（Ultimate Cloning）**  \n"
    "开启 **极致克隆模式** 并提供参考音频的文字内容（可自动识别）。"
    "模型会将参考音频视为已说出的前文，以**音频续写**的方式完整还原参考音频中的所有声音细节。"
    "注意：该模式与可控克隆模式互斥，将禁用Control Instruction。\n\n"
)

_EXAMPLES_FOOTER_ZH = (
    "---\n"
    "**💡 声音描述示例（中英文均可）：**  \n\n"
    "**示例 1 — 深宫太后**  \n"
    '`Control Instruction`: *"中老年女性，声音低沉阴冷，语速缓慢而有力，'
    '字字深思熟虑，带有深不可测的城府与威慑感。"*  \n'
    '`Target Text`: *"哀家在这深宫待了四十年，什么风浪没见过？你以为瞒得过哀家？"*  \n\n'
    "**示例 2 — 暴躁驾校教练**  \n"
    '`Control Instruction`: *"暴躁的中年男声，语速快，充满无奈和愤怒"*  \n'
    '`Target Text`: *"踩离合！踩刹车啊！你往哪儿开呢？前面是树你看不见吗？'
    '我教了你八百遍了，打死方向盘！你是不是想把车给我开到沟里去？"*  \n\n'
    "---\n"
    "**🗣️ 方言生成指南：**  \n"
    "要生成地道的方言语音，请在 **Target Text** 中直接使用方言词汇和句式，"
    "并在 **Control Instruction** 中描述方言特征。  \n\n"
    "**示例 — 广东话**  \n"
    '`Control Instruction`: *"粤语，中年男性，语气平淡"*  \n'
    '✅ 正确（粤语表达）：*"伙計，唔該一個A餐，凍奶茶少甜！"*  \n'
    '❌ 错误（普通话原文）：*"伙计，麻烦来一个A餐，冻奶茶少甜！"*  \n\n'
    "**示例 — 河南话**  \n"
    '`Control Instruction`: *"河南话，接地气的大叔"*  \n'
    '✅ 正确（河南话表达）：*"恁这是弄啥嘞？晌午吃啥饭？"*  \n'
    '❌ 错误（普通话原文）：*"你这是在干什么呢？中午吃什么饭？"*  \n\n'
    "🤖 **小技巧：** 不知道方言怎么写？可以用豆包、DeepSeek、Kimi 等 AI 助手"
    "将普通话翻译为方言文本，再粘贴到 Target Text 中即可。  \n\n"
)

_I18N_TRANSLATIONS = {
    "en": {
        "reference_audio_label": "🎤 အညွှန်းအသံဖိုင် (မဖြစ်မနေမဟုတ်ပါ — အသံကူးယူရန် တင်ပါ)",
        "show_prompt_text_label": "🎙️ အသေးစိတ်အသံကူးယူမှု (စာသားအညွှန်းဖြင့်)",
        "show_prompt_text_info": "အညွှန်းအသံထဲက စကားကို အလိုအလျောက် စာသားပြောင်းပေးပါမယ်။ ဖွင့်ထားချိန်မှာ အသံညွှန်ကြားချက်ကို ပိတ်ထားပါမယ်။",
        "prompt_text_label": "အညွှန်းအသံထဲက စာသား (အလိုအလျောက် ဖြည့်ပေးပြီး ပြင်နိုင်သည်)",
        "prompt_text_placeholder": "အညွှန်းအသံရဲ့ စာသားကို ဒီနေရာမှာ ပြပေးပါမယ် …",
        "control_label": "🎛️ အသံညွှန်ကြားချက် (မဖြစ်မနေမဟုတ်ပါ — မြန်မာ/အင်္ဂလိပ် စာသားများ ရေးနိုင်သည်)",
        "control_placeholder": "ဥပမာ — နူးညံ့သော အမျိုးသမီးအသံ / အားတက်သန်ပြီး မြန်မြန်ပြောပါ",
        "target_text_label": "✍️ ပြောမည့်စာသား",
        "generate_btn": "🔊 အသံထုတ်ရန်",
        "generated_audio_label": "ထုတ်ပြီးသောအသံ",
        "advanced_settings_title": "⚙️ အဆင့်မြင့်ဆက်တင်များ",
        "ref_denoise_label": "အညွှန်းအသံ မြှင့်တင်ခြင်း",
        "ref_denoise_info": "အသံကူးယူမီ အညွှန်းအသံကို ဆူညံသံလျှော့ပြီး မြှင့်တင်ပေးပါမယ်။",
        "normalize_label": "စာသားပုံမှန်ပြုလုပ်ခြင်း",
        "normalize_info": "ဂဏန်း၊ ရက်စွဲနဲ့ အတိုကောက်စာလုံးများကို ပုံမှန်ပြုလုပ်ပေးပါမယ်။",
        "cfg_label": "CFG (ညွှန်ကြားမှုအင်အား)",
        "cfg_info": "တန်ဖိုးမြင့်လေ အညွှန်း/အမိန့်နှင့် ပိုနီးစပ်လေ၊ တန်ဖိုးနိမ့်လေ ဖန်တီးမှုကွဲပြားမှု ပိုများလေ ဖြစ်ပါမယ်။",
        "dit_steps_label": "LocDiT ထုတ်လုပ်မှုအဆင့်များ",
        "dit_steps_info": "အဆင့်များလေလေ အသံအရည်အသွေး ပိုကောင်းနိုင်ပေမယ့် အချိန်ပိုကြာပါမယ်။",
        "seed_label": "Seed နံပါတ်",
        "seed_info": "တူညီသောရလဒ်ကို ပြန်ထုတ်နိုင်ရန် အသုံးပြုသည့် Seed နံပါတ် ဖြစ်ပါတယ်။",
        "random_seed_label": "Seed ကို အလိုအလျောက်ပြောင်းရန်",
        "random_seed_info": "အသံထုတ်တိုင်း Seed နံပါတ်အသစ်ကို အလိုအလျောက်သုံးပါမယ်။",
        "usage_instructions": _USAGE_INSTRUCTIONS_EN,
        "examples_footer": _EXAMPLES_FOOTER_EN,
    },
    "zh-CN": {
        "reference_audio_label": "🎤 参考音频（可选 — 上传后用于克隆）",
        "show_prompt_text_label": "🎙️ 极致克隆模式（基于文本引导的极致克隆）",
        "show_prompt_text_info": "自动识别参考音频文本，完整还原音色、节奏、情感等全部声音细节。开启后 Control Instruction 将暂时禁用",
        "prompt_text_label": "参考音频内容文本（上传后自动识别）",
        "prompt_text_placeholder": "参考音频的文字内容将自动识别并显示在此处 …",
        "control_label": "🎛️ Control Instruction（可选 — 支持中英文描述）",
        "control_placeholder": "如：年轻女性，温柔甜美 / A warm young woman / 暴躁老哥，语速飞快",
        "target_text_label": "✍️ Target Text — 要合成的目标文本",
        "generate_btn": "🔊 开始生成",
        "generated_audio_label": "生成结果",
        "advanced_settings_title": "⚙️ 高级设置",
        "ref_denoise_label": "参考音频降噪增强",
        "ref_denoise_info": "克隆前使用 ZipEnhancer 对参考音频进行降噪处理",
        "normalize_label": "文本规范化",
        "normalize_info": "自动规范化数字、日期及缩写（基于 wetext）",
        "cfg_label": "CFG（引导强度）",
        "cfg_info": "数值越高 → 越贴合提示/参考音色；数值越低 → 生成风格更自由",
        "dit_steps_label": "LocDiT 流匹配迭代步数",
        "dit_steps_info": "LocDiT 流匹配生成迭代步数 — 步数越多 → 可能生成更好的音频质量，但速度变慢",
        "usage_instructions": _USAGE_INSTRUCTIONS_ZH,
        "examples_footer": _EXAMPLES_FOOTER_ZH,
    },
    "zh-Hans": None,  # alias, filled below
    "zh": None,  # alias, filled below
}
# Use the Burmese interface for every supported browser locale.
_I18N_TRANSLATIONS["zh-CN"] = _I18N_TRANSLATIONS["en"].copy()
_I18N_TRANSLATIONS["zh-Hans"] = _I18N_TRANSLATIONS["en"].copy()
_I18N_TRANSLATIONS["zh"] = _I18N_TRANSLATIONS["en"].copy()

for _d in _I18N_TRANSLATIONS.values():
    if _d is not None:
        for _k, _v in _I18N_TRANSLATIONS["en"].items():
            _d.setdefault(_k, _v)

I18N = gr.I18n(**_I18N_TRANSLATIONS)

DEFAULT_TARGET_TEXT = (
    "ကေပီ ဗွိုက် ကလုံး မှ ဘာသာစကားမျိုးစုံကို ပံ့ပိုးပေးနိုင်ပြီး သဘာဝကျသော အသံများကို ဖန်တီးပေးနိုင်သည့် အသံထုတ်လုပ်မှု မော်ဒယ်တစ်ခု ဖြစ်ပါတယ်။"
)

_CUSTOM_CSS = """
:root {
    --kp-navy: #05082f;
    --kp-navy-2: #080d3d;
    --kp-panel: rgba(11, 19, 67, 0.88);
    --kp-panel-strong: #0d174f;
    --kp-cyan: #16c9ee;
    --kp-cyan-bright: #39e7ff;
    --kp-blue: #2473d9;
    --kp-text: #edfaff;
    --kp-muted: #9bb4cf;
    --kp-border: rgba(58, 209, 243, 0.22);
    --kp-glow: 0 0 28px rgba(22, 201, 238, 0.16);
}

body, .gradio-container {
    background: var(--kp-navy) !important;
    color: var(--kp-text) !important;
}
body {
    background-image: radial-gradient(circle at 12% 0%, rgba(19, 139, 202, 0.18), transparent 34%),
                      radial-gradient(circle at 88% 100%, rgba(23, 76, 185, 0.16), transparent 32%) !important;
}
.gradio-container {
    max-width: 1180px !important;
    min-height: 100vh;
    padding: 26px 22px 42px !important;
    font-family: Inter, "Segoe UI", Arial, sans-serif !important;
}

/* KP brand header */
.kp-shell {
    position: relative;
    overflow: hidden;
    padding: 26px 30px 24px;
    margin-bottom: 22px;
    border: 1px solid var(--kp-border);
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(9, 23, 74, .98), rgba(4, 9, 46, .98));
    box-shadow: var(--kp-glow), inset 0 1px 0 rgba(255,255,255,.05);
}
.kp-shell::after {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    right: -130px;
    top: -170px;
    border-radius: 50%;
    border: 1px solid rgba(38, 217, 244, .20);
    box-shadow: 0 0 0 28px rgba(38, 217, 244, .035), 0 0 0 56px rgba(38, 217, 244, .025);
}
.kp-brand {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    gap: 18px;
}
.kp-brand img {
    width: 86px;
    height: 86px;
    object-fit: cover;
    border-radius: 19px;
    border: 1px solid rgba(50, 221, 247, .55);
    box-shadow: 0 0 24px rgba(20, 200, 238, .32);
}
.kp-eyebrow {
    margin: 0 0 4px;
    color: var(--kp-cyan-bright);
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .22em;
    text-transform: uppercase;
}
.kp-title {
    margin: 0;
    color: #fff !important;
    font-size: clamp(1.65rem, 4vw, 2.35rem);
    line-height: 1.08;
    letter-spacing: -.04em;
}
.kp-subtitle {
    margin: 8px 0 0;
    color: var(--kp-muted) !important;
    font-size: .92rem;
}
.kp-status {
    position: relative;
    z-index: 1;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 18px;
    padding: 7px 12px;
    border: 1px solid rgba(48, 224, 247, .25);
    border-radius: 999px;
    color: #b9f6ff;
    background: rgba(26, 184, 215, .09);
    font-size: .78rem;
}
.kp-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--kp-cyan-bright);
    box-shadow: 0 0 10px var(--kp-cyan-bright);
}

/* Panel system */
.kp-panel {
    padding: 19px 20px 21px !important;
    border: 1px solid var(--kp-border) !important;
    border-radius: 18px !important;
    background: var(--kp-panel) !important;
    box-shadow: 0 14px 34px rgba(0, 0, 0, .22), inset 0 1px 0 rgba(255,255,255,.035) !important;
}
.kp-panel-title {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
    color: #f4fdff;
    font-size: .95rem;
    font-weight: 750;
    letter-spacing: .01em;
}
.kp-step {
    display: inline-grid;
    place-items: center;
    width: 25px;
    height: 25px;
    border-radius: 8px;
    color: #00152a;
    background: linear-gradient(135deg, var(--kp-cyan-bright), var(--kp-blue));
    font-size: .78rem;
    font-weight: 900;
    box-shadow: 0 0 14px rgba(22, 201, 238, .28);
}
.kp-help {
    color: var(--kp-muted) !important;
    font-size: .8rem !important;
}

/* Gradio controls */
.gradio-container label, .gradio-container .label-wrap span {
    color: #d8f5ff !important;
}
.gradio-container textarea, .gradio-container input, .gradio-container [data-testid="textbox"] textarea,
.gradio-container .wrap, .gradio-container .block {
    border-color: rgba(86, 193, 224, .22) !important;
}
.gradio-container textarea, .gradio-container input {
    color: #effcff !important;
    background: rgba(3, 10, 46, .72) !important;
}
.gradio-container textarea:focus, .gradio-container input:focus {
    border-color: var(--kp-cyan) !important;
    box-shadow: 0 0 0 2px rgba(22, 201, 238, .14) !important;
}
.gradio-container .wrap, .gradio-container .block {
    background: rgba(4, 12, 51, .66) !important;
    border-radius: 13px !important;
}
.gradio-container .form {
    background: transparent !important;
}
.kp-generate {
    margin-top: 14px !important;
    border: 0 !important;
    border-radius: 12px !important;
    color: #001a29 !important;
    background: linear-gradient(105deg, #0fd2ef, #42eaff 55%, #3c91ff) !important;
    box-shadow: 0 8px 24px rgba(22, 201, 238, .24) !important;
    font-weight: 850 !important;
}
.kp-generate:hover {
    filter: brightness(1.08);
    box-shadow: 0 10px 30px rgba(22, 201, 238, .38) !important;
}
.kp-output {
    min-height: 280px;
}
.kp-output audio {
    width: 100%;
}

/* Upload, accordion, toggles */
.gradio-container .upload-container, .gradio-container [data-testid="audio"] {
    border-color: rgba(46, 205, 238, .28) !important;
    background: rgba(4, 13, 53, .65) !important;
}
.gradio-container details, .gradio-container .accordion {
    border-color: var(--kp-border) !important;
    background: rgba(7, 16, 59, .62) !important;
    border-radius: 12px !important;
}
.gradio-container details > summary, .gradio-container .accordion > button {
    color: #c8f5ff !important;
}
.switch-toggle {
    padding: 9px 12px !important;
    border: 1px solid rgba(67, 201, 232, .18) !important;
    border-radius: 11px !important;
    background: rgba(9, 25, 75, .75) !important;
}
.switch-toggle input[type="checkbox"] {
    appearance: none;
    width: 43px;
    height: 24px;
    border-radius: 13px;
    background: #1c315b;
    position: relative;
    cursor: pointer;
    transition: background .25s ease;
}
.switch-toggle input[type="checkbox"]::after {
    content: "";
    position: absolute;
    top: 3px;
    left: 3px;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #afc5d7;
    transition: transform .25s ease, background .25s ease;
}
.switch-toggle input[type="checkbox"]:checked {
    background: linear-gradient(90deg, #087da9, #18d7ed);
}
.switch-toggle input[type="checkbox"]:checked::after {
    transform: translateX(19px);
    background: #fff;
}

/* Responsive stacking */
@media (max-width: 720px) {
    .gradio-container { padding: 14px 12px 28px !important; }
    .kp-shell { padding: 20px 18px; border-radius: 18px; }
    .kp-brand img { width: 68px; height: 68px; border-radius: 15px; }
    .kp-brand { gap: 13px; }
}

/* Compact, clean audio upload drop-zone */
.audio-container {
    height: 190px !important;
    min-height: 190px !important;
    overflow: hidden !important;
    border: 1px dashed rgba(57, 231, 255, .38) !important;
    border-radius: 16px !important;
    background: linear-gradient(145deg, rgba(12, 35, 92, .72), rgba(4, 13, 51, .82)) !important;
}
.audio-container button[aria-dropeffect="copy"] {
    height: 150px !important;
    min-height: 150px !important;
    border-radius: 15px !important;
    background: transparent !important;
}
.audio-container .wrap {
    display: flex !important;
    width: 100% !important;
    height: 150px !important;
    min-height: 150px !important;
    padding: 0 !important;
    flex-direction: column !important;
    justify-content: center !important;
    gap: 7px !important;
    color: var(--kp-muted) !important;
    background: transparent !important;
    box-shadow: none !important;
    font-size: 0 !important;
}
.audio-container .icon-wrap {
    display: grid !important;
    place-items: center !important;
    width: 46px !important;
    height: 46px !important;
    border: 1px solid rgba(57, 231, 255, .30) !important;
    border-radius: 14px !important;
    color: var(--kp-cyan-bright) !important;
    background: rgba(22, 201, 238, .10) !important;
}
.audio-container .wrap::after {
    content: "Audio တင်ရန်\\A သို့မဟုတ် ကိုယ်တိုင်အသံသွင်းနိုင်သည်။";
    white-space: pre;
    color: var(--kp-muted);
    font-size: .88rem;
    line-height: 1.65;
    text-align: center;
}
.audio-container button.record {
    font-size: 0 !important;
}
.audio-container button.record::after {
    content: "အသံသွင်းခြင်း ရပ်ရန်";
    font-size: .86rem;
}

/* Remove Gradio's default footer controls */
footer[aria-label="Gradio footer navigation"],
footer[aria-label="Gradio footer navigation"] .show-api,
footer[aria-label="Gradio footer navigation"] .built-with,
footer[aria-label="Gradio footer navigation"] .settings {
    display: none !important;
}
"""

_APP_THEME = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)


# ---------- Model ----------


class VoxCPMDemo:
    def __init__(self, model_id: str = "openbmb/VoxCPM2", device: str = "auto") -> None:
        self.device = resolve_runtime_device(device, "cuda")
        logger.info(f"Running VoxCPM on device: {self.device}")
        self.optimize = self.device.startswith("cuda")

        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_device = "cuda:0" if self.device.startswith("cuda") else "cpu"
        self.asr_model: Optional[AutoModel] = None

        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self._model_id = model_id

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        logger.info(f"Loading model: {self._model_id}")
        self.voxcpm_model = voxcpm.VoxCPM.from_pretrained(
            self._model_id,
            optimize=self.optimize,
            device=self.device,
        )
        logger.info("Model loaded successfully.")
        return self.voxcpm_model

    def get_or_load_asr_model(self) -> AutoModel:
        if self.asr_model is not None:
            return self.asr_model
        logger.info(f"Loading ASR model: {self.asr_model_id} on device: {self.asr_device}")
        self.asr_model = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level="DEBUG",
            device=self.asr_device,
        )
        logger.info("ASR model loaded successfully.")
        return self.asr_model

    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.get_or_load_asr_model().generate(
            input=prompt_wav,
            language="auto",
            use_itn=True,
        )
        return res[0]["text"].split("|>")[-1]

    def _build_generate_kwargs(
        self,
        *,
        final_text: str,
        audio_path: Optional[str],
        prompt_text_clean: Optional[str],
        cfg_value_input: float,
        do_normalize: bool,
        denoise: bool,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
    ) -> dict:
        generate_kwargs = dict(
            text=final_text,
            reference_wav_path=audio_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=inference_timesteps,
            normalize=do_normalize,
            denoise=denoise,
            seed=seed,
        )
        if prompt_text_clean and audio_path:
            generate_kwargs["prompt_wav_path"] = audio_path
            generate_kwargs["prompt_text"] = prompt_text_clean
        return generate_kwargs

    def generate_tts_audio(
        self,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        denoise: bool = True,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
    ) -> Tuple[int, np.ndarray, Optional[int]]:
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        control = (control_instruction or "").strip()
        # Strip any parentheses (half-width/full-width) from control text to avoid
        # breaking the "(control)text" prompt format expected by the model.
        control = re.sub(r"[()（）]", "", control).strip()
        final_text = f"({control}){text}" if control else text

        audio_path = reference_wav_path_input if reference_wav_path_input else None
        prompt_text_clean = (prompt_text or "").strip() or None

        if audio_path and prompt_text_clean:
            logger.info(f"[Voice Cloning] prompt_wav + prompt_text + reference_wav")
        elif audio_path:
            logger.info(f"[Voice Control] reference_wav only")
        else:
            logger.info(f"[Voice Design] control: {control[:50] if control else 'None'}...")

        logger.info(f"Generating audio for text: '{final_text[:80]}...'")
        generate_kwargs = self._build_generate_kwargs(
            final_text=final_text,
            audio_path=audio_path,
            prompt_text_clean=prompt_text_clean,
            cfg_value_input=cfg_value_input,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=inference_timesteps,
            seed=seed,
        )
        wav = current_model.generate(**generate_kwargs)
        last_successful_seed = getattr(current_model.tts_model, "last_successful_seed", seed)
        return (current_model.tts_model.sample_rate, wav, last_successful_seed)


# ---------- UI ----------


def create_demo_interface(demo: VoxCPMDemo):
    gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])

    def _coerce_seed(seed_value) -> Optional[int]:
        if seed_value is None or seed_value == "":
            return None
        return int(seed_value)

    def _prepare_seed(use_random_seed: bool, seed_value):
        if use_random_seed:
            return random.randint(0, 2**32 - 1)
        return _coerce_seed(seed_value)

    def _on_random_seed_toggle(checked):
        return gr.update(interactive=not checked)

    def _generate(
        text: str,
        control_instruction: str,
        ref_wav: Optional[str],
        use_prompt_text: bool,
        prompt_text_value: str,
        cfg_value: float,
        do_normalize: bool,
        denoise: bool,
        dit_steps: int,
        seed_value,
    ):
        actual_prompt_text = prompt_text_value.strip() if use_prompt_text else ""
        actual_control = "" if use_prompt_text else control_instruction
        seed = _coerce_seed(seed_value)
        sr, wav_np, last_successful_seed = demo.generate_tts_audio(
            text_input=text,
            control_instruction=actual_control,
            reference_wav_path_input=ref_wav,
            prompt_text=actual_prompt_text,
            cfg_value_input=cfg_value,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=int(dit_steps),
            seed=seed,
        )
        return (sr, wav_np), last_successful_seed

    def _on_toggle_instant(checked):
        """Instant UI toggle — no ASR, no blocking."""
        if checked:
            return (
                gr.update(visible=True, value="", placeholder="အညွှန်းအသံကို စာသားပြောင်းနေပါတယ်..."),
                gr.update(visible=False),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True, interactive=True),
        )

    def _run_asr_if_needed(checked, audio_path):
        """Run ASR after the UI has updated. Only when toggled ON."""
        if not checked or not audio_path:
            return gr.update()
        try:
            logger.info("Running ASR on reference audio...")
            asr_text = demo.prompt_wav_recognition(audio_path)
            logger.info(f"ASR result: {asr_text[:60]}...")
            return gr.update(value=asr_text)
        except Exception as e:
            logger.warning(f"ASR recognition failed: {e}")
            return gr.update(value="")

    def _on_reference_audio_change(audio_path):
        """Automatically transcribe a newly uploaded reference audio clip."""
        if not audio_path:
            return (
                gr.update(value=False),
                gr.update(value="", visible=False, placeholder=I18N("prompt_text_placeholder")),
                gr.update(visible=True),
            )
        try:
            logger.info("Automatically transcribing uploaded reference audio...")
            asr_text = demo.prompt_wav_recognition(audio_path)
            logger.info(f"Automatic ASR result: {asr_text[:60]}...")
            return (
                gr.update(value=True),
                gr.update(value=asr_text, visible=True, placeholder=I18N("prompt_text_placeholder")),
                gr.update(visible=False),
            )
        except Exception as e:
            logger.warning(f"Automatic ASR recognition failed: {e}")
            return (
                gr.update(value=True),
                gr.update(
                    value="",
                    visible=True,
                    placeholder="အလိုအလျောက် စာသားပြောင်းခြင်း မအောင်မြင်ပါ။ စာသားကို ကိုယ်တိုင် ရေးထည့်နိုင်ပါတယ်။",
                ),
                gr.update(visible=False),
            )

    with gr.Blocks(title="KP Voice Studio") as interface:
        gr.HTML(
            '<header class="kp-shell">'
            '  <div class="kp-brand">'
            '    <img src="/gradio_api/file=assets/kp_logo.png" alt="KP လိုဂို">'
            '    <div>'
            '      <p class="kp-eyebrow">KP VOICE CLONE</p>'
            '      <h1 class="kp-title">KP</h1>'
            '      <p class="kp-subtitle">KP မှ အသံဒီဇိုင်း၊ အသံကူးယူမှုနှင့် အသံထုတ်လုပ်မှု</p>'
            '    </div>'
            '  </div>'
            '  <div class="kp-status"><span class="kp-status-dot"></span> AI အသံစနစ် အသင့်ဖြစ်ပါပြီ</div>'
            '</header>'
        )

        with gr.Row():
            with gr.Column(scale=7):
                with gr.Group(elem_classes=["kp-panel"]):
                    gr.Markdown(
                        '<div class="kp-panel-title"><span class="kp-step">၀၁</span> အညွှန်းအသံ</div>'
                        '<div class="kp-help">အသံကူးယူရန် အသံနမူနာတင်ပါ။ အသံအသစ် ဒီဇိုင်းလုပ်လိုပါက မတင်ဘဲ ထားနိုင်ပါတယ်။</div>'
                    )
                    reference_wav = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label=I18N("reference_audio_label"),
                    )
                    show_prompt_text = gr.Checkbox(
                        value=False,
                        label=I18N("show_prompt_text_label"),
                        info="အညွှန်းအသံဖိုင် တင်လိုက်တာနဲ့ အသံထဲက စကားကို စာသားအဖြစ် အလိုအလျောက် ပြောင်းပေးပါမယ်။",
                        elem_classes=["switch-toggle"],
                        visible=False,
                    )
                    prompt_text = gr.Textbox(
                        value="",
                        label=I18N("prompt_text_label"),
                        placeholder=I18N("prompt_text_placeholder"),
                        lines=2,
                        visible=False,
                    )
                    control_instruction = gr.Textbox(
                        value="",
                        label=I18N("control_label"),
                        placeholder=I18N("control_placeholder"),
                        lines=2,
                        visible=False,
                    )

                with gr.Group(elem_classes=["kp-panel"]):
                    gr.Markdown(
                        '<div class="kp-panel-title"><span class="kp-step">၀၂</span> ပြောမည့်စာသား</div>'
                        '<div class="kp-help">ရွေးချယ်ထားသောအသံက ပြောရမည့် စာသားကို ရေးထည့်ပါ။</div>'
                    )
                    text = gr.Textbox(
                        value=DEFAULT_TARGET_TEXT,
                        label=I18N("target_text_label"),
                        lines=5,
                        show_label=False,
                    )

                with gr.Group(elem_classes=["kp-panel"]):
                    with gr.Accordion(I18N("advanced_settings_title"), open=False):
                        DoDenoisePromptAudio = gr.Checkbox(
                            value=False,
                            label=I18N("ref_denoise_label"),
                            elem_classes=["switch-toggle"],
                            info=I18N("ref_denoise_info"),
                        )
                        DoNormalizeText = gr.Checkbox(
                            value=False,
                            label=I18N("normalize_label"),
                            elem_classes=["switch-toggle"],
                            info=I18N("normalize_info"),
                        )
                        cfg_value = gr.Slider(
                            minimum=1.0,
                            maximum=3.0,
                            value=2.0,
                            step=0.1,
                            label=I18N("cfg_label"),
                            info=I18N("cfg_info"),
                        )
                        dit_steps = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=10,
                            step=1,
                            label=I18N("dit_steps_label"),
                            info=I18N("dit_steps_info"),
                        )
                        with gr.Row():
                            seed_value = gr.Number(
                                value=random.randint(0, 2**32 - 1),
                                precision=0,
                                label=I18N("seed_label"),
                                info=I18N("seed_info"),
                                interactive=False,
                            )
                            random_seed = gr.Checkbox(
                                value=True,
                                label=I18N("random_seed_label"),
                                elem_classes=["switch-toggle"],
                                info=I18N("random_seed_info"),
                            )

                    run_btn = gr.Button(I18N("generate_btn"), variant="primary", size="lg", elem_classes=["kp-generate"])

            with gr.Column(scale=5):
                with gr.Group(elem_classes=["kp-panel", "kp-output"]):
                    gr.Markdown(
                        '<div class="kp-panel-title"><span class="kp-step">၀၃</span> ထုတ်ပြီးသောအသံ</div>'
                        '<div class="kp-help">ထုတ်ပြီးသောအသံကို ဒီနေရာမှာ နားထောင်နိုင်ပါတယ်။</div>'
                    )
                    audio_output = gr.Audio(label=I18N("generated_audio_label"))
                with gr.Accordion("အသံဒီဇိုင်းလမ်းညွှန်", open=False):
                    gr.Markdown(I18N("usage_instructions"))
                with gr.Accordion("အသံဖော်ပြချက် နမူနာများ", open=False):
                    gr.Markdown(I18N("examples_footer"))

        reference_wav.change(
            fn=_on_reference_audio_change,
            inputs=[reference_wav],
            outputs=[show_prompt_text, prompt_text, control_instruction],
            show_progress="full",
        )

        random_seed.change(
            fn=_on_random_seed_toggle,
            inputs=[random_seed],
            outputs=[seed_value],
        )

        run_btn.click(
            fn=_prepare_seed,
            inputs=[random_seed, seed_value],
            outputs=[seed_value],
            show_progress=False,
        ).then(
            fn=_generate,
            inputs=[
                text,
                control_instruction,
                reference_wav,
                show_prompt_text,
                prompt_text,
                cfg_value,
                DoNormalizeText,
                DoDenoisePromptAudio,
                dit_steps,
                seed_value,
            ],
            outputs=[audio_output, seed_value],
            show_progress=True,
            api_name="generate",
        )

    return interface


def run_demo(
    server_name: str = "0.0.0.0",
    server_port: int = 8808,
    show_error: bool = True,
    model_id: str = "openbmb/VoxCPM2",
    device: str = "auto",
):
    demo = VoxCPMDemo(model_id=model_id, device=device)
    interface = create_demo_interface(demo)
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
    server_name=server_name,
    server_port=server_port,
    show_error=show_error,
    i18n=I18N,
    theme=_APP_THEME,
    css=_CUSTOM_CSS,
    share=True,
    favicon_path=str(Path.cwd() / "assets" / "kp_logo.png"),
)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="openbmb/VoxCPM2",
        help="Local path or HuggingFace repo ID (default: openbmb/VoxCPM2)",
    )
    parser.add_argument("--port", type=int, default=8808, help="Server port")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address. Use 127.0.0.1 to restrict access to the local machine; "
             "the default 0.0.0.0 exposes the unauthenticated UI/API to the network (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Runtime device: auto, cpu, mps, cuda, or cuda:N (default: auto)",
    )
    args = parser.parse_args()
    run_demo(
        model_id=args.model_id,
        server_name=args.host,
        server_port=args.port,
        device=args.device,
    )
