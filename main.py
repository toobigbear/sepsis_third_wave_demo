# -*- coding: utf-8 -*-
"""
sepsis_third_wave_demo —— 重症 AI 第三波浪潮仿真 DEMO
========================================================
核心思想来源：《Intensive Care Medicine》2026 述评（Elbers / Ercole / Angus）：
    第三波 = 在线强化学习 + 床旁自适应随机化
    有把握时直接推荐最优方案；没把握时在临床均势选项池内自适应随机；
    观察结局后持续更新策略，让"照护本身成为持续学习的系统"。

硬隔离原则（需求文档强制）：
    贝叶斯多臂老虎机（MAB）决策逻辑 100% 由 numpy 独立计算；
    本地大模型（LM-Studio）仅做结果自然语言解读；
    向量检索仅输出参考证据；
    二者绝对不干预、不修改 MAB 的决策输出。

⚠️ 学术仿真原型：不接入任何真实患者与医院系统，全部为模拟数据。
"""

import csv
import json
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ------------------------------------------------------------------
# 路径与常量
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
AUDIT_PATH = DATA_DIR / "audit_log.jsonl"
DEMO_CASES_PATH = DATA_DIR / "demo_cases.csv"

LMSTUDIO_BASE = "http://127.0.0.1:1234/v1"
LLM_MODELS = ["qwen3-8b", "qwen/qwen3.5-9b"]
# 用户 LM-Studio 中可用的 embedding 模型 id（按顺序尝试）
EMBED_MODEL_CANDIDATES = [
    "nomic-ai/text-embedding-nomic-embed-text-v1.5",
    "yorkieoh10/text-embedding-nomic-embed-text-v1.5",
]

app = FastAPI(title="sepsis_third_wave_demo", docs_url=None, redoc_url=None)


# ------------------------------------------------------------------
# 配置加载
# ------------------------------------------------------------------
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


POOL_CONFIG = load_json(CONFIG_DIR / "pool_config.json")
KNOWLEDGE_BASE = load_json(CONFIG_DIR / "sepsis_knowledge.json")["knowledge_base"]

# ------------------------------------------------------------------
# 审计日志：每条操作追加一行 JSON，不覆盖旧记录
# ------------------------------------------------------------------
def write_audit(event: dict):
    """追加写入审计日志（JSONL）。任何失败只打印警告，绝不影响主流程。"""
    record = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **event}
    try:
        with open(AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[audit] 写入失败：{exc}")


# ==================================================================
# 核心算法内核：贝叶斯多臂老虎机（Thompson 采样，纯 numpy）
# ==================================================================
class ThompsonBandit:
    """
    简化版贝叶斯多臂老虎机：
      - 每个臂的收益服从 Bernoulli 分布，后验为 Beta(alpha, beta)，先验 Beta(1,1)（均匀先验）；
      - Thompson 采样：从每个臂的后验中抽一个 theta，选 theta 最大的臂；
      - 不确定性分数（0-100）：用蒙特卡洛估计"当前最优臂的置信度"，
        uncertainty = 100 * (1 - |2 * P(最优臂) - 1|)。
        P=0.5（完全分不清优劣）→ 100；P=1（确定）→ 0。
        该定义对临床更直观：数值越高 = 当前亚组证据越不足。
    """

    def __init__(self, arm_ids):
        self.arm_ids = list(arm_ids)
        self.reset()

    def reset(self):
        self.alpha = {a: 1.0 for a in self.arm_ids}
        self.beta = {a: 1.0 for a in self.arm_ids}
        self.trials = {a: 0 for a in self.arm_ids}
        self.total_trials = 0

    def sample(self):
        """执行一次 Thompson 采样，返回 (选中臂id, 各臂本次采样值)。"""
        draws = {a: float(np.random.beta(self.alpha[a], self.beta[a])) for a in self.arm_ids}
        chosen = max(draws, key=draws.get)
        return chosen, draws

    def uncertainty(self, n_mc: int = 20000) -> float:
        """蒙特卡洛估计 P(最优臂)，换算成 0-100 决策不确定性。"""
        p_best = {a: 0 for a in self.arm_ids}
        # 一次性向量化采样，速度快
        for a in self.arm_ids:
            samples = np.random.beta(self.alpha[a], self.beta[a], size=n_mc)
            if a == self.arm_ids[0]:
                stacked = samples
            else:
                stacked = np.vstack([stacked, samples])
        best_idx = np.argmax(stacked, axis=0)
        for i, a in enumerate(self.arm_ids):
            p_best[a] = float(np.mean(best_idx == i))
        p = p_best[self.arm_ids[0]]
        unc = 100.0 * (1.0 - abs(2.0 * p - 1.0))
        return round(min(100.0, max(0.0, unc)), 1)

    def posterior_stats(self, n_mc: int = 20000):
        """返回每个臂的后验期望收益与 95% 区间（蒙特卡洛分位数，避免依赖 scipy）。"""
        stats = {}
        for a in self.arm_ids:
            samples = np.random.beta(self.alpha[a], self.beta[a], size=n_mc)
            stats[a] = {
                "mean": round(float(np.mean(samples)), 3),
                "ci_low": round(float(np.percentile(samples, 2.5)), 3),
                "ci_high": round(float(np.percentile(samples, 97.5)), 3),
                "alpha": round(self.alpha[a], 2),
                "beta": round(self.beta[a], 2),
                "trials": self.trials[a],
            }
        return stats

    def update(self, arm: str, reward: float):
        """结局反馈：按奖励值更新后验。reward ∈ [0,1]。"""
        reward = float(np.clip(reward, 0.0, 1.0))
        self.alpha[arm] += reward
        self.beta[arm] += (1.0 - reward)
        self.trials[arm] += 1
        self.total_trials += 1


# ==================================================================
# 虚拟患者模拟器（numpy 生成，不依赖任何外部数据源）
# ==================================================================
class PatientSim:
    """
    生成一位虚拟脓毒症休克患者"初始复苏完成后"的病程时序。
    隐藏一个"潜在反应类型"（补液反应型 / 升压药反应型 / 混合型），
    使不同患者的指标走向有差异——用于演示"亚组证据不足→随机探索"的思想。
    """

    RESPONDER_TYPES = ["补液反应型", "升压药反应型", "混合型"]

    def __init__(self, case_id: str, age: int, sex: str, seed: int):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.case_id = case_id
        self.age = int(age)
        self.sex = sex
        self.responder_type = str(rng.choice(self.RESPONDER_TYPES))
        self.step = 0
        self.history = []
        # 初始状态：初始复苏刚完成（对应述评场景演练 T1 决策点）
        self.map = float(rng.normal(63, 3))
        self.lactate = float(rng.normal(4.2, 0.5))
        self.urine = float(rng.normal(22, 5))          # ml/h
        self.ne_dose = float(round(rng.normal(0.25, 0.03), 2))  # μg/kg/min
        self.sofa = int(rng.integers(7, 11))
        self._push_record()

    def _push_record(self):
        rec = {
            "step": self.step,
            "map": round(self.map, 1),
            "lactate": round(self.lactate, 2),
            "urine": round(self.urine, 1),
            "ne_dose": round(self.ne_dose, 2),
            "sofa": self.sofa,
        }
        self.history.append(rec)
        return rec

    def next_step(self):
        """推进一个时间步：按潜在反应类型产生不同漂移 + 噪声。"""
        self.step += 1
        r = self.rng
        if self.responder_type == "补液反应型":
            map_drift, lac_drift, urine_drift = +4.5, -0.55, +6.0
        elif self.responder_type == "升压药反应型":
            map_drift, lac_drift, urine_drift = +2.0, -0.25, +2.0
        else:  # 混合型：恢复慢
            map_drift, lac_drift, urine_drift = +1.0, -0.10, +1.0
        # 随时间递减的恢复效应 + 噪声
        decay = 1.0 / (1.0 + 0.15 * self.step)
        self.map = float(np.clip(self.map + map_drift * decay + r.normal(0, 2), 45, 110))
        self.lactate = float(np.clip(self.lactate + lac_drift * decay + r.normal(0, 0.25), 0.8, 12))
        self.urine = float(np.clip(self.urine + urine_drift * decay + r.normal(0, 3), 0, 200))
        self.ne_dose = float(np.clip(self.ne_dose - 0.01 + r.normal(0, 0.015), 0.05, 1.2))
        if r.random() < 0.25:
            self.sofa = int(np.clip(self.sofa + r.integers(-1, 2), 1, 20))
        return self._push_record()

    def info(self):
        return {
            "case_id": self.case_id,
            "age": self.age,
            "sex": self.sex,
            "diagnosis": "脓毒症休克（模拟）",
            "responder_type": self.responder_type,  # 仅演示可见，真实系统不可观测
            "current_step": self.step,
            "history": self.history,
        }


# ==================================================================
# 会话状态（单机单会话演示）
# ==================================================================
STATE = {
    "session_id": None,
    "mode": None,            # "auto" | "csv"
    "patient": None,         # PatientSim 实例
    "bandit": ThompsonBandit([o["id"] for o in POOL_CONFIG["options"]]),
    "last_decision": None,   # 最近一次决策记录（结局反馈时更新对应臂）
    "safety_gates": {"contraindication": False, "emergency": False, "palliative": False},
    "csv_cases": {},         # 回放模式病例数据
    "csv_case_id": None,
}

# ==================================================================
# LM-Studio 客户端（可选增强；异常时友好降级，绝不崩溃主流程）
# ==================================================================
def lmstudio_client():
    """惰性创建 openai 客户端。LM-Studio 未启动时返回 None。"""
    try:
        from openai import OpenAI
        return OpenAI(base_url=LMSTUDIO_BASE, api_key="lm-studio", timeout=60.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[llm] 客户端创建失败：{exc}")
        return None


def probe_lmstudio():
    """健康探测：/v1/models 是否可达。页面加载时调用一次。"""
    client = lmstudio_client()
    if client is None:
        return {"connected": False, "models": [], "message": "openai 客户端不可用"}
    try:
        models = client.models.list()
        ids = [m.id for m in models.data]
        return {"connected": True, "models": ids,
                "message": "已连接" if ids else "已连接但无已加载模型"}
    except Exception as exc:  # noqa: BLE001
        return {"connected": False, "models": [],
                "message": f"未连接：LM-Studio 未启动或未开启 API 服务（{type(exc).__name__}）"}


LLM_PROMPT_TEMPLATE = (
    "你是 ICU 重症临床助理。下面给出贝叶斯多臂老虎机仿真计算结果，"
    "你只对结果做临床层面解释，严禁修改、质疑算法输出的动作结论。"
    "用重症医生能看懂的简洁中文，不要编造新治疗方案。\n"
    "患者状态：{patient}\n"
    "不确定性分数：{score}\n"
    "分支类型：{branch}\n"
    "算法输出动作：{action}\n"
    "临床均势选项池：{pool}"
)


def llm_interpret(patient_text: str, score: float, branch_text: str,
                  action_text: str, pool_text: str, model: str) -> str:
    """调用本地 LLM 生成临床解读。任何异常抛出由上层捕获并友好提示。"""
    client = lmstudio_client()
    if client is None:
        raise RuntimeError("openai 客户端不可用")
    prompt = LLM_PROMPT_TEMPLATE.format(
        patient=patient_text, score=score, branch=branch_text,
        action=action_text, pool=pool_text)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是严谨的 ICU 重症临床助理，只做结果解读，严禁修改算法结论。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


# ---------- 向量检索（内存 numpy，无第三方向量库） ----------
_KB_VECTORS = None  # 启动后一次性向量化并缓存


def embed_texts(texts, embed_model: str | None = None):
    """调用 LM-Studio embedding 接口。按候选列表自动尝试已加载模型。"""
    client = lmstudio_client()
    if client is None:
        raise RuntimeError("openai 客户端不可用")
    candidates = ([embed_model] if embed_model else []) + EMBED_MODEL_CANDIDATES
    last_err = None
    for m in dict.fromkeys(candidates):  # 去重保序
        try:
            resp = client.embeddings.create(model=m, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(f"embedding 调用失败（已尝试 {candidates}）：{last_err}")


def cosine_topk(query_vec, matrix, k=2):
    """numpy 余弦相似度，返回 Top-k 下标与分数。"""
    q = np.asarray(query_vec, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    sims = m @ q
    idx = np.argsort(-sims)[:k]
    return [(int(i), round(float(sims[i]), 4)) for i in idx]


def retrieve_evidence(query_text: str, embed_model: str | None = None, k: int = 2):
    """检索 Top-k 指南证据。知识库首次使用时一次性向量化并缓存。"""
    global _KB_VECTORS
    if _KB_VECTORS is None:
        vectors = embed_texts([f"{item['text']}（来源：{item['source']}）" for item in KNOWLEDGE_BASE])
        _KB_VECTORS = np.asarray(vectors, dtype=np.float64)
    qv = embed_texts([query_text])[0]
    hits = cosine_topk(qv, _KB_VECTORS, k=k)
    results = []
    for i, score in hits:
        item = KNOWLEDGE_BASE[i]
        results.append({"id": item["id"], "source": item["source"],
                        "text": item["text"], "score": score})
    return results


# ==================================================================
# 请求体定义
# ==================================================================
class StartSessionReq(BaseModel):
    mode: str                       # "auto" | "csv"
    case_id: str | None = None      # csv 模式必填
    seed: int | None = None


class DecideReq(BaseModel):
    threshold: float = 60.0
    llm_enabled: bool = False       # 仅记录开关状态，不影响决策


class VetoReq(BaseModel):
    reason: str
    action: str                     # "A" | "B"


class SafetyReq(BaseModel):
    contraindication: bool = False
    emergency: bool = False
    palliative: bool = False


class LLMReq(BaseModel):
    model: str = "qwen3-8b"


class OutcomeReq(BaseModel):
    survival_28d: bool
    lactate_drop: float             # mmol/L 下降幅度
    vasopressor_reduced: bool
    icu_days: float
    weights: dict | None = None     # {"survival":..,"lactate_improvement":..,"vasopressor_reduction":..}


# ==================================================================
# 工具函数
# ==================================================================
def load_demo_cases():
    """读取 data/demo_cases.csv，按 case_id 分组缓存。"""
    cases = {}
    if not DEMO_CASES_PATH.exists():
        return cases
    with open(DEMO_CASES_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["case_id"]
            cases.setdefault(cid, {"case_id": cid, "age": int(row["age"]),
                                   "sex": row["sex"], "steps": [], "outcome": None})
            step_rec = {
                "step": int(row["step"]),
                "map": float(row["map"]),
                "lactate": float(row["lactate"]),
                "urine": float(row["urine"]),
                "ne_dose": float(row["ne_dose"]),
                "sofa": int(row["sofa"]),
            }
            cases[cid]["steps"].append(step_rec)
            # 末行带结局字段 → 作为该病例的预设结局（回放时预填表单）
            if row.get("outcome_survival"):
                cases[cid]["outcome"] = {
                    "survival_28d": row["outcome_survival"].strip() == "1",
                    "lactate_drop": float(row["lactate_drop"] or 0),
                    "vasopressor_reduced": row["vasopressor_reduced"].strip() == "1",
                    "icu_days": float(row["icu_days"] or 0),
                    "last_action": row.get("last_action", "").strip(),
                }
    for c in cases.values():
        c["steps"].sort(key=lambda x: x["step"])
    return cases


def compute_reward(outcome: OutcomeReq) -> float:
    """奖励函数：三个维度的加权和，归一化到 [0,1]。"""
    w = dict(POOL_CONFIG["reward_weights"])
    if outcome.weights:
        w.update({k: float(v) for k, v in outcome.weights.items()})
    w_surv = w.get("survival", 0.5)
    w_lac = w.get("lactate_improvement", 0.3)
    w_vaso = w.get("vasopressor_reduction", 0.2)
    total = w_surv + w_lac + w_vaso
    if total <= 0:
        total = 1.0
    reward = (w_surv * (1.0 if outcome.survival_28d else 0.0)
              + w_lac * min(max(outcome.lactate_drop, 0.0) / 2.0, 1.0)
              + w_vaso * (1.0 if outcome.vasopressor_reduced else 0.0)) / total
    return round(float(reward), 4)


def patient_snapshot_text() -> str:
    """把当前患者状态拼成一行文本，供 LLM 解读与向量检索共用。"""
    p = STATE["patient"]
    if p is None or not p.history:
        return "（无患者数据）"
    cur = p.history[-1]
    return (f"{p.age}岁{p.sex}性，脓毒症休克（初始复苏完成后第{cur['step']}时间步）："
            f"MAP {cur['map']} mmHg，乳酸 {cur['lactate']} mmol/L，"
            f"尿量 {cur['urine']} ml/h，去甲肾上腺素 {cur['ne_dose']} μg/kg/min，"
            f"SOFA {cur['sofa']} 分。")


def pool_text() -> str:
    return "；".join(f"{o['id']}={o['short']}" for o in POOL_CONFIG["options"])


# ==================================================================
# API：健康与配置
# ==================================================================
@app.get("/api/health")
def api_health():
    """LM-Studio 连接状态探测（页面加载时调用一次）。"""
    return probe_lmstudio()


@app.get("/api/config")
def api_config():
    return {
        "options": POOL_CONFIG["options"],
        "decision_point": POOL_CONFIG["decision_point"],
        "default_threshold": POOL_CONFIG["default_threshold"],
        "reward_weights": POOL_CONFIG["reward_weights"],
        "llm_models": LLM_MODELS,
        "embed_models": EMBED_MODEL_CANDIDATES,
        "notes": POOL_CONFIG.get("notes", ""),
    }


@app.get("/api/cases")
def api_cases():
    """列出 CSV 回放模式的全部模拟病例（含结局摘要，便于选择）。"""
    cases = load_demo_cases()
    overview = []
    for cid, c in sorted(cases.items()):
        overview.append({
            "case_id": cid,
            "age": c["age"],
            "sex": c["sex"],
            "steps": len(c["steps"]),
            "has_outcome": c["outcome"] is not None,
        })
    return {"cases": overview}


# ==================================================================
# API：会话与患者
# ==================================================================
@app.post("/api/session/start")
def api_start_session(req: StartSessionReq):
    """初始化会话：重置老虎机、创建/载入虚拟患者。"""
    if req.mode not in ("auto", "csv"):
        raise HTTPException(400, "mode 必须是 auto 或 csv")
    STATE["session_id"] = f"S-{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}"
    STATE["mode"] = req.mode
    STATE["bandit"].reset()
    STATE["last_decision"] = None
    STATE["safety_gates"] = {"contraindication": False, "emergency": False, "palliative": False}

    if req.mode == "auto":
        seed = req.seed if req.seed is not None else int(time.time()) % 100000
        sim = PatientSim(case_id=f"SIM-{seed}", age=int(np.random.default_rng(seed).integers(35, 85)),
                         sex=str(np.random.default_rng(seed + 1).choice(["男", "女"])), seed=seed)
        STATE["patient"] = sim
        STATE["csv_case_id"] = None
        msg = "已生成一位虚拟脓毒症休克患者（自动生成模式）"
    else:
        cases = load_demo_cases()
        if not cases:
            raise HTTPException(400, "data/demo_cases.csv 不存在或为空")
        cid = req.case_id or sorted(cases.keys())[0]
        if cid not in cases:
            raise HTTPException(400, f"病例 {cid} 不存在")
        c = cases[cid]
        sim = PatientSim(case_id=cid, age=c["age"], sex=c["sex"], seed=0)
        sim.history = [dict(s) for s in c["steps"]]
        sim.step = sim.history[-1]["step"] if sim.history else 0
        STATE["patient"] = sim
        STATE["csv_case_id"] = cid
        STATE["csv_cases"] = cases
        msg = f"已载入回放病例 {cid}（共 {len(sim.history)} 个时间步）"

    write_audit({"session_id": STATE["session_id"], "event": "session_start",
                 "mode": req.mode, "patient_id": sim.case_id, "detail": msg})
    return {"session_id": STATE["session_id"], "patient": sim.info(),
            "csv_outcome": (STATE["csv_cases"].get(req.case_id) or {}).get("outcome")
            if req.mode == "csv" else None,
            "message": msg}


@app.get("/api/patient/state")
def api_patient_state():
    if STATE["patient"] is None:
        raise HTTPException(400, "请先初始化会话（选择模式并开始）")
    return STATE["patient"].info()


@app.post("/api/patient/next_step")
def api_next_step():
    """推进模拟患者病程一个时间步（回放模式：仅当还有未展示的步骤）。"""
    if STATE["patient"] is None:
        raise HTTPException(400, "请先初始化会话")
    p = STATE["patient"]
    if STATE["mode"] == "csv":
        # 回放模式：CSV 的所有步骤已在载入时放入 history，
        # 这里按"尚未展示游标"推进，最多到病例末步。
        shown = getattr(p, "_shown_steps", 1)
        if shown < len(p.history):
            p._shown_steps = shown + 1  # noqa: SLF001
        rec = p.history[p._shown_steps - 1]
        return {"current": rec, "history": p.history[:p._shown_steps],
                "finished": p._shown_steps >= len(p.history),
                "responder_type": p.responder_type}
    rec = p.next_step()
    return {"current": rec, "history": p.history,
            "finished": p.step >= 8, "responder_type": p.responder_type}


# ==================================================================
# API：核心决策（MAB 内核，numpy 独立计算）
# ==================================================================
@app.post("/api/decide")
def api_decide(req: DecideReq):
    if STATE["patient"] is None:
        raise HTTPException(400, "请先初始化会话")
    bandit = STATE["bandit"]
    chosen, draws = bandit.sample()
    uncertainty = bandit.uncertainty()
    threshold = float(np.clip(req.threshold, 0, 100))
    # 两条分支严格还原述评原文：高置信→推荐；低置信→自适应随机
    branch = "recommend" if uncertainty < threshold else "randomize"
    branch_text = "高置信推荐" if branch == "recommend" else "低置信自适应随机"
    options = {o["id"]: o for o in POOL_CONFIG["options"]}
    chosen_option = options[chosen]

    decision = {
        "uncertainty": uncertainty,
        "threshold": threshold,
        "branch": branch,
        "branch_text": branch_text,
        "chosen_arm": chosen,
        "chosen_action": chosen_option["name"],
        "posterior": bandit.posterior_stats(),
        "draws": {k: round(v, 4) for k, v in draws.items()},
        "gates": dict(STATE["safety_gates"]),
    }
    STATE["last_decision"] = decision
    write_audit({
        "session_id": STATE["session_id"], "event": "decision",
        "patient_id": STATE["patient"].case_id,
        "uncertainty": uncertainty, "threshold": threshold,
        "branch": branch_text, "system_action": chosen,
        "llm_enabled": req.llm_enabled,
    })
    return decision


# ==================================================================
# API：医生否决 / 安全闸门（状态记录）
# ==================================================================
@app.post("/api/veto")
def api_veto(req: VetoReq):
    if STATE["last_decision"] is None:
        raise HTTPException(400, "请先执行决策计算")
    options = {o["id"]: o for o in POOL_CONFIG["options"]}
    if req.action not in options:
        raise HTTPException(400, "否决动作必须是选项池内的 A 或 B")
    options_short = {o["id"]: o["short"] for o in POOL_CONFIG["options"]}
    veto_record = {
        "session_id": STATE["session_id"], "event": "veto",
        "patient_id": STATE["patient"].case_id if STATE["patient"] else None,
        "original_action": STATE["last_decision"]["chosen_arm"],
        "veto_action": req.action,
        "veto_action_name": options[req.action]["name"],
        "reason": req.reason,
        "note": "医生否决：最终执行动作以医生为准",
    }
    # 否决后，结局反馈更新到医生指定的臂
    STATE["last_decision"]["chosen_arm"] = req.action
    STATE["last_decision"]["chosen_action"] = options[req.action]["name"]
    STATE["last_decision"]["branch"] = "veto"
    STATE["last_decision"]["branch_text"] = "医生手动否决"
    write_audit(veto_record)
    return {"ok": True, "message": f"已记录否决，最终动作改为：{options_short[req.action]}",
            "audit": veto_record}


@app.post("/api/safety")
def api_safety(req: SafetyReq):
    """安全闸门状态写入审计日志。DEMO 仅界面模拟，不做医学逻辑校验。"""
    STATE["safety_gates"] = {
        "contraindication": req.contraindication,
        "emergency": req.emergency,
        "palliative": req.palliative,
    }
    names = {"contraindication": "禁忌条件（肢端缺血模拟剔除）",
             "emergency": "紧急急救模式（关闭随机探索分支）",
             "palliative": "姑息治疗目标切换（退出决策输出）"}
    detail = "；".join(f"{names[k]}:{'开' if v else '关'}"
                       for k, v in STATE["safety_gates"].items())
    write_audit({"session_id": STATE["session_id"], "event": "safety_gates",
                 "patient_id": STATE["patient"].case_id if STATE["patient"] else None,
                 "detail": detail,
                 "note": "UI 模拟，无医学判断逻辑"})
    return {"ok": True, "gates": dict(STATE["safety_gates"]), "detail": detail}


# ==================================================================
# API：本地 LLM 解读 / 向量检索（可选增强）
# ==================================================================
@app.post("/api/llm/interpret")
def api_llm_interpret(req: LLMReq):
    """LLM 仅对 MAB 结果做自然语言解读，不参与决策。"""
    if STATE["last_decision"] is None:
        raise HTTPException(400, "请先执行决策计算")
    d = STATE["last_decision"]
    try:
        text = llm_interpret(
            patient_text=patient_snapshot_text(),
            score=d["uncertainty"],
            branch_text=d["branch_text"],
            action_text=d["chosen_action"],
            pool_text=pool_text(),
            model=req.model,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "text": "",
                "message": f"本地大模型调用失败：{exc}（请确认 LM-Studio 已启动且模型已加载）"}
    write_audit({"session_id": STATE["session_id"], "event": "llm_interpret",
                 "patient_id": STATE["patient"].case_id if STATE["patient"] else None,
                 "model": req.model, "branch": d["branch_text"],
                 "system_action": d["chosen_arm"], "llm_text": text})
    return {"ok": True, "text": text, "model": req.model}


@app.post("/api/retrieve")
def api_retrieve(req: LLMReq):
    """向量检索 Top-2 指南证据。仅参考，不改变算法决策。"""
    if STATE["last_decision"] is None:
        raise HTTPException(400, "请先执行决策计算")
    d = STATE["last_decision"]
    query = (f"脓毒症休克复苏决策：{patient_snapshot_text()} "
             f"算法分支：{d['branch_text']}，动作：{d['chosen_action']}。"
             f"查询相关指南证据。")
    try:
        results = retrieve_evidence(query)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "results": [],
                "message": f"向量检索失败：{exc}（请确认 LM-Studio 已加载 nomic-embed 模型）"}
    write_audit({"session_id": STATE["session_id"], "event": "retrieve",
                 "patient_id": STATE["patient"].case_id if STATE["patient"] else None,
                 "branch": d["branch_text"], "system_action": d["chosen_arm"],
                 "evidence": results})
    return {"ok": True, "results": results}


# ==================================================================
# API：结局反馈 & 策略更新（仿真闭环）
# ==================================================================
@app.post("/api/outcome")
def api_outcome(req: OutcomeReq):
    if STATE["last_decision"] is None:
        raise HTTPException(400, "请先执行决策计算再提交结局")
    bandit = STATE["bandit"]
    arm = STATE["last_decision"]["chosen_arm"]
    reward = compute_reward(req)
    before = bandit.posterior_stats()
    bandit.update(arm, reward)
    after = bandit.posterior_stats()
    STATE["last_decision"] = None  # 本次决策闭环完成

    record = {
        "session_id": STATE["session_id"], "event": "outcome",
        "patient_id": STATE["patient"].case_id if STATE["patient"] else None,
        "arm_updated": arm, "reward": reward,
        "outcome": {"survival_28d": req.survival_28d,
                    "lactate_drop": req.lactate_drop,
                    "vasopressor_reduced": req.vasopressor_reduced,
                    "icu_days": req.icu_days},
        "weights": req.weights or POOL_CONFIG["reward_weights"],
        "policy_updated": True,
    }
    write_audit(record)
    return {
        "ok": True,
        "reward": reward,
        "arm_updated": arm,
        "before": before,
        "after": after,
        "message": f"已将奖励 {reward} 回传至臂 {arm} 的后验分布，策略已更新。",
    }


@app.get("/api/bandit/state")
def api_bandit_state():
    """当前老虎机后验状态 + 决策不确定性（供状态面板进度条使用）。"""
    return {"posterior": STATE["bandit"].posterior_stats(),
            "uncertainty": STATE["bandit"].uncertainty(),
            "total_trials": STATE["bandit"].total_trials}


# ==================================================================
# API：审计日志
# ==================================================================
@app.get("/api/audit")
def api_audit():
    records = []
    if AUDIT_PATH.exists():
        with open(AUDIT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return {"records": records[-300:]}  # 最多返回最近 300 条，避免页面过长


@app.get("/api/audit/export")
def api_audit_export():
    if not AUDIT_PATH.exists():
        raise HTTPException(404, "暂无审计日志")
    return FileResponse(AUDIT_PATH, media_type="application/jsonl",
                        filename="audit_log.jsonl")


@app.post("/api/audit/clear")
def api_audit_clear():
    """清空当前会话日志（演示用）。删除旧文件并写一个清空标记。"""
    try:
        if AUDIT_PATH.exists():
            AUDIT_PATH.unlink()
        write_audit({"event": "audit_cleared", "session_id": STATE["session_id"],
                     "note": "日志已清空（演示功能）"})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"清空失败：{exc}")
    return {"ok": True, "message": "审计日志已清空"}


# ==================================================================
# 前端静态页
# ==================================================================
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print(" sepsis_third_wave_demo 启动中：http://127.0.0.1:8000")
    print(" 学术仿真原型，不构成任何诊疗建议")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
