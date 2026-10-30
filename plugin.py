"""百度千帆联网搜索插件

给 Planner 提供 web_search 工具，并在结果后注入可编辑的提示词，
避免模型把搜索结果整段念给用户。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import httpx

from maibot_sdk import Field, MaiBotPlugin, PluginConfigBase
from maibot_sdk.components import Tool
from maibot_sdk.types import ToolParamType, ToolParameterInfo

# ── 默认值 ────────────────────────────────────────────────────────────

_DEFAULT_BASE_URL = "https://qianfan.baidubce.com"

_DEFAULT_AFTER_SEARCH = (
    "（以上是搜索结果，不是你说的话。你要做的是：把里面跟当前聊天有关的一两点，"
    "用你自己的口语说出来，别念标题、别列条目、别贴链接、别说「根据搜索」。"
    "没用的就别提，一句带过也行。）"
)

_DEFAULT_TOOL_DESC = (
    "联网搜索。当需要查实时信息、新闻、价格、作品资料等你不确定的事时使用。"
    "拿到结果后用自己的话说，不要念原文。"
)

# 面向模型的错误文案：只说结论，不暴露上游细节
_ERR_NO_KEY = "搜索不可用，直接说不知道。"
_ERR_TIMEOUT = "搜索超时了，直接说没查到。"
_ERR_HTTP = "搜索出错了，直接说没查到。"
_ERR_JSON = "搜索结果异常，直接说没查到。"
_ERR_EMPTY = "没搜到相关内容，直接说不知道。"


# ── 配置 ──────────────────────────────────────────────────────────────


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(default=True, description="是否启用插件")
    config_version: str = Field(default="1.0.0", description="配置版本")


class QianfanSectionConfig(PluginConfigBase):
    """千帆接口配置。"""

    __ui_label__ = "接口"
    __ui_icon__ = "key"
    __ui_order__ = 1

    api_key: str = Field(
        default="",
        description="百度千帆 API Key（bce-v3/ALTAK-... 格式）",
        json_schema_extra={"label": "API Key", "placeholder": "bce-v3/ALTAK-xxx/xxx"},
    )
    base_url: str = Field(
        default=_DEFAULT_BASE_URL,
        description="接口根地址，一般不用改",
        json_schema_extra={"label": "接口地址"},
    )
    timeout: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="单次请求超时（秒）",
        json_schema_extra={"label": "超时（秒）", "x-widget": "slider", "step": 5},
    )
    retry: int = Field(
        default=1,
        ge=0,
        le=3,
        description="失败重试次数（0 = 不重试）",
        json_schema_extra={"label": "重试次数", "x-widget": "slider", "step": 1},
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="一次最多返回几条搜索结果",
        json_schema_extra={"label": "结果条数", "x-widget": "slider", "step": 1},
    )
    preview_chars: int = Field(
        default=260,
        ge=100,
        le=2000,
        description="每条结果最多截取多少字给模型看",
        json_schema_extra={"label": "每条截取字数", "x-widget": "slider", "step": 20},
    )
    include_url: bool = Field(
        default=True,
        description="结果里是否带上来源链接；关掉更省 token，也更不容易被念出来",
        json_schema_extra={"label": "附带来源链接"},
    )
    dedupe: bool = Field(
        default=True,
        description="是否去掉标题或内容重复的结果",
        json_schema_extra={"label": "结果去重"},
    )
    cache_seconds: int = Field(
        default=60,
        ge=0,
        le=600,
        description="相同搜索词的缓存时间（秒）；0 = 不缓存",
        json_schema_extra={"label": "缓存秒数", "x-widget": "slider", "step": 10},
    )


class ToolSectionConfig(PluginConfigBase):
    """工具开关与描述。"""

    __ui_label__ = "工具"
    __ui_icon__ = "wrench"
    __ui_order__ = 2

    enable_search: bool = Field(
        default=True,
        description="是否提供联网搜索工具",
        json_schema_extra={"label": "启用搜索"},
    )
    search_description: str = Field(
        default=_DEFAULT_TOOL_DESC,
        description="搜索工具的描述，会展示给模型（改完需重载插件）",
        json_schema_extra={"label": "搜索工具描述"},
    )


class PromptSectionConfig(PluginConfigBase):
    """结果后注入的提示词。"""

    __ui_label__ = "结果提示词"
    __ui_icon__ = "message"
    __ui_order__ = 3

    enabled: bool = Field(
        default=True,
        description="是否在搜索结果后附加提示词（强烈建议开）",
        json_schema_extra={"label": "启用结果提示词"},
    )
    after_search: str = Field(
        default=_DEFAULT_AFTER_SEARCH,
        description="搜索完成后附在结果末尾的提示词，用来防止模型把结果整段念出来",
        json_schema_extra={"label": "搜索后提示词"},
    )


class QianfanSearchConfig(PluginConfigBase):
    """插件总配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    qianfan: QianfanSectionConfig = Field(default_factory=QianfanSectionConfig)
    tools: ToolSectionConfig = Field(default_factory=ToolSectionConfig)
    prompt: PromptSectionConfig = Field(default_factory=PromptSectionConfig)


# ── 插件主体 ──────────────────────────────────────────────────────────


class QianfanSearchPlugin(MaiBotPlugin):
    """百度千帆联网搜索。"""

    config_model = QianfanSearchConfig

    def __init__(self) -> None:
        super().__init__()
        self._calls = 0
        self._failures = 0
        # (query) -> (到期时间戳, 结果文本)
        self._cache: Dict[str, Tuple[float, str]] = {}
        self._lock: Optional[asyncio.Lock] = None

    # ── 生命周期 ──────────────────────────────────────────────────────

    async def on_load(self) -> None:
        cfg = self.config.qianfan
        self._lock = asyncio.Lock()
        self.ctx.logger.info(
            f"[千帆搜索] 已加载：搜索={self.config.tools.enable_search} "
            f"结果提示词={self.config.prompt.enabled} "
            f"key={'已填' if cfg.api_key.strip() else '未填'}"
        )
        self._log_prompt_preview()

    async def on_unload(self) -> None:
        self._cache.clear()
        self.ctx.logger.info(
            f"[千帆搜索] 已卸载，本次共调用 {self._calls} 次，失败 {self._failures} 次"
        )

    async def on_config_update(self, scope: str, config_data: dict[str, object], version: str) -> None:
        del scope, config_data, version
        self._cache.clear()
        self.ctx.logger.info("[千帆搜索] 配置已刷新，缓存已清空")
        self._log_prompt_preview()

    def _log_prompt_preview(self) -> None:
        """把当前注入提示词打到控制台，方便核对。"""

        cfg = self.config
        if not cfg.prompt.enabled:
            self.ctx.logger.info("[千帆搜索] 结果提示词：已关闭（搜索结果会原样交给模型）")
            return

        after = cfg.prompt.after_search.strip()
        if after:
            self.ctx.logger.info(f"[千帆搜索] 搜索后注入（{len(after)} 字）：{after}")
        else:
            self.ctx.logger.info("[千帆搜索] 搜索后注入：未填写")

        self.ctx.logger.info(
            f"[千帆搜索] 参数：条数={cfg.qianfan.max_results} "
            f"每条截取={cfg.qianfan.preview_chars} 字 "
            f"超时={cfg.qianfan.timeout} 秒 "
            f"重试={cfg.qianfan.retry} 次 "
            f"缓存={cfg.qianfan.cache_seconds} 秒 "
            f"链接={'带' if cfg.qianfan.include_url else '不带'}"
        )

    # ── HTTP ──────────────────────────────────────────────────────────

    def _endpoint(self) -> str:
        """搜索接口地址；base_url 必须以 http(s) 开头且无凭据。"""

        cfg = self.config.qianfan
        base = (cfg.base_url or _DEFAULT_BASE_URL).strip().rstrip("/")
        parts = urlsplit(base)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise RuntimeError("base_url 无效")
        return base + "/v2/ai_search"

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发一次请求，失败按配置重试。"""

        cfg = self.config.qianfan
        key = cfg.api_key.strip()
        if not key:
            raise RuntimeError("no_key")

        url = self._endpoint()
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "MaiBot-QianfanSearch/1.0",
        }

        attempts = max(0, int(cfg.retry)) + 1
        last_error = "unknown"

        for i in range(attempts):
            try:
                async with httpx.AsyncClient(
                    timeout=cfg.timeout, follow_redirects=False
                ) as client:
                    resp = await client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException:
                last_error = "timeout"
                self.ctx.logger.warning(
                    f"[千帆搜索] 超时（第 {i + 1}/{attempts} 次）"
                )
                continue
            except httpx.HTTPError as exc:
                last_error = "http"
                self.ctx.logger.warning(
                    f"[千帆搜索] 连接失败（第 {i + 1}/{attempts} 次）：{type(exc).__name__}"
                )
                continue

            if resp.status_code >= 500:
                last_error = "http"
                self.ctx.logger.warning(
                    f"[千帆搜索] 服务端 {resp.status_code}（第 {i + 1}/{attempts} 次）"
                )
                continue

            if not resp.is_success:
                # 4xx 不重试：通常是 key 错或参数错
                self.ctx.logger.warning(
                    f"[千帆搜索] HTTP {resp.status_code}：{resp.text[:200]}"
                )
                raise RuntimeError("http")

            try:
                data = resp.json()
            except Exception:
                last_error = "json"
                self.ctx.logger.warning("[千帆搜索] 返回非 JSON")
                continue

            if not isinstance(data, dict):
                last_error = "json"
                continue

            err = data.get("error")
            if err:
                msg = err.get("message") if isinstance(err, dict) else str(err)
                self.ctx.logger.warning(f"[千帆搜索] 业务错误：{msg}")
                raise RuntimeError("http")

            return data

        raise RuntimeError(last_error)

    # ── 工具：搜索 ────────────────────────────────────────────────────

    @Tool(
        "web_search",
        description=_DEFAULT_TOOL_DESC,
        parameters=[
            ToolParameterInfo(
                name="query",
                param_type=ToolParamType.STRING,
                description="要搜索的内容，用简短的关键词或一句话",
                required=True,
            ),
        ],
    )
    async def web_search(self, query: str, **kwargs: Any) -> str:
        """联网搜索。"""

        del kwargs
        cfg = self.config
        query = (query or "").strip()
        if not query:
            return "搜索词是空的，直接说不知道。"

        # 缓存
        now = time.time()
        ttl = max(0, int(cfg.qianfan.cache_seconds))
        if ttl > 0:
            hit = self._cache.get(query)
            if hit and hit[0] > now:
                self.ctx.logger.info(f"[千帆搜索] 命中缓存：{query}")
                return hit[1]

        # 并发保护：同一时刻只允许一个请求
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            self._calls += 1
            try:
                data = await self._post(
                    {
                        "messages": [{"role": "user", "content": query}],
                        "stream": False,
                    }
                )
            except RuntimeError as exc:
                self._failures += 1
                code = str(exc)
                mapping = {
                    "no_key": _ERR_NO_KEY,
                    "timeout": _ERR_TIMEOUT,
                    "http": _ERR_HTTP,
                    "json": _ERR_JSON,
                }
                return mapping.get(code, _ERR_EMPTY)

        refs = data.get("references")
        if not isinstance(refs, list) or not refs:
            return _ERR_EMPTY

        body = self._format(query, refs)
        if ttl > 0:
            self._cache[query] = (now + ttl, body)
            # 简单清理过期项
            if len(self._cache) > 50:
                self._cache = {
                    k: v for k, v in self._cache.items() if v[0] > now
                }
        return body

    def _format(self, query: str, refs: List[Any]) -> str:
        """把结果格式化成给模型看的文本，并在末尾注入提示词。"""

        cfg = self.config
        limit = int(cfg.qianfan.max_results)
        cut = int(cfg.qianfan.preview_chars)
        want_url = bool(cfg.qianfan.include_url)
        do_dedupe = bool(cfg.qianfan.dedupe)

        seen_titles: set = set()
        seen_contents: set = set()
        lines: List[str] = [f"搜索结果（{query}）："]
        used = 0

        for item in refs:
            if used >= limit:
                break
            if not isinstance(item, dict):
                continue

            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            content = str(item.get("content") or "").strip()
            date = str(item.get("date") or "").strip()

            if do_dedupe:
                title_key = title[:40]
                content_key = content[:80]
                if title_key and title_key in seen_titles:
                    continue
                if content_key and content_key in seen_contents:
                    continue
                if title_key:
                    seen_titles.add(title_key)
                if content_key:
                    seen_contents.add(content_key)

            if len(content) > cut:
                content = content[:cut] + "…"

            used += 1
            head = f"{used}. {title}"
            if date:
                head += f"（{date}）"
            lines.append(head)
            if content:
                lines.append(f"   {content}")
            if want_url and url:
                lines.append(f"   来源：{url}")

        if used == 0:
            return _ERR_EMPTY

        body = "\n".join(lines)
        after = cfg.prompt.after_search.strip() if cfg.prompt.enabled else ""
        if after:
            body += "\n\n" + after
            self.ctx.logger.info(
                f"[千帆搜索] 搜索「{query}」：{used} 条结果 "
                f"+ 注入提示词 {len(after)} 字（总计 {len(body)} 字）"
            )
            self.ctx.logger.info(f"[千帆搜索] 注入内容：{after}")
        else:
            self.ctx.logger.info(
                f"[千帆搜索] 搜索「{query}」：{used} 条结果"
                f"（未注入提示词，总计 {len(body)} 字）"
            )
        return body


def create_plugin() -> QianfanSearchPlugin:
    """插件入口。"""

    return QianfanSearchPlugin()
