
import os
import logging
import sys
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# 将项目根目录加入 sys.path，保证包内引用可用（以便从项目根运行脚本）
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir)) # 调试断点，检查路径设置是否正确
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
from utils.config import config

# 单次调用超时（秒）。注意：initialize_llm/get_single_llm 内部有局部变量 config
# 会遮蔽上面的模块级 config，因此这里先取成模块常量再使用。
_LLM_TIMEOUT = config.LLM_TIMEOUT

# 设置日志模版。首次 basicConfig 决定全应用根 logger 级别，
# 因此这里读 LOG_LEVEL 环境变量（默认 DEBUG），让各流程节点的调试日志可见。
_LEVEL = getattr(logging, str(os.getenv("LOG_LEVEL", "DEBUG")).upper(), logging.DEBUG)
logging.basicConfig(level=_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.debug("日志级别已设为 %s", logging.getLevelName(_LEVEL))

# 模型配置字典：API key 与 base_url 均从 .env 读取（key 名与 .env 一致）
MODEL_CONFIGS = {
    "openai": {
        "base_url": os.getenv("LAOZHANG_BASE_URL", "https://api.laozhang.ai/v1"),
        "api_key": os.getenv("LAOZHANG_API_KEY"),
        "chat_model": "gpt-4o-mini"
    },
    "qwen": {
        "base_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "api_key": os.getenv("QWEN_API_KEY"),
        "chat_model": "qwen-max",
    },
    "deepseek": {
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "chat_model": "deepseek-chat",
    },
    "vllm": {
        "base_url": "http://ai.bygpu.com:58132/v1",
        "api_key": "vllm",
        "chat_model": "Qwen/Qwen3-4B",
    }
}


# 默认配置
DEFAULT_LLM_TYPE = "openai"
DEFAULT_TEMPERATURE = 0.0


class LLMInitializationError(Exception):
    """自定义异常类用于LLM初始化错误"""
    pass


def get_embedding_model() -> HuggingFaceEmbeddings:
    """返回固定的 bge 中文 embedding 模型（768 维，CPU，归一化）。

    长期记忆的语义检索等场景复用同一个模型，避免重复初始化。
    """
    model_name = r"C:\Users\qian gao\models\BAAI\bge-base-zh-v1___5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )


def initialize_llm(llm_type: str = DEFAULT_LLM_TYPE) -> tuple[ChatOpenAI, HuggingFaceBgeEmbeddings]:
    """
    初始化LLM实例

    Args:
        llm_type (str): LLM类型，可选值为 'openai', 'oneapi', 'qwen', 'ollama'

    Returns:
        ChatOpenAI: 初始化后的LLM实例

    Raises:
        LLMInitializationError: 当LLM初始化失败时抛出
    """
    try:
        # 检查llm_type是否有效
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"不支持的LLM类型: {llm_type}. 可用的类型: {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[llm_type]

        # 特殊处理 ollama 类型
        if llm_type == "vllm":
            os.environ["OPENAI_API_KEY"] = "NA"

        # 创建LLM实例
        llm_chat = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["chat_model"],
            temperature=DEFAULT_TEMPERATURE,
            timeout=_LLM_TIMEOUT,  # 超时配置（秒），见 config.LLM_TIMEOUT
            max_retries=2  # 添加重试次数
        )
        
        # 默认为768维的向量，复用固定的 bge 中文 embedding 模型
        llm_embedding = get_embedding_model()
        logger.info(f"成功初始化 {llm_type} LLM")
        # return llm_chat
        return llm_chat, llm_embedding

    except ValueError as ve:
        logger.error(f"LLM配置错误: {str(ve)}")
        raise LLMInitializationError(f"LLM配置错误: {str(ve)}")
    except Exception as e:
        logger.error(f"初始化LLM失败: {str(e)}")
        raise LLMInitializationError(f"初始化LLM失败: {str(e)}")


def get_single_llm(llm_type: str = DEFAULT_LLM_TYPE):
    try:
        # 检查llm_type是否有效
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"不支持的LLM类型: {llm_type}. 可用的类型: {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[llm_type]

        # 特殊处理 ollama 类型
        if llm_type == "vllm":
            os.environ["OPENAI_API_KEY"] = "NA"

        # 创建LLM实例
        llm_chat = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["chat_model"],
            temperature=DEFAULT_TEMPERATURE,
            timeout=_LLM_TIMEOUT,  # 超时配置（秒），见 config.LLM_TIMEOUT
            max_retries=2  # 添加重试次数
        )
        logger.info(f"成功初始化 {llm_type} LLM")
        return llm_chat
    except ValueError as ve:
        logger.error(f"LLM配置错误: {str(ve)}")
        raise LLMInitializationError(f"LLM配置错误: {str(ve)}")
    except Exception as e:
        logger.error(f"初始化LLM失败: {str(e)}")
        raise LLMInitializationError(f"初始化LLM失败: {str(e)}")

def get_llm(llm_type: str = DEFAULT_LLM_TYPE) -> tuple[ChatOpenAI, HuggingFaceBgeEmbeddings]:
    """
    获取LLM实例的封装函数，提供默认值和错误处理

    Args:
        llm_type (str): LLM类型

    Returns:
        ChatOpenAI: LLM实例
    """
    try:
        return initialize_llm(llm_type)
    except LLMInitializationError as e:
        logger.warning(f"使用默认配置重试: {str(e)}")
        if llm_type != DEFAULT_LLM_TYPE:
            return initialize_llm(DEFAULT_LLM_TYPE)
        raise  # 如果默认配置也失败，则抛出异常


async def acall_with_fallback(messages, llm_chain: list[str] | None = None):
    """按优先级链调用 LLM，某一家失败（无 key / 超时 / 报错）自动切换下一家。

    默认回退链来自 config.LLM_FALLBACK_CHAIN（qwen -> deepseek -> openai）。
    全部失败时抛出 LLMInitializationError。
    """
    chain = llm_chain or config.LLM_FALLBACK_CHAIN
    last_err = None
    for llm_type in chain:
        try:
            llm = get_single_llm(llm_type)
            logger.info("使用 %s 调用 LLM", llm_type)
            return await llm.ainvoke(messages)
        except Exception as e:  # noqa: BLE001 - 需要拦截所有失败以尝试下一家
            last_err = e
            logger.warning("LLM %s 调用失败：%s，切换下一家", llm_type, e)
    raise LLMInitializationError(
        f"回退链 {chain} 全部调用失败，最后错误：{last_err}"
    )


# 示例使用
if __name__ == "__main__":
    try:
        # 测试不同类型的LLM初始化
        # llm_openai = get_llm("openai")
        llm_chat, llm_embedding = get_llm("openai")
        llm = get_single_llm("openai")
        print(llm.invoke(["Hello, world!"]))
        # 测试embedding生成
        embeddings = llm_embedding.embed_documents(["Hello, world!", "How are you?"])
        print(len(embeddings), len(embeddings[0]))  # 输出嵌入的数量和维度
        # llm_invalid = get_llm("invalid_type")
    except LLMInitializationError as e:
        logger.error(f"程序终止: {str(e)}")
