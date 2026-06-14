"""文本分块器：滑动窗口将长文档切分为语义完整的文本块"""
from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    """一个文本块及其元数据"""
    content: str           # 块文本内容
    chunk_index: int       # 块在文档中的全局序号
    start_char: int        # 在原文中的起始字符位置
    end_char: int          # 在原文中的结束字符位置
    source_page: int = -1  # 来源页码（-1 表示未知）
    chunk_type: str = "text"  # text / image_caption / section_title


def split_text_with_overlap(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    source_page: int = -1,
    chunk_type: str = "text"
) -> List[TextChunk]:
    """
    滑动窗口分块。

    为什么需要 overlap？
    — 如果一个完整的句子被切到两个块的边界，没有 overlap 的话，
      任何一个块的语义都不完整，检索时无法命中。
    — overlap=64 意味着相邻块共享约 2-3 个句子，确保边界语义不丢失。

    为什么 chunk_size=512？
    — 512 字符约 150-200 个英文单词，包含足够语义信息又不会太长导致检索噪声。
      BGE-m3 最大输入 8192 tokens，512 字符远小于限制，留出余量。
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        chunks.append(TextChunk(
            content=chunk_text,
            chunk_index=chunk_index,
            start_char=start,
            end_char=min(end, len(text)),
            source_page=source_page,
            chunk_type=chunk_type
        ))

        # 已经到末尾，停止
        if end >= len(text):
            break

        # 步长 = chunk_size - chunk_overlap
        start += chunk_size - chunk_overlap
        chunk_index += 1

    return chunks
