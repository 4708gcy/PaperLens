"""临时验证脚本：测试 chunker 和 pdf_splitter"""
from app.core.chunker import split_text_with_overlap
from app.core.pdf_splitter import (
    is_pdf, split_pdf, get_pdf_page_count,
    SUPPORTED_NON_PDF, MAX_PAGES_PER_CHUNK
)
import shutil


def test_chunker():
    # 基本分块
    text = 'Attention is all you need. ' * 100
    chunks = split_text_with_overlap(text, chunk_size=200, chunk_overlap=30)
    assert len(chunks) > 5, f'分块数太少: {len(chunks)}'
    print(f'[PASS] 基本分块: {len(text)} 字符 -> {len(chunks)} 块')

    # overlap 验证
    assert chunks[1].start_char < chunks[0].end_char, 'overlap 错误'
    print(f'[PASS] overlap 正确: 块0结束 {chunks[0].end_char} > 块1开始 {chunks[1].start_char}')

    # 空文本
    assert split_text_with_overlap('') == []
    assert split_text_with_overlap('   ') == []
    print('[PASS] 空文本处理')

    # 短文本
    short = split_text_with_overlap('短文本', chunk_size=512, chunk_overlap=64)
    assert len(short) == 1
    assert short[0].content == '短文本'
    print('[PASS] 短文本处理')


def test_pdf_splitter():
    assert MAX_PAGES_PER_CHUNK == 200, f'页数上限错误: {MAX_PAGES_PER_CHUNK}'
    print(f'[PASS] MAX_PAGES_PER_CHUNK = {MAX_PAGES_PER_CHUNK}')

    assert is_pdf('test.pdf') is True
    assert is_pdf('test.docx') is False
    assert is_pdf('test.PDF') is True
    print('[PASS] is_pdf 判断')

    assert '.docx' in SUPPORTED_NON_PDF
    assert '.pptx' in SUPPORTED_NON_PDF
    print(f'[PASS] 支持格式: {sorted(SUPPORTED_NON_PDF)}')

    assert shutil.which('soffice'), 'soffice 不在 PATH'
    print(f'[PASS] soffice 可用: {shutil.which("soffice")}')


if __name__ == '__main__':
    print('=== 测试 chunker ===')
    test_chunker()
    print('\n=== 测试 pdf_splitter ===')
    test_pdf_splitter()
    print('\n[ALL PASSED] chunker + pdf_splitter 全部通过')
