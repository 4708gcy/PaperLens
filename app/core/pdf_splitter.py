"""
PDF 拆分器 + 文档格式转换

两个职责：
1. format_converter：非 PDF（DOC/DOCX/PPT/PPTX/...）→ PDF（用 LibreOffice）
2. split_pdf：超长 PDF（>200 页）→ 多个子 PDF（用 pypdf），每 200 页一份

为什么先用 LibreOffice 转 PDF 再交给 MinerU？
— MinerU open-api 虽然直接支持 DOCX/PPTX，但对 PDF 的解析质量最高、最稳
— LibreOffice headless 模式转 PDF 零成本、保真度高
— 转成统一 PDF 后，后续分块、向量化、图表提取全部走同一条路径，简化代码
"""
import subprocess
import shutil
import sys
from pypdf import PdfReader, PdfWriter
from pathlib import Path
from typing import Optional, List


# MinerU open-api 单文件上限 200 页
MAX_PAGES_PER_CHUNK = 200

# 支持的所有输入格式（除 PDF 外，可被 LibreOffice 转 PDF 的）
SUPPORTED_NON_PDF = {".doc", ".docx", ".ppt", ".pptx", ".html", ".htm",
                     ".odt", ".ods", ".odp", ".rtf"}

# soffice 命令名（Windows 下可能需完整路径，建议已加入 PATH）
SOFFICE_CMD = "soffice"


def _resolve_cmd(cmd_name: str) -> Optional[str]:
    """
    探测命令的完整可执行路径。

    背景：Windows 下 uvicorn 的 BackgroundTasks 线程池 worker 可能不继承
    conda 环境激活时的 PATH，导致 subprocess 找不到 mineru-open-api / soffice。
    本函数依次尝试：shutil.which → 常见安装位置 → 返回 None。

    返回：完整路径（可直接传给 subprocess）或 None。
    """
    # 1. 先用 shutil.which（最可靠，若 PATH 有则直接返回）
    found = shutil.which(cmd_name)
    if found:
        return found

    # 2. 兜底：在 conda 环境目录下找（uvicorn 后台线程可能 PATH 缺失）
    # Windows 上 conda 环境根目录、Scripts、node_modules/.bin 都可能放命令
    candidates = []
    if sys.platform == "win32":
        # 从当前 Python 解释器推断 conda 环境根目录
        # sys.executable = .../envs/ocr/python.exe → .parent = envs/ocr/
        py_exe = sys.executable
        env_root = str(Path(py_exe).parent)  # envs/ocr/
        for ext in (".cmd", ".exe", ".bat", ""):
            # conda 环境根目录（npm 全局装的话会在这里生成 .cmd 包装）
            candidates.append(Path(env_root) / f"{cmd_name}{ext}")
            candidates.append(Path(env_root) / "Scripts" / f"{cmd_name}{ext}")
            candidates.append(Path(env_root) / "node_modules" / ".bin" / f"{cmd_name}{ext}")
            # node 包装的二进制（如 mineru-open-api-win32-x64）
            candidates.append(Path(env_root) / "node_modules" / cmd_name / "node_modules" /
                              f"{cmd_name}-win32-x64" / "bin" / f"{cmd_name}.exe")
    else:
        for env_root in ["/opt/conda", "/root/miniconda3", str(Path.home() / "miniconda3")]:
            candidates.append(Path(env_root) / "envs" / "ocr" / "bin" / cmd_name)

    for c in candidates:
        if c.exists():
            return str(c)

    return None


def convert_to_pdf(input_path: str, output_dir: Optional[str] = None) -> str:
    """
    用 LibreOffice 把非 PDF 文档转为 PDF。

    返回：生成的 PDF 路径。如果输入已是 PDF，原样返回。
    """
    src = Path(input_path)
    if src.suffix.lower() == ".pdf":
        return input_path

    if src.suffix.lower() not in SUPPORTED_NON_PDF:
        raise ValueError(f"不支持的格式: {src.suffix}")

    soffice_path = _resolve_cmd(SOFFICE_CMD)
    if not soffice_path:
        raise RuntimeError(
            f"'{SOFFICE_CMD}' 未找到！请确认 LibreOffice 已安装并加入 PATH，"
            "或装在 conda ocr 环境下。"
        )

    out_dir = str(output_dir or src.parent.resolve())
    cmd = [
        soffice_path,
        "--headless", "--invisible",
        "--convert-to", "pdf",
        "--outdir", out_dir,
        str(src.resolve())
    ]
    print(f"  [LibreOffice] {' '.join(cmd)}")

    try:
        # Windows 下若命令是 .cmd/.bat 包装脚本，必须 shell=True
        use_shell = sys.platform == "win32" and soffice_path.lower().endswith((".cmd", ".bat"))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            shell=use_shell
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"LibreOffice 转换失败 (exit {result.returncode}): {result.stderr[:300]}"
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError("LibreOffice 转换超时（>120s），文件可能过大或损坏")

    pdf_path = str(src.with_suffix(".pdf"))
    if not Path(pdf_path).exists():
        raise RuntimeError(f"LibreOffice 输出 PDF 未找到: {pdf_path}")
    return pdf_path


def split_pdf(pdf_path: str, max_pages: int = MAX_PAGES_PER_CHUNK) -> List[str]:
    """
    将 PDF 按 max_pages 拆分为多个子文件。

    返回子文件路径列表。不需要拆分时返回 [pdf_path] 本身。
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    if total_pages <= max_pages:
        return [pdf_path]

    base_name = str(Path(pdf_path).with_suffix(""))
    output_paths = []

    for i in range(0, total_pages, max_pages):
        writer = PdfWriter()
        end = min(i + max_pages, total_pages)

        for page_num in range(i, end):
            writer.add_page(reader.pages[page_num])

        part_num = i // max_pages + 1
        output_path = f"{base_name}_part_{part_num}.pdf"

        with open(output_path, "wb") as f:
            writer.write(f)

        output_paths.append(output_path)
        print(f"  [拆分] {Path(output_path).name}: 第 {i+1}-{end} 页 / 共 {total_pages} 页")

    return output_paths


def is_pdf(file_path: str) -> bool:
    """判断文件是否为 PDF"""
    return Path(file_path).suffix.lower() == ".pdf"


def get_pdf_page_count(pdf_path: str) -> int:
    """获取 PDF 页数"""
    return len(PdfReader(pdf_path).pages)
