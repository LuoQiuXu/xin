"""Read-only local dataset inventory; write fresh reports outside source data."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REQUIRED_CSV = ("ISIC_2019_Training_Metadata.csv", "ISIC_2019_Training_GroundTruth.csv")


def inspect(root: Path, inspection_date: str) -> dict:
    errors, csv_files, links, fish_directories, isic_directories = [], [], [], [], []
    count = 0
    fish = root / "fishRecognition_GT"
    isic = root / "ISIC_2019_Training_Input"
    if not root.is_dir():
        raise FileNotFoundError(root)
    for directory, children, filenames in os.walk(root, followlinks=False, onerror=errors.append):
        directory = Path(directory)
        children.sort()
        filenames.sort()
        count += len(filenames)
        for child in list(children):
            path = directory / child
            if path.is_symlink() or path.is_junction():
                links.append(str(path))
                children.remove(child)
        for name in filenames:
            if Path(name).suffix.lower() == ".csv":
                path = directory / name
                csv_files.append({"path": str(path.resolve()), "bytes": path.stat().st_size})
        if directory.is_relative_to(fish) or directory.is_relative_to(isic):
            image_names = [name for name in filenames if Path(name).suffix.lower() in IMAGE_EXTENSIONS]
            row = {
                "path": str(directory), "direct_files": len(filenames),
                "formats": dict(sorted(Counter(Path(name).suffix.lower() for name in filenames).items())),
                "image_files": len(image_names), "image_example": image_names[0] if image_names else None,
            }
            (fish_directories if directory.is_relative_to(fish) else isic_directories).append(row)
    return {
        "data_root": str(root), "inspection_date": inspection_date,
        "total_file_entries": count, "csv_files": csv_files,
        "required_csv_matches": {
            name: [row["path"] for row in csv_files if Path(row["path"]).name.casefold() == name.casefold()]
            for name in REQUIRED_CSV
        },
        "fish_directories": fish_directories, "isic_directories": isic_directories,
        "directory_links": links, "traversal_errors": [str(error) for error in errors],
        "search_complete_without_links_or_errors": not errors and not links,
        "read_only_source_data": True,
    }


def render(result: dict) -> str:
    complete = result["search_complete_without_links_or_errors"]
    lines = [
        "# DATASET_DIRECTORY_INSPECTION", "", f"检查日期：{result['inspection_date']}（Asia/Shanghai）。",
        "", f"数据根目录：`{result['data_root']}`。使用 os.walk 递归全部子目录（含隐藏项），按大小写不敏感的 .csv 扩展名筛选。",
        f"共检查 {result['total_file_entries']:,} 个文件条目；遍历错误 {len(result['traversal_errors'])}；目录链接 {len(result['directory_links'])}。",
        "本次只读检查目录、文件名和 CSV 大小，没有移动、删除、改写或下载数据。", "",
        "## 全部 CSV", "", "| 绝对路径 | 字节数 |", "|---|---:|",
    ]
    for row in result["csv_files"]:
        lines.append(f"| `{row['path']}` | {row['bytes']} |")
    if not result["csv_files"]:
        lines.append("| 未找到 | — |")
    lines += ["", "## ISIC 必需 CSV", ""]
    for name, matches in result["required_csv_matches"].items():
        state = "; ".join(f"`{path}`" for path in matches) if matches else ("缺失：完整递归搜索未发现" if complete else "未发现，但遍历不完整，不能确认缺失")
        lines.append(f"- `{name}`：{state}。")
    lines += ["", "其他数据集的 CSV 不能替代 ISIC 标签或 lesion_id 元数据。", "", "## Fish4Knowledge 完整目录结构", "",
              "下表列出该目录本身及全部子目录；路径相对于数据根目录。图像与 mask 分开计数，mask 不作为分类图像输入。", "",
              "| 相对目录 | 直接图像文件数 | 扩展名及文件数 | 文件名样例 |", "|---|---:|---|---|"]
    for row in result["fish_directories"]:
        rel = Path(row["path"]).relative_to(Path(result["data_root"])).as_posix()
        formats = ", ".join(f"{ext}={n}" for ext, n in row["formats"].items()) or "—"
        lines.append(f"| `{rel}` | {row['image_files']} | {formats} | `{row['image_example'] or '—'}` |")
    lines += ["", "## ISIC 图像目录", "", "| 绝对目录 | 直接图像文件数 | 扩展名及文件数 | 图像文件名样例 |", "|---|---:|---|---|"]
    for row in result["isic_directories"]:
        formats = ", ".join(f"{ext}={n}" for ext, n in row["formats"].items()) or "—"
        lines.append(f"| `{row['path']}` | {row['image_files']} | {formats} | `{row['image_example'] or '—'}` |")
    lines += ["", "目录检查仅确认数量、路径与扩展名；不将扩展名或文件数视为解码、八类标签、病灶独立性或无泄漏的证明。",
              "尚未提供的必需 CSV 会阻止 ISIC 完整审计；没有据此启动训练。", ""]
    if not complete:
        lines += ["## 遍历限制", "", json.dumps({"errors": result["traversal_errors"], "links": result["directory_links"]}, ensure_ascii=False, indent=2), ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--inspection-date", required=True, help="Local date in Asia/Shanghai, YYYY-MM-DD")
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    outputs = [Path(args.output_json).resolve(), Path(args.output_report).resolve()]
    for output in outputs:
        if output.is_relative_to(root) or output.exists():
            raise ValueError(f"Output must be new and outside source data: {output}")
    result = inspect(root, args.inspection_date)
    for output, text in zip(outputs, (json.dumps(result, ensure_ascii=False, indent=2), render(result))):
        with output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    print(json.dumps({key: result[key] for key in ("data_root", "total_file_entries", "csv_files", "required_csv_matches", "search_complete_without_links_or_errors")}, ensure_ascii=False, indent=2))
    return 0 if result["search_complete_without_links_or_errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
