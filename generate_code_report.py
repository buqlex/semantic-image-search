import os

EXCLUDED_DIRS = {'.git', '.venv', '__pycache__', '.mypy_cache'}
OUTPUT_FILE = 'project_code_report.txt'
PROJECT_ROOT = '.'


def get_project_structure(base_dir):
    tree_lines = []

    for root, dirs, files in os.walk(base_dir):
        # Пропускаем скрытые и системные директории
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in EXCLUDED_DIRS]

        level = root.replace(base_dir, '').count(os.sep)
        indent = '├── ' if level else ''
        folder = os.path.basename(root) if root != '.' else os.path.basename(os.getcwd())
        tree_lines.append(f"{'│   ' * (level - 1)}{indent}{folder}/")

        for f in sorted(files):
            if f.endswith('.py') or f.endswith('.bin') or f.endswith('.sqlite3'):
                file_indent = '│   ' * level + '└── '
                tree_lines.append(f"{file_indent}{f}")

    return '\n'.join(tree_lines)


def collect_python_files(base_dir):
    py_files = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in EXCLUDED_DIRS]
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                py_files.append(full_path)
    return py_files


def write_code_report():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        # Добавим структуру проекта
        out.write("Project Structure:\n\n")
        out.write(get_project_structure(PROJECT_ROOT))
        out.write("\n\n")

        # Добавим содержимое каждого .py файла
        py_files = collect_python_files(PROJECT_ROOT)
        for file_path in py_files:
            relative_path = os.path.relpath(file_path, PROJECT_ROOT)
            out.write(f"\n\n{relative_path}:\n\n")
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                out.write(code)
                out.write("\n" + "-" * 80)

    print(f"✅ Файл отчета с кодом сохранён в: {OUTPUT_FILE}")


if __name__ == '__main__':
    write_code_report()
