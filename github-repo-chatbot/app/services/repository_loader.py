from langchain_community.document_loaders import DirectoryLoader, TextLoader, PythonLoader
from core.utils import clone_repo
from pathlib import Path
def load_repo_documents(repo_url: str):
    repo_path = clone_repo(repo_url)
    loader = DirectoryLoader(
        path=repo_path,
        glob="**/*",
        loader_cls=lambda p: PythonLoader(p) if Path(p).suffix == '.py' else TextLoader(p),
        recursive=True,
        show_progress=True,
    )
    return loader.load()
