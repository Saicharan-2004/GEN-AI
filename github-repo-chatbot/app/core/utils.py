import os
from git import Repo

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLONE_DIR = os.path.join(BASE_DIR, "data", "cloned_repo")

def clone_repo(repo_url: str, dest_folder: str = CLONE_DIR):
    if os.path.exists(dest_folder):
        return dest_folder #Recursively deletes the directory if its already present.
    Repo.clone_from(repo_url, dest_folder)
    return dest_folder
