#!/usr/bin/env python3
"""
Push NeuralForge to 4 GitHub accounts.
"""

import os
import subprocess
import requests
import json

# Load credentials
creds = {}
with open(os.path.expanduser("~/.hermes/credentials/github-pat.env")) as f:
    for line in f:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            creds[key] = val

# Account configurations
accounts = [
    {"token_key": "GITHUB_TOKEN_1", "repo_name": "NeuralForge"},
    {"token_key": "GITHUB_TOKEN_2", "repo_name": "NeuralForge"},
    {"token_key": "GITHUB_TOKEN_3", "repo_name": "NeuralForge"},
    {"token_key": "GITHUB_TOKEN_4", "repo_name": "NeuralForge"},
]

def get_username(token):
    """Get GitHub username from token."""
    headers = {"Authorization": f"token {token}"}
    r = requests.get("https://api.github.com/user", headers=headers)
    if r.status_code == 200:
        return r.json()["login"]
    return None

def create_repo(token, repo_name, description="NeuralForge - AI Pipeline Orchestration Framework"):
    """Create a new repository."""
    headers = {"Authorization": f"token {token}"}
    data = {
        "name": repo_name,
        "description": description,
        "private": False,
        "auto_init": False,
    }
    r = requests.post("https://api.github.com/user/repos", headers=headers, json=data)
    return r.status_code in [201, 422]  # 201 = created, 422 = already exists

def push_to_repo(token, username, repo_name):
    """Push code to repository."""
    remote_name = f"github_{username}"
    remote_url = f"https://{token}@github.com/{username}/{repo_name}.git"
    
    # Remove existing remote if any
    subprocess.run(["git", "remote", "remove", remote_name], capture_output=True)
    
    # Add remote
    result = subprocess.run(
        ["git", "remote", "add", remote_name, remote_url],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  Failed to add remote: {result.stderr}")
        return False
    
    # Push
    print(f"  Pushing to {username}/{repo_name}...")
    result = subprocess.run(
        ["git", "push", remote_name, "main", "--force"],
        capture_output=True, text=True, timeout=120
    )
    
    if result.returncode == 0:
        print(f"  SUCCESS!")
        return True
    else:
        print(f"  Push failed: {result.stderr[:200]}")
        return False

def main():
    print("=" * 60)
    print("NeuralForge - GitHub Push Script")
    print("=" * 60)
    
    # Check if git repo
    result = subprocess.run(["git", "status"], capture_output=True)
    if result.returncode != 0:
        print("ERROR: Not a git repository!")
        return
    
    success_count = 0
    
    for account in accounts:
        token_key = account["token_key"]
        repo_name = account["repo_name"]
        
        token = creds.get(token_key)
        if not token:
            print(f"\nSKIP {token_key}: not found in credentials")
            continue
        
        # Get username
        username = get_username(token)
        if not username:
            print(f"\nSKIP {token_key}: auth failed")
            continue
        
        print(f"\n=== {username} / {repo_name} ===")
        
        # Create repo
        if not create_repo(token, repo_name):
            print(f"  Failed to create repository")
            continue
        
        # Push
        if push_to_repo(token, username, repo_name):
            success_count += 1
            print(f"  URL: https://github.com/{username}/{repo_name}")
    
    print("\n" + "=" * 60)
    print(f"Results: {success_count}/{len(accounts)} repositories pushed successfully")
    print("=" * 60)

if __name__ == "__main__":
    main()
