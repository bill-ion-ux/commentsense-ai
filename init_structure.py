import os

# This is to init the files that we need based on the README
# requirements

folders = [
    "src/commentsense/config",
    "src/commentsense/data",
    "src/commentsense/features",
    "src/commentsense/models",
    "src/commentsense/pipelines",
    "src/commentsense/services",
    "src/commentsense/dashboard",
    "src/commentsense/utils",
    "src/commentsense/assets",
    "data/raw", "data/interim", "data/processed", "data/external",
    "models/artifacts", "notebooks", "scripts", "tests"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    # add __init__.py for Python packages
    if "src/commentsense" in folder and "assets" not in folder:
        open(os.path.join(folder, "__init__.py"), "a").close()

print("✅ Project structure initialized")
