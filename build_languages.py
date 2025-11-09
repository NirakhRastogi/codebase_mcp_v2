import os
from tree_sitter import Language

os.makedirs("build", exist_ok=True)

Language.build_library(
    "build/my-languages.so",
    [
        "grammars/tree-sitter-javascript",
        "grammars/tree-sitter-typescript/typescript",
        "grammars/tree-sitter-typescript/tsx",
        "grammars/tree-sitter-python",
        "grammars/tree-sitter-json",
        "grammars/tree-sitter-yaml",
        "grammars/tree-sitter-java",
        "grammars/tree-sitter-kotlin",
        'grammars/tree-sitter-css'
    ],
)

print("✅ Successfully built build/my-languages.so with Java + Kotlin + others")
