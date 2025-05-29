from setuptools import setup

setup(
    name="claude-cli",
    version="0.1.0",
    py_modules=["AnthropicCLI"],
    install_requires=["click", "anthropic"],
    entry_points={
        "console_scripts": [
            "claude=AnthropicCLI:main",
        ],
    },
)