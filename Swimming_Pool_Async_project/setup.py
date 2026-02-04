from setuptools import setup, find_packages

setup(
    name="Swimming_Pool_Async",
    version="0.1",
    #packages=find_packages(),
    description="A package for LLM and process control",
    long_description="A longer description of your package",
    author="HaoLu",
    author_email="2012456373@qq.com",
    url="https://github.com/Minami-su/Swimming_Pool",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    install_requires=[
        # 列出你的包的依赖，例如：
        # 'numpy>=1.18.0',
        # 'pandas>=1.0.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
