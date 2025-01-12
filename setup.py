from setuptools import setup, find_packages

setup(
    name='dorie',
    version='0.1.0',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    entry_points={
        'console_scripts': [
            # Add command line scripts here
            # e.g., 'dorie=dorie.cli:main'
        ],
    },
    author='Steven Loaiza',
    author_email='your.email@example.com',
    description='DORIE (Dynamic Omnichannel RoBERTa Intent Engine) is an advanced natural language processing system designed for multi-channel intent classification. Built on RoBERTa architecture, it provides enterprise-grade NLP capabilities for automated response handling.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/loaizasteven/dorie',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache License 2.0',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
