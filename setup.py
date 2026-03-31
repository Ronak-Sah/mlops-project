import setuptools

with open("README.md","r",encoding="utf-8") as fobj:
    long_description=fobj.read()


__version__="1.0.0"
REPO_NAME="mlops-project"
AUTHOR_USER_NAME="Ronak-Sah"
AUTHOR_EMAIL="ronaksah777@gmail.com"

setuptools.setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description= "A small package mlops.",
    long_description=long_description,
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    packages=setuptools.find_packages(),
    install_requires=[]

)