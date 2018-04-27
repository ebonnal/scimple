from setuptools import setup

setup(
    name='scimple',
    version='1.6',
    py_modules=['scimple'],
    install_requires=['matplotlib', 'numpy', 'pandas'],
    packages=['scimple_data'],
    package_data={'scimple_data': ['*']},
    url='http://github.com/EnzoBnl/Scimple',
    license='',
    author='Enzo Bonnal',
    author_email='enzobonnal@gmail.com',
    description='Parse and Plot your data in 2 lines of code'
)
