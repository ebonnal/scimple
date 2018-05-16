from setuptools import setup

setup(
    name='scimple',
    version='1.9.0',
    py_modules=['scimple'],
    install_requires=['matplotlib', 'numpy', 'pandas'],
    packages=['scimple_data'],
    package_data={'scimple_data': ['*']},
    url='http://github.com/EnzoBnl/Scimple',
    license='',  # MIT
    author='Enzo Bonnal',
    author_email='enzobonnal@gmail.com',
    description='Plot your data scimply in 1 line'
)
