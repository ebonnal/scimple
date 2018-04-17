from setuptools import setup

setup(
    name='scimple',
    version='1.6',
    packages=['scimple_data'],
	py_modules=['scimple'],
    install_requires=['matplotlib','numpy','mpl_toolkits'],
	package_data={'scimple_data': ['*']},
    url='http://github.com/EnzoBnl/Scimple',
    license='',
    author='Enzo Bonnal',
    author_email='enzobonnal@gmail.com',
    description='Plot your data in 3 lines'
)
