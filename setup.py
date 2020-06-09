from setuptools import setup

setup(
    name='scimple',
    version='1.11.3',
    #py_modules=['scimple'],
    install_requires=['matplotlib', 'numpy', 'pandas', 'pyspark', 'pyarrow', 'networkx'],
    packages=['scimple/scimple_data', 'scimple'],
    package_data={'scimple/scimple_data': ['*']},
    url='http://github.com/EnzoBnl/Scimple',
    license='',  # MIT
    author='Enzo Bonnal',
    author_email='enzobonnal@gmail.com',
    description='Scimplify Ploting, Graph manipulation, Spark Streaming & Kafka and other tools'
)
