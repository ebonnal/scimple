import importlib
import os

# #####
# PYSPARK
# #####
_sc = None
_spark = None

try:
    import pyspark
    os.environ['SPARK_HOME'] = pyspark.__file__[:pyspark.__file__.index('__init__.py') - 1]
    importlib.reload(pyspark)
except:
    pass


def pyspark_contexts():
    """

    :return: (SparkContext, SQLContext)
    """
    try:
        import pyspark
        from pyspark.sql import SparkSession
        sc = pyspark.SparkContext.getOrCreate()
        sqlc = pyspark.SQLContext.getOrCreate(sc)
        sqlc.setConf("sqlc.sql.execution.arrow.enabled", "true")
        return sc, sqlc
    except:
        raise Warning('pyspark not available')
