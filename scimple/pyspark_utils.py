import importlib
import os
from .utils import print_markdown, type_value_checks, default, is_default
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

def to_markdown(df_or_rdd, in_range=default):
    """

    :param df_or_rdd: pyspark dataFrame or RDD (or collected)
    :param in_range: Collection of size 2
    :param columns: Collection
    :return: str
    """
    type_value_checks(df_or_rdd,
                      good_types={pyspark.RDD, pyspark.sql.DataFrame, list},
                      type_message="Parameter should be pyspark RDD or DataFrame or list of Rows")

    if isinstance(df_or_rdd, pyspark.sql.DataFrame):
        df = df_or_rdd.collect()
    elif isinstance(df_or_rdd, pyspark.RDD):
        df = df_or_rdd.toDF().collect()
    else:
        df = df_or_rdd

    if isinstance(df[0], pyspark.Row):
        columns = df[0].asDict().keys()
    else:
        columns = [i for i in range(len(df[0]))]
    res = "|"
    under = "|"
    for col in columns:
        res += f"{col}|"
        under += "--|"
    res += "\n" + under
    for row in df if is_default(in_range) else df[in_range[0]: in_range[1]]:
        res += "\n|" + "|".join([f"`{row[col]}`" for col in columns]) + "|"
    return res


def show(df_or_rdd, in_range=default):
    """
    :param df_or_rdd: pyspark dataFrame or RDD
    :param range_: 2 values unpackable: indexes of data to print
    :return: None
    """
    print_markdown(to_markdown(df_or_rdd, in_range=in_range))

def contexts():
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
