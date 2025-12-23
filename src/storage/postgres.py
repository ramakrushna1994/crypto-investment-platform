from src.config.settings import POSTGRES

def write_spark_df(df, table):
    df.write \
        .format("jdbc") \
        .option("url", POSTGRES.jdbc_url) \
        .option("dbtable", table) \
        .option("user", POSTGRES.user) \
        .option("password", POSTGRES.password) \
        .option("driver", "org.postgresql.Driver")\
        .mode("overwrite") \
        .save()
