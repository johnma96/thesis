from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lit, countDistinct, sum, add_months
from delta.tables import DeltaTable
import sys

# Inicializar Spark
spark = SparkSession.builder.getOrCreate()

# Parametrizar el mes (formato yyyy-MM)
if len(sys.argv) != 2:
    print("Uso: script.py <mes_proceso: yyyy-MM>")
    sys.exit(1)

mes_proceso = sys.argv[1]
fecha_inicio = f"{mes_proceso}-01"
fecha_fin = f"{mes_proceso}-31"

print(f"Procesando mes: {mes_proceso}")

# Leer datos de transacciones del mes
df_transacciones = (
    spark.read.table("tu_base_datos.transacciones")
    .filter((col("fecha") >= fecha_inicio) & (col("fecha") <= fecha_fin))
)

if df_transacciones.isEmpty():
    print(f"No hay datos para el mes {mes_proceso}")
    sys.exit(0)

# Verificar si ya fue procesado
if DeltaTable.isDeltaTable(spark, "dbfs:/user/hive/warehouse/tu_base_datos.db/agg_mensual_categoria"):
    df_existente = spark.read.table("tu_base_datos.agg_mensual_categoria")
    ya_procesado = df_existente.filter(col("mes") == to_date(lit(f"{mes_proceso}-01"))).count() > 0
    if ya_procesado:
        print(f"El mes {mes_proceso} ya fue procesado.")
        sys.exit(0)

# Agregación mensual
df_agg_mes = (
    df_transacciones
    .groupBy("cliente_id", "categoria")
    .agg(
        countDistinct("producto_id").alias("total_productos"),
        sum("monto").alias("monto_total")
    )
    .withColumn("mes", to_date(lit(f"{mes_proceso}-01")))
)

df_agg_mes.write.format("delta")     .mode("append")     .partitionBy("mes")     .saveAsTable("tu_base_datos.agg_mensual_categoria")

print("Agregados mensuales guardados.")

# Cálculo rolling 6 meses
fecha_ref = f"{mes_proceso}-01"
fecha_inicio_6m = spark.sql(f"SELECT add_months(to_date('{fecha_ref}'), -6)").collect()[0][0]

df_agg_6m = (
    spark.read.table("tu_base_datos.agg_mensual_categoria")
    .filter(col("mes") >= fecha_inicio_6m)
)

df_final = (
    df_agg_6m
    .groupBy("cliente_id", "categoria")
    .agg(
        sum("total_productos").alias("total_productos_6m"),
        sum("monto_total").alias("monto_total_6m")
    )
)

df_final.write.format("delta")     .mode("overwrite")     .saveAsTable("tu_base_datos.agg_rolling_6m")

print("Agregados de los últimos 6 meses guardados.")