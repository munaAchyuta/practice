import datetime
import time
from datetime import timedelta
import re
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, udf, lit, unix_timestamp
from pyspark.sql import functions as F
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import *

# global variables
json_file_path = "ecom.json"
timeFmt = "yyyy-MM-dd'T'HH:mm:ss.SSS"
submitted_timestamp = "2018-06-22 14:55:13"
submitted_timestamp = datetime.datetime.strptime(submitted_timestamp,"%Y-%m-%d %H:%M:%S")
target_timestamp = submitted_timestamp-timedelta(days=90)

# schema config
# we can make it dynamic. for now let be static.

# get json to dataframe
spark = SparkSession.builder.appName("json process").config("spark.some.config.option", "some-value").getOrCreate()
source_df = spark.read.option("multiline", "true").json(json_file_path)

# schema variables defination.
# here after looking at json data found that there is one schema variable which make transactions differ from each other. which is ecom_{service_type}. Others are same for each transaction.
# so we can split data by ecom_{service_type} variable as this one is major differentiator from schema variables.
users = source_df.select(explode("users").alias("users"))
users_data = users.select(col("users.user_id").alias("user_id"), col("users.timestamp").alias("timestamp"), col("users.data.ecom1").alias("data_ecom1"), col("users.data.ecom2").alias("data_ecom2"), col("users.data.ecom3").alias("data_ecom3"), col("users.data.ecom4").alias("data_ecom4"), col("users.data.ecom5").alias("data_ecom5"))

users_data_trnx1 = users_data.select(col("user_id"), col("timestamp"), col("data_ecom1.id").alias("data_id"), col("data_ecom1.user_id").alias("data_userid"), col("data_ecom1.provider").alias("provider"), col("data_ecom1.creation_date").alias("creation_date"), explode("data_ecom1.transactions").alias("data_ecom_transactions"),col("data_ecom_transactions.transaction_id").alias("transaction_id"),col("data_ecom_transactions.transaction_date").alias("transaction_date"),col("data_ecom_transactions.transaction_amount").alias("transaction_amount"),col("data_ecom_transactions.std_status").alias("std_status"),col("data_ecom_transactions.creation_time").alias("creation_time"),col("data_ecom_transactions.modification_time").alias("modification_time"))

users_data_trnx2 = users_data.select(col("user_id"), col("timestamp"), col("data_ecom2.id").alias("data_id"), col("data_ecom2.user_id").alias("data_userid"), col("data_ecom2.provider").alias("provider"), col("data_ecom2.creation_date").alias("creation_date"), explode("data_ecom2.transactions").alias("data_ecom_transactions"),col("data_ecom_transactions.transaction_id").alias("transaction_id"),col("data_ecom_transactions.transaction_date").alias("transaction_date"),col("data_ecom_transactions.transaction_amount").alias("transaction_amount"),col("data_ecom_transactions.std_status").alias("std_status"),col("data_ecom_transactions.creation_time").alias("creation_time"),col("data_ecom_transactions.modification_time").alias("modification_time"))

users_data_trnx3 = users_data.select(col("user_id"), col("timestamp"), col("data_ecom3.id").alias("data_id"), col("data_ecom3.user_id").alias("data_userid"), col("data_ecom3.provider").alias("provider"), col("data_ecom3.creation_date").alias("creation_date"), explode("data_ecom3.transactions").alias("data_ecom_transactions"),col("data_ecom_transactions.transaction_id").alias("transaction_id"),col("data_ecom_transactions.transaction_date").alias("transaction_date"),col("data_ecom_transactions.transaction_amount").alias("transaction_amount"),col("data_ecom_transactions.std_status").alias("std_status"),col("data_ecom_transactions.creation_time").alias("creation_time"),col("data_ecom_transactions.modification_time").alias("modification_time"))

users_data_trnx4 = users_data.select(col("user_id"), col("timestamp"), col("data_ecom4.id").alias("data_id"), col("data_ecom4.user_id").alias("data_userid"), col("data_ecom4.provider").alias("provider"), col("data_ecom4.creation_date").alias("creation_date"), explode("data_ecom4.transactions").alias("data_ecom_transactions"),col("data_ecom_transactions.transaction_id").alias("transaction_id"),col("data_ecom_transactions.transaction_date").alias("transaction_date"),col("data_ecom_transactions.transaction_amount").alias("transaction_amount"),col("data_ecom_transactions.std_status").alias("std_status"),col("data_ecom_transactions.creation_time").alias("creation_time"),col("data_ecom_transactions.modification_time").alias("modification_time"))

users_data_trnx5 = users_data.select(col("user_id"), col("timestamp"), col("data_ecom5.id").alias("data_id"), col("data_ecom5.user_id").alias("data_userid"), col("data_ecom5.provider").alias("provider"), col("data_ecom5.creation_date").alias("creation_date"), explode("data_ecom5.transactions").alias("data_ecom_transactions"),col("data_ecom_transactions.transaction_id").alias("transaction_id"),col("data_ecom_transactions.transaction_date").alias("transaction_date"),col("data_ecom_transactions.transaction_amount").alias("transaction_amount"),col("data_ecom_transactions.std_status").alias("std_status"),col("data_ecom_transactions.creation_time").alias("creation_time"),col("data_ecom_transactions.modification_time").alias("modification_time"))

# merge all separate ecom_{service_type} data(dataframes) to one as single dataframe.
df_final12 = users_data_trnx1.union(users_data_trnx2)
df_final123 = df_final12.union(users_data_trnx3)
df_final1234 = df_final123.union(users_data_trnx4)
df_final12345 = df_final1234.union(users_data_trnx5)
df_final = df_final12345.drop("data_ecom_transactions")

#Persist dataframe for faster operations.
df_final.cache()

# Operations
# filter records having success transaction and have done transaction in last 90 days from submitted datetime.
df_filter = df_final.filter(~df_final.std_status.isin(["CANCELLED","FAILED"]))
df_filter = df_filter.filter(col("transaction_date").between(target_timestamp, submitted_timestamp))

# clean transaction amount column.
udf = UserDefinedFunction(lambda x: re.sub('(?:^.| )\w','',x.replace(".","").replace(",","")), StringType())
df_filter1 = df_filter.withColumn("transaction_amount_clean", udf(col("transaction_amount")).cast("int"))

# get count and amount_sum
result_cnt = df_filter1.count()
result_sum = df_filter1.select(F.sum('transaction_amount_clean')).collect()[0][0]

# get length between first transaction and submitted date
first_transaction_date = df_final.agg({"transaction_date": "min"}).collect()[0][0]
duration = submitted_timestamp - datetime.datetime.strptime(first_transaction_date,"%Y-%m-%d %H:%M:%S")

print("=========================")
print("Report : considering all users transactions :")
print("count success transaction in last 90 days before submitted timestamp : ",result_cnt)
#print("total transaction amount in last 90 days before submitted timestamp : ",result_sum)
print("average transaction amount in last 90 days before submitted timestamp : ",result_sum/result_cnt)
print("length between first transaction to submitted timestamp : ",duration.days)
print("=========================")

# write to txt
with open("result.txt",'w') as f:
     f.write("count success transaction in last 90 days before submitted timestamp : {}".format(result_cnt))
     f.write("\n")
     f.write("average transaction amount in last 90 days before submitted timestamp : {}".format(result_sum/result_cnt))
     f.write("\n")
     f.write("length between first transaction to submitted timestamp : {}".format(duration.days))

# to csv.
df_final.toPandas().to_csv('csvfile.csv', index=False, encoding='utf-8')
