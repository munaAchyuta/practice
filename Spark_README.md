#Problemset-2

# Environments & Requirements
OS : Ubuntu 16.04.6 LTS (i have used Ubuntu. but it's not limited, you can try Windows.)
install python 3.6
install pyspark(you can follow this link. <https://www.roseindia.net/bigdata/pyspark/install-pyspark-on-ubuntu.shtml> or <https://www.knowledgehut.com/blog/big-data/install-spark-on-ubuntu>)
for pyspark : i have used pre-executable spark. and the cersion i used is `spark-2.4.4-bin-hadoop2.7`

# follow below structure for configuring spark and script.
1. let's say you are in home directory. which will be `/home`.
2. create a directory in home directory called `spark`. absolute path will be like `/home/spark`. you can then get pre-executable spark binary file inside spark directory. and if you follow pyspark installation link <https://www.roseindia.net/bigdata/pyspark/install-pyspark-on-ubuntu.shtml>, then your path to spark executable will be like `/home/spark/spark-2.4.4-bin-hadoop2.7/bin/spark` , `/home/spark/spark-2.4.4-bin-hadoop2.7/bin/spark-submit`, `/home/spark/spark-2.4.4-bin-hadoop2.7/bin/pyspark`.
3. create a task directory in home directory by unzipping task.zip. which will be like `/home/task`

# Execution Steps.
go to script directory. get spark-submit(which resides inside spark/bin directory.) path and execute below script.

       ## go inside task directory & execute script following below mentioned way:
        `path-to-spark-submit pyspark-script.py`
       Example::
	`/home/spark/spark-2.4.4-bin-hadoop2.7/bin/spark-submit pyspark-script.py`

# result
user can see result on output log. otherwise user can see result on `result.txt` file inside task directory.
json to csv file (`csvfile.csv`) will be also created inside task directory.
