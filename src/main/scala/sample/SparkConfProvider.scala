package sample

import org.apache.spark.SparkConf

trait SparkConfProvider {
  def appName: String

  def sparkConf: SparkConf
}

trait SparkAppConfProvider extends SparkConfProvider {
  override def sparkConf: SparkConf = new SparkConf().setAppName(appName)
}

trait LocalSparkConfProvider extends SparkConfProvider {
  override def sparkConf: SparkConf = new SparkConf().setAppName(appName).setMaster("local")
}