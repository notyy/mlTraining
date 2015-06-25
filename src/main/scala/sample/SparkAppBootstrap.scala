package sample

import org.apache.spark.SparkContext

trait SparkAppBootstrap extends SparkContextHelper {
  this: SparkConfProvider =>

  def doMain(sc: SparkContext, args: Array[String]): Unit
  
  def main(args: Array[String]) {
    withSparkContext(sparkConf) { sc =>
      doMain(sc,args)
    }
  }
}
