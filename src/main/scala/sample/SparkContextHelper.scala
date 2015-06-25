package sample

import com.typesafe.scalalogging.slf4j.LazyLogging
import org.apache.spark.{SparkConf, SparkContext}

trait SparkContextHelper extends Serializable {

  def withSparkContext[T](sparkConf: SparkConf)(f: SparkContext => T): T = {
    println("starting spark context")
    val sc = new SparkContext(sparkConf)
    println("spark context started")
    try {
      println("calling function on spark context")
      val rs = f(sc)
      println("function completed")
      rs
    } finally {
      println("stopping spark context")
      sc.stop()
      println("spark context stopped")
    }
  }

  def withLocalSparkContext[T](appName: String) = withSparkContext[T](
    new SparkConf().setAppName(appName).setMaster("local").set("spark.ui.port", "4041")) _
}
