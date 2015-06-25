package sample

import org.apache.spark.SparkConf
import org.scalatest.FunSpec

class SubAggregateExample extends FunSpec with SparkContextHelper {
  describe("SubAggregate example") {
    it("show how to count array datas encoded in string") {
      val conf = new SparkConf().setAppName("Spark Pi").setMaster("local")
      withSparkContext(conf) { sc =>
        val l = List("1001,E$I", "1001,E$X$T", "1002,I$T")
        val rdd = sc.parallelize(l)
        val kvs = rdd.map(_.split(",")).flatMap(x => x(1).split("\\$").map(y => ((x(0),y),1)))
        kvs.reduceByKey(_ + _).map{case ((id,t),freq) => s"$id,$t,$freq"}.foreach(println)
      }
    }
  }
}
