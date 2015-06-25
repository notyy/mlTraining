package sample

import org.apache.spark._
import org.scalatest.{BeforeAndAfter, FunSpec}

import scala.math.random

class MySparkPiSpec extends FunSpec with BeforeAndAfter with SparkContextHelper {

  describe("MySparkPi") {
    it("can be run locally") {
      val conf = new SparkConf().setAppName("Spark Pi").setMaster("local")
      withSparkContext(conf) { sc =>
        val slices = 2
        val n = 1000 * slices
        val count = sc.parallelize(1 to n, slices).map { i =>
          val x = random * 2 - 1
          val y = random * 2 - 1
          if (x * x + y * y < 1) 1 else 0
        }.reduce(_ + _)
        val pi = 4.0 * count / n
        println("Pi is roughly " + pi)
      }
    }
  }

}
