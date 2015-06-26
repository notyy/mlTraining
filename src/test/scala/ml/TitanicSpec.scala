package ml

import org.scalatest.{FunSpec, ShouldMatchers}
import sample.SparkContextHelper
import Titanic._

class TitanicSpec extends FunSpec with ShouldMatchers with SparkContextHelper {
  describe("Titanic module") {
    it("contains function that can detect na values from rdd") {
      withLocalSparkContext("titanic") { sc =>
        val rdd = sc.parallelize(Array("a,a,,a", "b,,a,a"))
        checkNA(rdd.map(_.split(",", -1))).toSet shouldBe Set(2, 1)
      }
    }
    it("contains functions to turn source csv to data frame"){
      withLocalSparkContext("titanic") { sc =>
        val rdd = sc.textFile("data/train.csv")
//        checkNA(rdd.map(_.split(",", -1))).toSet shouldBe Set(2, 1)
      }
    }
  }
}
