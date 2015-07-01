package ml

import com.typesafe.scalalogging.slf4j.StrictLogging
import org.apache.spark.sql.SQLContext
import org.scalatest.{FunSpec, ShouldMatchers}
import sample.SparkContextHelper
import Titanic._

class TitanicSpec extends FunSpec with ShouldMatchers with SparkContextHelper with StrictLogging {
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

    it("can turn data into labeled points"){
      withLocalSparkContext("titanic") { sc =>
        val sqlContext = new SQLContext(sc)
        val df = sqlContext.read.parquet("data_parquet")
        val labeledPoints = dataFrame2LabeledPoints(df)
        labeledPoints.count() shouldBe df.count()
      }
    }

    it("can calculate prediction model from labeled points"){
      withLocalSparkContext("titanic") { sc =>
        val sqlContext = new SQLContext(sc)
        val df = sqlContext.read.parquet("data_parquet")
        val labeledPoints = dataFrame2LabeledPoints(df)
        val model = train(labeledPoints)
        println(model.toDebugString)
      }
    }
  }
}
