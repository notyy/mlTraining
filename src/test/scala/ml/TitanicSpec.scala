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

    it("can check NA values of test.csv"){
      withLocalSparkContext("titanic") { sc =>
        val data = sc.textFile("data/test.csv")
        println(checkNA(data.map(_.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)", -1))).toSet)
      }
    }

    it("can turn data into labeled points"){
      withLocalSparkContext("titanic") { sc =>
        val sqlContext = new SQLContext(sc)
        val df = sqlContext.read.parquet("data_parquet")
        val labeledPoints = trainDF2LabeledPoints(df)
        labeledPoints.count() shouldBe df.count()
      }
    }

    it("can calculate prediction model from labeled points"){
      withLocalSparkContext("titanic") { sc =>
        val sqlContext = new SQLContext(sc)
        val df = sqlContext.read.parquet("data_parquet")
        val labeledPoints = trainDF2LabeledPoints(df)
        val model = train(labeledPoints)
        println(model.toDebugString)
      }
    }

    ignore("can predict test data by the calculated model"){
      withLocalSparkContext("titanic") { sc =>
        val sqlContext = new SQLContext(sc)
        val df = sqlContext.read.parquet("data_parquet")
        val labeledPoints = trainDF2LabeledPoints(df)
        val model = train(labeledPoints)
        println(model.toDebugString)
        val testDF = test2DataFrame(sc, "data/test.csv")
        val testVectors = testDF2Vectors(testDF)
        testVectors.map{ case (pid,features) =>
          s"$pid,${model.predict(features).toInt}"
        }.coalesce(1).saveAsTextFile("out")
      }
    }
  }
}
