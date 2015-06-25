package sample

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.scalatest._

class OlympicCalcSpec extends FunSpec with ShouldMatchers with SparkContextHelper with OlympicCalc {

  describe("OlymicCalc"){

    val olympicFilePath = "src/test/resources/olympic.csv"
    it("can find which country scored most medals"){
      withLocalSparkContext("OlymicCalc"){ sc =>
        val olympicsRDD = sc.textFile(olympicFilePath)

        findCountryHavingMostMedals(olympicsRDD) shouldBe ("United States",1312)
        //can also use DataFrame to archive this
        val sqlContext = new SQLContext(sc)
        import sqlContext.implicits._

        //if you don't have olmypics data in parquet format, you should transform
        //from rdd to dataframe
//        val olympicsDF = olympicsRDD.map(_.split(",")).
//          map(o => Olympic(o(0),o(1),o(2),o(3),o(4).toInt,o(5).toInt,o(6).toInt,o(7).toInt)).
//          toDF()

        //if you already have parquet file, just load it with sqlContext.read.parquet
        val olympicsDF = sqlContext.read.parquet("src/test/resources/olympicParqet")
        findCountryHavingMostMedalsDF(olympicsDF, sqlContext) shouldBe ("United States",1312)
      }
    }

    it("can find the year in which USA scored most medals"){
      withLocalSparkContext("OlymicCalc"){ sc =>
        val olympicsRDD = sc.textFile(olympicFilePath)
        val sqlContext = new SQLContext(sc)
        val olympicsDF = sqlContext.read.parquet("src/test/resources/olympicParqet")
        findMostMedalsYear4USA(olympicsDF,sqlContext) shouldBe "2008"
      }
    }

    it("can List the years when  “United States” scored less than 200 medals"){
      withLocalSparkContext("OlymicCalc"){ sc =>
        val olympicsRDD = sc.textFile(olympicFilePath)
        findYearsUSAMedalLt200(olympicsRDD).toSet shouldBe Array(("2010",97),("2002",84),("2006",52)).toSet
      }
    }

    it("can find Which player has medals in more than one sport"){
      withLocalSparkContext("OlymicCalc") { sc =>
        val olympicsRDD = sc.textFile(olympicFilePath)
        findPlayerHasMedalInSportsGT1(olympicsRDD) should have size 19
      }
    }
  }
}
