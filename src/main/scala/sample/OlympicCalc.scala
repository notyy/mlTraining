package sample

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, DataFrame}

import scala.reflect.internal.util.TableDef.Column
import org.apache.spark.sql.functions._

case class Olympic(player: String, country: String,year: String,
                   sport: String,bronze: Int, siver: Int, gold: Int, total: Int)

trait OlympicCalc {
  type CountryName = String
  type MedalCount = Int
  type Year = String
  type Player = String
  type Sport = String

  def findMostMedalsYear4USA: (DataFrame,SQLContext) => String = (df,sc) => {
//    toYearMedalPair(rdd, "United States").
//      sortBy(_._2, ascending = false).map(_._1).first()
    import sc.implicits._
    toYearMedalPairDF(df,"United States",sc).orderBy($"total".desc).first().getAs[String]("year")
  }

  def toYearMedalPair(rdd: RDD[String], country: CountryName): RDD[(Year, MedalCount)] = {
    rdd.map(_.split(",")).filter(_(1) == country).
      map(a => (a(2), a(7).toInt)).reduceByKey(_ + _)
  }

  def toYearMedalPairDF(dataFrame: DataFrame, country: CountryName, sqlContext: SQLContext): DataFrame = {
    import sqlContext.implicits._

    dataFrame.filter($"country" === country).
      select("year","total").groupBy("year").agg(sum("total") as "total")
  }

  def findCountryHavingMostMedals: RDD[String] => (CountryName, MedalCount) = rdd => {
    rdd.map(_.split(",")).map(a => (a(1),a(7).toInt)).reduceByKey(_ + _).
      sortBy(_._2, ascending = false).first()
  }

  /**
   * a sample that use dataframe to do the same thing as rdd
   */
  def findCountryHavingMostMedalsDF: (DataFrame,SQLContext) => (CountryName, MedalCount) = { (df,sc) =>
    import org.apache.spark.sql.functions._
    import sc.implicits._

    val rs = df.select("country","total").groupBy("country").agg(sum("total").as("total")).
      orderBy($"total".desc).first()
    (rs.getAs[String]("country"), rs.getAs[Int]("total"))
  }

  def findYearsUSAMedalLt200: RDD[String] => Array[(Year,MedalCount)] = { rdd =>
    toYearMedalPair(rdd,"United States").filter{ case (y,m) => m < 200}.collect()
  }

  def findPlayerHasMedalInSportsGT1: RDD[String] => Array[Player] = { rdd =>
    rdd.map(_.split(",")).map(a => (a(0),a(3))).
      aggregateByKey(Set[Sport]())({(ss,s) => ss + s },{ (ss1,ss2) => ss1 ++ ss2 }).
      filter(_._2.size > 1).map(_._1).collect()
  }
}
