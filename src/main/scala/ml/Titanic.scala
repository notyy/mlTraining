package ml

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.sql.functions._

object Titanic {
  type TrainingSetPath = String
  type Discrete = Int
  type Linear = Double

  //PassengerId,Survived,Pclass,Name,Sex,Age(nullable),SibSp,Parch,Ticket,Fare,Cabin(nullable),Embarked(nullable)
  case class Survivor(passengerId: String, survived: Discrete, pClass: Discrete, name: String,
                      sex: String, age: Linear, sibSp: Discrete, parch: Discrete, tickNo: String,
                      fare: Linear, cabin: String, emarkPort: String)

  def csv2DataFrame: (SparkContext, TrainingSetPath) => DataFrame = (sc, path) => {
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val data = sc.textFile(path)
    data.map(_.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)", -1)).map(s =>
      Survivor(s(0), s(1).toInt, s(2).toInt, s(3), s(4), if (s(5).isEmpty) -1 else s(5).toDouble,
        s(6).toInt, s(7).toInt, s(8), s(9).toDouble, s(10), s(11))).toDF()
//    (?=([^\"]*\"[^\"]*\")*[^\"]*$)
  }

  /**
   * this function tests RDD of Array[String] to find columns with empty value
   * @return
   */
  def checkNA: RDD[Array[String]] => Array[Int] = rdd => {
    val count = rdd.first().length
    val indexRange = 0 to count - 1
    rdd.flatMap { d =>
      var haveNA = false
      var naIndexs = Set[Int]()
      indexRange.map { i =>
        if (d(i).isEmpty) {
          haveNA = true
          naIndexs += i
        }
        (haveNA, naIndexs)
      }
    }.filter{case (k,vs) => k}.values.reduce(_ ++ _).toArray
//    (0 to count - 1).map(i => rdd.map(_(i)).filter(_.isEmpty).count())
  }
}
