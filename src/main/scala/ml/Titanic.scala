package ml

import com.typesafe.scalalogging.slf4j.StrictLogging
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext, DataFrame}
import org.apache.spark.sql.functions._

object Titanic extends StrictLogging {
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
    data.countByValue()
    data.map(_.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)", -1)).map(s =>
      Survivor(s(0), s(1).toInt, s(2).toInt, s(3), s(4), if (s(5).isEmpty) -1 else s(5).toDouble,
        s(6).toInt, s(7).toInt, s(8), s(9).toDouble, s(10), s(11))).toDF()
//    (?=([^\"]*\"[^\"]*\")*[^\"]*$)
  }

  import org.apache.spark.mllib.linalg.{Vector, Vectors}
  def dataFrame2LabeledPoints: DataFrame => RDD[LabeledPoint] = df => {
    df.map{
      case Row(id,survived:Int,pClass:Int,name: String,sex: String,age: Double,
      sibSp:Int,parch:Int,tickNo: String,fare:Double,cabin:String,emarkPort:String) =>
        LabeledPoint(survived,
          Vectors.dense(pClass.toDouble,if(sex == "female") 0.0 else 1.0, age,sibSp.toDouble,parch.toDouble,fare))
    }
  }

  def train: RDD[LabeledPoint] => DecisionTreeModel = data => {
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 1
    val maxBins = 32

    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)
    model
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
