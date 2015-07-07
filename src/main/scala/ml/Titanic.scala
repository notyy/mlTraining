package ml

import com.typesafe.scalalogging.slf4j.StrictLogging
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.{Vector, Vectors}


object Titanic extends StrictLogging {
  type TrainingSetPath = String
  type Discrete = Int
  type Linear = Double
  type PassengerID = String

  //PassengerId,Survived,Pclass,Name,Sex,Age(nullable),SibSp,Parch,Ticket,Fare,Cabin(nullable),Embarked(nullable)
  case class Survivor(passengerId: String, survived: Discrete, pClass: Discrete, name: String,
                      sex: String, age: Linear, sibSp: Discrete, parch: Discrete, tickNo: String,
                      fare: Linear, cabin: String, emarkPort: String)

  case class Passanger(passengerId: String, pClass: Discrete, name: String,
                       sex: String, age: Linear, sibSp: Discrete, parch: Discrete, tickNo: String,
                       fare: Linear, cabin: String, emarkPort: String)

  def train2DataFrame: (SparkContext, TrainingSetPath) => DataFrame = (sc, path) => {
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val data = sc.textFile(path)
    data.map(_.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)", -1)).map(s =>
      Survivor(s(0), s(1).toInt, s(2).toInt, s(3), s(4), if (s(5).isEmpty) -1 else s(5).toDouble,
        s(6).toInt, s(7).toInt, s(8), s(9).toDouble, s(10), s(11))).toDF()
//    (?=([^\"]*\"[^\"]*\")*[^\"]*$)
  }

  def test2DataFrame: (SparkContext, TrainingSetPath) => DataFrame = (sc, path) => {
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val data = sc.textFile(path)
    data.map(_.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)", -1)).map(s =>
      Passanger(s(0), s(1).toInt, s(2), s(3), if (s(4).isEmpty) -1 else s(4).toDouble,
        s(5).toInt, s(6).toInt, s(7), if(s(8).isEmpty) -1.0 else s(8).toDouble, s(9), s(10))).toDF()
  }

  def testDF2Vectors: DataFrame => RDD[(PassengerID,Vector)] = df => {
    df.map{
      case Row(id: PassengerID,pClass:Int,name: String,sex: String,age: Double,
      sibSp:Int,parch:Int,tickNo: String,fare:Double,cabin:String,emarkPort:String) =>
        (id,toVector(pClass,sex,age,sibSp,parch))
    }
  }

  def trainDF2LabeledPoints: DataFrame => RDD[LabeledPoint] = df => {
    df.map{
      case Row(id,survived:Int,pClass:Int,name: String,sex: String,age: Double,
      sibSp:Int,parch:Int,tickNo: String,fare:Double,cabin:String,emarkPort:String) =>
        LabeledPoint(survived,toVector(pClass,sex, age,sibSp,parch))
    }
  }

  def toVector: (Int,String,Double,Int,Int) => Vector = (pClass, sex,age,sibSp,parch) =>{
    Vectors.dense(pClass.toDouble - 1.0,if(sex == "female") 0.0 else 1.0,
      if(age < 0 ) 6
      else if(age < 10 && age >= 0) 0
      else if(age < 16 && age >= 10) 1
      else if(age < 24 && age >= 16) 2
      else if(age < 36 && age >= 24) 3
      else if(age < 60 && age >= 36) 4
      else 5
      ,sibSp.toDouble,parch.toDouble)
  }

  def train: RDD[LabeledPoint] => DecisionTreeModel = data => {
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int](0 -> 3, 1 -> 2, 2 -> 7)
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32

    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
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

  def prettify: Map[Int,String] => String => String = dic => debugString => {
    dic.foldLeft(debugString){ case (ds,(fieldIndex, name)) =>
      ds.replaceAll(s"feature $fieldIndex", name)
    }
  }
}
