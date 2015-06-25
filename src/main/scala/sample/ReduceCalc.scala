package sample

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

trait ReduceCalc extends Serializable {
  def rateByKeyByAndReduceBy(src: RDD[String]): RDD[String] = {
    //t001,c001,5
    val data = src.map(_.split(",")).cache()
    val tRate = rateItem(data, 0, 2)
    val cRate = rateItem(data, 1, 2)
    tRate union cRate
  }

  def rateItem(data: RDD[Array[String]], idPos: Int, tValuePos: Int): RDD[String] = {
    val tCount = data.keyBy(_(idPos)).countByKey()
    val tRate = data.map(x => {((x(idPos), x(tValuePos)), 1.0)}).reduceByKey(_ + _).
      map { case ((k, v), c) => s"$k,$v,${c / tCount(k)}"}
    tRate
  }

  def rateByGroupByAndMap(src: RDD[String]): RDD[String] = {
    //t001,c001,5
    val data = src.map(_.split(",")).cache()
    val tRate = data.groupBy(_(0)).flatMap{
      case (k, a) => {
        val keyCount = a.size
        a.groupBy(x => (x(0), x(2))).mapValues(_.size.toDouble).map{
          case ((k1,v),c) => s"$k1,$v,${c / keyCount}"
        }
      }
    }
    val cRate = data.groupBy(_(1)).flatMap{
      case (k, a) => {
        val keyCount = a.size
        a.groupBy(x => (x(1), x(2))).mapValues(_.size.toDouble).map{
          case ((k1,v),c) => s"$k1,$v,${c / keyCount}"
        }
      }
    }
    tRate union cRate
  }

  def rateByReduce(sc: SparkContext,src: RDD[String]): Set[String] = {
    val data = src.map(_.split(",")).cache()
    val tCount = data.keyBy(_(0)).countByKey()
    val cCount = data.keyBy(_(1)).countByKey()
    val keyCounts = sc.broadcast(tCount ++ cCount)

    data.map(x => Map((x(0),x(2)) -> 1.0, (x(1),x(2)) -> 1.0)).reduce { (m1,m2) =>
      (m1.keySet ++ m2.keySet).map(k => {
        val keyCount: Long = keyCounts.value(k._1)
        k -> (m1.getOrElse(k, 0.0) + m2.getOrElse(k, 0.0))
      }).toMap
    }.map{case ((k,v),c) => s"$k,$v,${c / keyCounts.value(k)}"}.toSet
  }

}
