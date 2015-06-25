package sample

import com.typesafe.scalalogging.slf4j.StrictLogging
import org.scalatest.{BeforeAndAfterAll, ShouldMatchers, FunSpec}

class ReduceCalcSpec extends FunSpec with ReduceCalc with ShouldMatchers with SparkContextHelper with BeforeAndAfterAll with StrictLogging {

  val src =
    """
      |t001,c001,5
      |t001,c002,10
      |t002,c001,10
    """.stripMargin.trim

  val exp =
    """
      |t001,5,0.5
      |t001,10,0.5
      |t002,10,1.0
      |c001,5,0.5
      |c002,10,1.0
      |c001,10,0.5
    """.stripMargin.trim

  describe("ReduceCalc") {
    it("can rate data by using countByKey and reduceByKey multiple times") {
      withLocalSparkContext("ReduceCalcSec") { sc =>
        val data = sc.parallelize(src.lines.toSeq)
        rateByKeyByAndReduceBy(data).collect().toSet should be (exp.lines.toSet)
      }
    }

    it("can rate data by using groupByKey without countByKey") {
      withLocalSparkContext("ReduceCalcSec") { sc =>
        val data = sc.parallelize(src.lines.toSeq)
        rateByGroupByAndMap(data).collect().toSet should be (exp.lines.toSet)
      }
    }

    it("can rate data by using reduce only one") {
      withLocalSparkContext("ReduceCalcSec") { sc =>
        val data = sc.parallelize(src.lines.toSeq)
        rateByReduce(sc,data) should be (exp.lines.toSet)
      }
    }

  }
}
