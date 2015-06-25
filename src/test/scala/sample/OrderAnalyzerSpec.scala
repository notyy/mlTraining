package sample

import java.io.File

import com.typesafe.scalalogging.slf4j.LazyLogging
import org.scalatest.{BeforeAndAfterAll, FunSpec, ShouldMatchers}

class OrderAnalyzerSpec extends FunSpec with ShouldMatchers with SparkContextHelper with BeforeAndAfterAll with LazyLogging {

  val outputPath: String = "temp/output"

  override def beforeAll() = {
    val outputDir = new File(outputPath)
    if (outputDir.exists() && outputDir.isDirectory) {
      outputDir.listFiles().foreach(_.delete())
      val rs = outputDir.delete()
      logger.info(s"existing output dir delete with status: $rs")
    }
  }

  describe("OrderAnalyzer") {
    it("can analyze order count and output to text file") {
      withLocalSparkContext("OrderAnalyzerSpec") { sc =>
        new OrderAnalyzer {}.analyzeOrderCount(sc, "src/test/resources/order.csv", outputPath)
      }
    }
    it("can analyze order count") {
      withLocalSparkContext("OrderAnalyzerSpec") { sc =>
        val src =
          """
            |2014-01-06,xx,99.38
            |2014-01-06,xx,100.0
            |2014-01-06,yy,77.0
            |2014-02-08,xx,59.0
            |2014-02-08,yy,37.0
          """.stripMargin.trim

        val srcRDD = sc.parallelize(src.lines.toSeq)
        val rs = new OrderAnalyzer {}.doAnalyzeOrderCount(srcRDD).collect()
        rs.length shouldBe 2
        rs(0) shouldBe "2014-01-06,3,2,1"
        rs(1) shouldBe "2014-02-08,2,1,1"
      }
    }
  }
}
