package example

import org.apache.spark.sql.SparkSession

object Hello {
  val greeting = "hello"
  val appName = "My App"

  val spark: SparkSession =
    SparkSession
      .builder()
      .appName(appName)
      .config("spark.master", "local") // local[n]-> n threads
      .getOrCreate()

  def main(args: Array[String]): Unit = {
    println(greeting)
  }
}
