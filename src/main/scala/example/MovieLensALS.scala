package example

import java.nio.file.Paths

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

// From Spark 2.3.0-snapshot: https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/mllib/MovieLensALS.scala
// dataset 10m entries: http://grouplens.org/datasets/movielens/10m/
// dataset 1m entries: http://grouplens.org/datasets/movielens/1m/
// Unpack datasets in resources.
// Imports: org.apache.spark.ml (newer, DataFrames), org.apache.spark.mllib (older, RDDs)
object MovieLensALS {
  val appName = "MovieLensALS"
  val DATASET_SIZE = "1m"

  val spark: SparkSession =
    SparkSession
      .builder()
      .appName(appName)
      .config("spark.master", "local")
      .config("spark.executor.memory", "2g")
      .getOrCreate()
  val sc = spark.sparkContext

  // UserID::MovieID::Rating::Timestamp - Rating scale: 1-5 (best), with half star scale, or 0 if not seen
  val ratingsTemplate =
    """0::1::?::1400000000::Toy Story (1995)
      |0::780::?::1400000000::Independence Day (a.k.a. ID4) (1996)
      |0::590::?::1400000000::Dances with Wolves (1990)
      |0::1210::?::1400000000::Star Wars: Episode VI - Return of the Jedi (1983)
      |0::648::?::1400000000::Mission: Impossible (1996)
      |0::344::?::1400000000::Ace Ventura: Pet Detective (1994)
      |0::165::?::1400000000::Die Hard: With a Vengeance (1995)
      |0::153::?::1400000000::Batman Forever (1995)
      |0::597::?::1400000000::Pretty Woman (1990)
      |0::1580::?::1400000000::Men in Black (1997)
      |0::231::?::1400000000::Dumb & Dumber (1994)""".stripMargin.replace("\r\n", "\n")

  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)

  def main(args: Array[String]): Unit = {
    execute()
  }

  def execute(): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    import spark.implicits._
    val myRatingsRDD = sc.parallelize(loadMyRatings(), 1)
    val ratings = loadMovieLensRatings().union(myRatingsRDD).toDF.cache
    val movies = loadMovies().collect().toMap // (movieId, movieName)
    ratings.toDF.show(20)
    println(s"Movies: ${movies.take(20)}")

    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
    val model = als.fit(training)

    // Evaluate the model by computing the RMSE on the test data
    // Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    model.setColdStartStrategy("drop")
    val predictions = model.transform(test)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")

    // Generate top 10 movie recommendations for each user
    val userRecs = model.recommendForAllUsers(10)
    // Generate top 10 user recommendations for each movie
    val movieRecs = model.recommendForAllItems(10)

    userRecs.show()
    movieRecs.show()

    println("All done!")
    //StdIn.readChar()
    sc.stop()
  }


  def fsPath(name: String): String = Paths.get(getClass.getResource(name).toURI).toString

  def loadMovies(): RDD[(Int, String)] = {
    sc.textFile(fsPath(s"/${DATASET_SIZE}/movies.dat")).map(line => {
      val t = line.split("::")
      (t(0).toInt, t(1))
    })
  }

  def loadMovieLensRatings(): RDD[Rating] = {
    sc.textFile(fsPath(s"/${DATASET_SIZE}/ratings.dat")).map(
      line => {
        val fields = line.split("::")
        Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
      }
    )
  }

  def loadMyRatings(): Seq[Rating] = {
    val personalRatings = Array(4, 0, 0, 2, 2, 3, 3, 3, 0, 4.5, 3)
    ratingsTemplate.split("\n").zipWithIndex.map {
      case (line, index) => {
        val fields = line.split("::")
        Rating(fields(0).toInt, fields(1).toInt, personalRatings(index).toFloat, fields(3).toLong)
      }
    }
  }
}
