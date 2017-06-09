package example

import java.nio.file.Paths
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.io.StdIn

// From: https://databricks-training.s3.amazonaws.com/movie-recommendation-with-mllib.html
// RMSE adapted from https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/mllib/MovieLensALS.scala
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

  /*
    ratings.dat:
      UserID::MovieID::Rating::Timestamp

    movies.data
      MovieID::Title::Genres
   */


  def main(args: Array[String]): Unit = {
    execute()
  }

  def execute(): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    val myRatings: Seq[Rating] = loadMyRatings()
    println(s"myRatings: $myRatings")
    val myRatingsRDD = sc.parallelize(myRatings, 1)

    import spark.implicits._
    val ratings: RDD[(Int, Rating)] = loadMovieLensRatings().cache() // RDD[bucket, Ratings]
    ratings.toDF.show(20)

    val movies = loadMovies().collect().toMap // (movieId, movieName)
    println(s"Movies: $movies")

    //    val numRatings = ratings.count
    //    val numUsers = ratings.map(_._2.userId).distinct.count
    //    val numMovies = ratings.map(_._2.movieId).distinct.count
    //    println("Got " + numRatings + " ratings from " + numUsers + " users on " + numMovies + " movies.")
    // 1m:  Got 1000209 ratings from 6040 users on 3706 movies
    // 10m: Got 10000054 ratings from 69878 users on 10677 movies.

    // Split dataset in training/validation/test categories based on timestamp
    val numPartitions = 4
    val training = ratings.filter(x => x._1 < 6)
      .values
      .union(myRatingsRDD)
      .repartition(numPartitions)
      .cache()
    val validation = ratings.filter(x => x._1 >= 6 && x._1 < 8)
      .values
      .repartition(numPartitions)
      .cache()
    val test = ratings.filter(x => x._1 >= 8).values.cache()

    val numTraining = training.count()
    val numValidation = validation.count()
    val numTest = test.count()
    println("Training: " + numTraining + ", validation: " + numValidation + ", test: " + numTest)


    // Train and select best model
    val useImplicitPrefs = true
    val ranks = List(8, 12)
    val lambdas = List(1.0, 10.0)
    val numIters = List(10, 20)
    var bestModel: Option[MatrixFactorizationModel] = None
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1
    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {

      val model: MatrixFactorizationModel = new ALS()
        .setRank(rank)
        .setIterations(numIter)
        .setLambda(lambda)
        .setImplicitPrefs(useImplicitPrefs)
        //.setUserBlocks(params.numUserBlocks)
        //.setProductBlocks(params.numProductBlocks)
        .run(training)

      val validationRmse = computeRmse(model, test, useImplicitPrefs)

      println("RMSE (validation) = " + validationRmse + " for the model trained with rank = " + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
      if (validationRmse < bestValidationRmse) {
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }

    val testRmse = computeRmse(bestModel.get, test, useImplicitPrefs)
    println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
      + ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRmse + ".")

    // Use model to recommend movies
    val myRatedMovieIds = myRatings.map(_.product).toSet
    val candidates = sc.parallelize(movies.keys.filter(!myRatedMovieIds.contains(_)).toSeq)
    val recommendations = bestModel.get
      .predict(candidates.map((0, _)))
      .collect()
      .sortBy(- _.rating)
      .take(50)

    var i = 1
    println("Movies recommended for you:")
    recommendations.foreach { r =>
      println("%2d".format(i) + ": " + movies(r.product))
      i += 1
    }


    println("All done!")
    StdIn.readChar()
    sc.stop()
  }


  def fsPath(name: String): String = Paths.get(getClass.getResource(name).toURI).toString

  def loadMovies(): RDD[(Int, String)] = {
    sc.textFile(fsPath(s"/${DATASET_SIZE}/movies.dat")).map(line => {
      val t = line.split("::")
      (t(0).toInt, t(1))
    })
  }

  def loadMovieLensRatings(): RDD[(Int, Rating)] = {
    sc.textFile(fsPath(s"/${DATASET_SIZE}/ratings.dat")).map(
      line => {
        val t = line.split("::")
        (t(3).toInt % 10, Rating(t(0).toInt, t(1).toInt, t(2).toDouble))
      }
    )
  }

  def loadMyRatings(): Seq[Rating] = {
    val personalRatings = Array(4, 0, 0, 2, 2, 3, 3, 3, 0, 4.5, 3)

    val r = new scala.util.Random
    ratingsTemplate.split("\n").zipWithIndex.map {
      case (line, index) => {
        val t = line.split("::")
        Rating(t(0).toInt, t(1).toInt, personalRatings(index)) // t(3).toInt + r.nextInt(1000)
      }
    }
  }

  /** Compute RMSE (Root Mean Squared Error). */
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], implicitPrefs: Boolean)
  : Double = {

    def mapPredictedRating(r: Double): Double = {
      if (implicitPrefs) math.max(math.min(r, 1.0), 0.0) else r
    }

    val v = data.map(x => ((x.user, x.product), x.rating))
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val w = predictions.map { x => ((x.user, x.product), mapPredictedRating(x.rating)) }
    val predictionsAndRatings = w.join(v).values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean())
  }
}
