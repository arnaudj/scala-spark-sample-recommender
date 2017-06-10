import Dependencies._
import sbt.Keys.libraryDependencies

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "com.example",
      scalaVersion := "2.11.8",
      version      := "0.1.0-SNAPSHOT"
    )),
    name := "Recommender",
    resolvers += "ASF snapshots" at "https://repository.apache.org/snapshots/",
    libraryDependencies += scalaTest % Test,
    libraryDependencies += "org.apache.spark" %% "spark-core" %  "2.3.0-SNAPSHOT",
    libraryDependencies += "org.apache.spark" %% "spark-sql" %   "2.3.0-SNAPSHOT",
    libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.3.0-SNAPSHOT"
  )
