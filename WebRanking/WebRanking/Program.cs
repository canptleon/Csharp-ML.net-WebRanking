using Microsoft.ML;
using WebRanking.Common;
using WebRanking.DataStructures;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;

namespace WebRanking
{
    class Program
    {
        const string AssetsPath = @"../../../Assets";
        const string TrainDatasetUrl = "https://aka.ms/mlnet-resources/benchmarks/MSLRWeb10KTrain720kRows.tsv";
        const string ValidationDatasetUrl = "https://aka.ms/mlnet-resources/benchmarks/MSLRWeb10KValidate240kRows.tsv";
        const string TestDatasetUrl = "https://aka.ms/mlnet-resources/benchmarks/MSLRWeb10KTest240kRows.tsv";

        readonly static string InputPath = Path.Combine(AssetsPath, "Input");
        readonly static string OutputPath = Path.Combine(AssetsPath, "Output");
        readonly static string TrainDatasetPath = Path.Combine(InputPath, "MSLRWeb10KTrain720kRows.tsv");
        readonly static string ValidationDatasetPath = Path.Combine(InputPath, "MSLRWeb10KValidate240kRows.tsv");
        readonly static string TestDatasetPath = Path.Combine(InputPath, "MSLRWeb10KTest240kRows.tsv");
        readonly static string ModelPath = Path.Combine(OutputPath, "RankingModel.zip");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            try
            {
                PrepareData(InputPath, OutputPath, TrainDatasetPath, TrainDatasetUrl, TestDatasetUrl, TestDatasetPath, ValidationDatasetUrl, ValidationDatasetPath);

                IDataView trainData = mlContext.Data.LoadFromTextFile<SearchResultData>(TrainDatasetPath, separatorChar: '\t', hasHeader: true);
                IEstimator<ITransformer> pipeline = CreatePipeline(mlContext, trainData);

                Console.WriteLine("===== Train the model on the training dataset =====\n");
                ITransformer model = pipeline.Fit(trainData);

                Console.WriteLine("===== Evaluate the model's result quality with the validation data =====\n");
                IDataView validationData = mlContext.Data.LoadFromTextFile<SearchResultData>(ValidationDatasetPath, separatorChar: '\t', hasHeader: false);
                EvaluateModel(mlContext, model, validationData);

                var validationDataEnum = mlContext.Data.CreateEnumerable<SearchResultData>(validationData, false);
                var trainDataEnum = mlContext.Data.CreateEnumerable<SearchResultData>(trainData, false);
                var trainValidationDataEnum = validationDataEnum.Concat<SearchResultData>(trainDataEnum);
                IDataView trainValidationData = mlContext.Data.LoadFromEnumerable<SearchResultData>(trainValidationDataEnum);

                Console.WriteLine("===== Train the model on the training + validation dataset =====\n");
                model = pipeline.Fit(trainValidationData);

                Console.WriteLine("===== Evaluate the model's result quality with the testing data =====\n");
                IDataView testData = mlContext.Data.LoadFromTextFile<SearchResultData>(TestDatasetPath, separatorChar: '\t', hasHeader: false);
                EvaluateModel(mlContext, model, testData);

                var testDataEnum = mlContext.Data.CreateEnumerable<SearchResultData>(testData, false);
                var allDataEnum = trainValidationDataEnum.Concat<SearchResultData>(testDataEnum);
                IDataView allData = mlContext.Data.LoadFromEnumerable<SearchResultData>(allDataEnum);

                Console.WriteLine("===== Train the model on the training + validation + test dataset =====\n");
                model = pipeline.Fit(allData);

                ConsumeModel(mlContext, model, ModelPath, testData);
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }

            Console.Write("Done!");
            Console.ReadLine();
        }

        static void PrepareData(string inputPath, string outputPath, string trainDatasetPath, string trainDatasetUrl,
            string testDatasetUrl, string testDatasetPath, string validationDatasetUrl, string validationDatasetPath)
        {
            Console.WriteLine("===== Prepare data =====\n");

            if (!Directory.Exists(outputPath))
            {
                Directory.CreateDirectory(outputPath);
            }

            if (!Directory.Exists(inputPath))
            {
                Directory.CreateDirectory(inputPath);
            }

            if (!File.Exists(trainDatasetPath))
            {
                Console.WriteLine("===== Download the train dataset - this may take several minutes =====\n");
                using (var client = new WebClient())
                {
                    client.DownloadFile(trainDatasetUrl, TrainDatasetPath);
                }
            }

            if (!File.Exists(validationDatasetPath))
            {
                Console.WriteLine("===== Download the validation dataset - this may take several minutes =====\n");
                using (var client = new WebClient())
                {
                    client.DownloadFile(validationDatasetUrl, validationDatasetPath);
                }
            }

            if (!File.Exists(testDatasetPath))
            {
                Console.WriteLine("===== Download the test dataset - this may take several minutes =====\n");
                using (var client = new WebClient())
                {
                    client.DownloadFile(testDatasetUrl, testDatasetPath);
                }
            }

            Console.WriteLine("===== Download is finished =====\n");
        }

        static IEstimator<ITransformer> CreatePipeline(MLContext mlContext, IDataView dataView)
        {
            const string FeaturesVectorName = "Features";

            Console.WriteLine("===== Set up the trainer =====\n");

            var featureCols = dataView.Schema.AsQueryable()
                .Select(s => s.Name)
                .Where(c =>
                    c != nameof(SearchResultData.Label) &&
                    c != nameof(SearchResultData.GroupId))
                 .ToArray();

            IEstimator<ITransformer> dataPipeline = mlContext.Transforms.Concatenate(FeaturesVectorName, featureCols)
                .Append(mlContext.Transforms.Conversion.MapValueToKey(nameof(SearchResultData.Label)))
                .Append(mlContext.Transforms.Conversion.Hash(nameof(SearchResultData.GroupId), nameof(SearchResultData.GroupId), numberOfBits: 20));

            IEstimator<ITransformer> trainer = mlContext.Ranking.Trainers.LightGbm(labelColumnName: nameof(SearchResultData.Label), featureColumnName: FeaturesVectorName, rowGroupColumnName: nameof(SearchResultData.GroupId));
            IEstimator<ITransformer> trainerPipeline = dataPipeline.Append(trainer);

            return trainerPipeline;
        }

        static void EvaluateModel(MLContext mlContext, ITransformer model, IDataView data)
        {
            IDataView predictions = model.Transform(data);

            Console.WriteLine("===== Use metrics for the data using NDCG@3 =====\n");

            
            ConsoleHelper.EvaluateMetrics(mlContext, predictions);

           
        }

        static void ConsumeModel(MLContext mlContext, ITransformer model, string modelPath, IDataView data)
        {
            Console.WriteLine("===== Save the model =====\n");

            mlContext.Model.Save(model, null, modelPath);

            Console.WriteLine("===== Consume the model =====\n");

            DataViewSchema predictionPipelineSchema;
            ITransformer predictionPipeline = mlContext.Model.Load(modelPath, out predictionPipelineSchema);

            IDataView predictions = predictionPipeline.Transform(data);

            IEnumerable<SearchResultPrediction> searchQueries = mlContext.Data.CreateEnumerable<SearchResultPrediction>(predictions, reuseRowObject: false);
            var firstGroupId = searchQueries.First<SearchResultPrediction>().GroupId;
            IEnumerable<SearchResultPrediction> firstGroupPredictions = searchQueries.Take(100).Where(p => p.GroupId == firstGroupId).OrderByDescending(p => p.Score).ToList();

            ConsoleHelper.PrintScores(firstGroupPredictions);
        }
    }
}
