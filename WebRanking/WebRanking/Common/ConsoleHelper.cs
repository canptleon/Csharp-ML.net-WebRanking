using Microsoft.ML;
using Microsoft.ML.Data;
using WebRanking.DataStructures;
using System;
using System.Collections.Generic;
using System.Linq;

namespace WebRanking.Common
{
    class ConsoleHelper
    {
        public static void EvaluateMetrics(MLContext mlContext, IDataView predictions)
        {
            RankingMetrics metrics = mlContext.Ranking.Evaluate(predictions);

            Console.WriteLine($"DCG: {string.Join(", ", metrics.DiscountedCumulativeGains.Select((d, i) => $"@{i + 1}:{d:F4}").ToArray())}");

            Console.WriteLine($"NDCG: {string.Join(", ", metrics.NormalizedDiscountedCumulativeGains.Select((d, i) => $"@{i + 1}:{d:F4}").ToArray())}\n");
        }

        public static void EvaluateMetrics(MLContext mlContext, IDataView predictions, int truncationLevel)
        {
            if (truncationLevel < 1 || truncationLevel > 10)
            {
                throw new InvalidOperationException("Currently metrics are only supported for 1 to 10 truncation levels.");
            }

            var mlAssembly = typeof(TextLoader).Assembly;
            var rankEvalType = mlAssembly.DefinedTypes.Where(t => t.Name.Contains("RankingEvaluator")).First();

            var evalArgsType = rankEvalType.GetNestedType("Arguments");
            var evalArgs = Activator.CreateInstance(rankEvalType.GetNestedType("Arguments"));

            var dcgLevel = evalArgsType.GetField("DcgTruncationLevel");
            dcgLevel.SetValue(evalArgs, truncationLevel);

            var ctor = rankEvalType.GetConstructors().First();
            var evaluator = ctor.Invoke(new object[] { mlContext, evalArgs });

            var evaluateMethod = rankEvalType.GetMethod("Evaluate");
            RankingMetrics metrics = (RankingMetrics)evaluateMethod.Invoke(evaluator, new object[] { predictions, "Label", "GroupId", "Score" });

            Console.WriteLine($"DCG: {string.Join(", ", metrics.DiscountedCumulativeGains.Select((d, i) => $"@{i + 1}:{d:F4}").ToArray())}");

            Console.WriteLine($"NDCG: {string.Join(", ", metrics.NormalizedDiscountedCumulativeGains.Select((d, i) => $"@{i + 1}:{d:F4}").ToArray())}\n");
        }

        public static void PrintScores(IEnumerable<SearchResultPrediction> predictions)
        {
            foreach (var prediction in predictions)
            {
                Console.WriteLine($"GroupId: {prediction.GroupId}, Score: {prediction.Score}");
            }
        }
    
    }
}
