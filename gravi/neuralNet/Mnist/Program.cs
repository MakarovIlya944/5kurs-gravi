using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;
using Mnist.Functions;
using Mnist.Pictures;
using NLog;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Mnist
{
    public class Program
    {
        private static Logger logger = LogManager.GetLogger("console");

        public static string basePath = @"D:\Projects\Mnist\";
        public static Data allData = MnistConverter.OpenMnist( Path.Combine(basePath, @"data\train-labels.idx1-ubyte"), Path.Combine(basePath, @"data\train-images.idx3-ubyte"), 1);
        public static Data allPredictData = MnistConverter.OpenMnist(Path.Combine(basePath, @"data\t10k-labels.idx1-ubyte"), Path.Combine(basePath, @"data\t10k-images.idx3-ubyte"), 1);
        public static string modelPath = Path.Combine(basePath, @"ready");
        public static int trainDataSize = 60000;
        public static List<int> width;
        public static int trainEpoch = 6;
        public static int trainBatch = 120;
        public static double trainTeachRate = 1E-3;
        public static double trainMatrixRandomCenter = 0;
        public static double trainMatrixRandomOffset = 1E-3;
        public static double trainReLUCoef = 1;
        public static double trainSigmoidCoef = 1;
        public static int logEpoch = 3, logBatch = 5;

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World! Version 5\n");
            //int inputSize = 784;
            //width = new List<int>() { inputSize, inputSize / 2 };
            //modelPath = Path.Combine(basePath, @"ready\newModel");
            //Train();
            //Predict();
            //TrainManyModels(modelPath);
            PredictManyModels(@"D:\Projects\Mnist\NeuralNet\Ready\Models");
            //modelPath = @"D:\Projects\Mnist\NeuralNet\Ready\Models\0_len_hidden\model_2";
            Console.WriteLine("Good bye World!");
            Console.ReadLine();
        }

        private static void SavePredict(List<int[]> res, string path)
        {
            double all, truly, falsly, trfa;
            using (StreamWriter file =
new StreamWriter(Path.Combine(path, $"predict.txt")))
            {
                file.WriteLine("#: t/f\t\tt%\t\tf%\t\tt/f");
                for (int j = 0; j < 10; j++)
                {
                    all = res[0][j] + res[1][j];
                    truly = res[0][j] / all;
                    falsly = res[1][j] / all;
                    trfa = res[0][j] / (double)res[1][j];
                    file.WriteLine(String.Format("{0}: {1:00}/{2:00}\t\t{3:0.00}%\t\t{4:0.00}%\t\t{5:0.00}", j, res[0][j], res[1][j], truly * 100, falsly * 100, trfa));
                }
            }
        }

        public static void PredictManyModels(string path)
        {
            string[] models = new string[3] { "model_0", "model_1", "model_2" };
            string model;
            List<int[]> res;

            model = Path.Combine(path, "0_len_hidden");
            for (int i = 0; i < 3; i++)
            {
                modelPath = Path.Combine(model, models[i]);
                res = Predict();
                SavePredict(res, modelPath);
            }

            model = Path.Combine(path, "1_capacity_hidden");
            for (int i = 0; i < 3; i++)
            {
                modelPath = Path.Combine(model, models[i]);
                res = Predict();
                SavePredict(res, modelPath);
            }

            model = Path.Combine(path, "2_size_batch");
            for (int i = 0; i < 3; i++)
            {
                modelPath = Path.Combine(model, models[i]);
                res = Predict();
                SavePredict(res, modelPath);
            }

            model = Path.Combine(path, "3_size_batch");
            for (int i = 0; i < 3; i++)
            {
                modelPath = Path.Combine(model, models[i]);
                res = Predict();
                SavePredict(res, modelPath);
            }
        }

        public static void TrainManyModels(string basePath)
        {
            int inputSize = 28 * 28;
            string path = basePath;
            List<double> times = new List<double>();


            List<int>[] widths =
            {
                new List<int>() { inputSize },
                new List<int>() { inputSize, inputSize / 2 },
                new List<int>() { inputSize, inputSize / 2, inputSize / 4 }
            };
            path = Path.Combine(basePath, $"0_len_hidden");
            if (!Directory.Exists(path))
                Directory.CreateDirectory(path);
            for (int i = 0; i < widths.Length; i++)
            {
                modelPath = Path.Combine(path, $"model_{i}");
                width = widths[i];
                times.Add(Train());
            }
            SaveTime(path, times);

            widths = new List<int>[3]
            {
                new List<int>() { inputSize / 2, inputSize / 4},
                new List<int>() { inputSize, inputSize / 2 },
                new List<int>() { inputSize * 2, inputSize }
            };
            path = Path.Combine(basePath, $"1_capacity_hidden");
            if (!Directory.Exists(path))
                Directory.CreateDirectory(path);
            for (int i = 0; i < widths.Length; i++)
            {
                modelPath = Path.Combine(path, $"model_{i}");
                width = widths[i];
                times.Add(Train());
            }
            SaveTime(path, times);

            path = Path.Combine(basePath, $"2_size_batch");
            if (!Directory.Exists(path))
                Directory.CreateDirectory(path);
            width = widths[0];
            int[] batches = new int[3] { trainBatch / 2, trainBatch, trainBatch * 2 };
            for (int i = 0; i < batches.Length; i++)
            {
                modelPath = Path.Combine(path, $"model_{i}");
                trainBatch = batches[i];
                times.Add(Train());
            }
            SaveTime(path, times);

            trainBatch /= 2;
            path = Path.Combine(basePath, $"3_size_batch");
            if (!Directory.Exists(path))
                Directory.CreateDirectory(path);
            int[] epoches = new int[3] { 6, 12, 24 };
            for (int i = 0; i < epoches.Length; i++)
            {
                modelPath = Path.Combine(path, $"model_{i}");
                trainEpoch = epoches[i];
                times.Add(Train());
            }
            SaveTime(path, times);
        }

        private static void SaveTime(string path, List<double> times)
        {
            using (System.IO.StreamWriter file =
new System.IO.StreamWriter(Path.Combine(path, $"times.txt")))
            {
                int i = 0;
                foreach (var t in times)
                {
                    file.Write($"train model_{i}: {t}ms");
                }
                times.Clear();
            }
        }

        public static int PredictedIndex(Vector<double> v)
        {
            double max = v[0];
            int a = 0;
            v.MapIndexed((i, x) => {
                if (max < x) { a = i; max = x; }; 
                return x; });
            return a;
        }

        public static List<int[]> Predict()
        {
            Model m = new Model();
            m.Load(modelPath);
            Data data = allPredictData;
            foreach (var item in m.layers)
            {
                item.InputDataSize = data.InputDataSize;
            }
            Matrix<double> pred = m.Predict(data), ans = data.AllAnswer;
            Vector<double> l2error = Vector<double>.Build.Dense((pred - ans).EnumerateRows().Select(x => x.L2Norm()).ToArray());

            logger.Error("Train:");
            logger.Error($"L2 errors: {l2error.ToString()}");
            logger.Error($"Max error: {l2error.Max()}:");

            int[] right = new int[10], falsive = new int[10];
            int a, b;
            for (int i = 0; i < pred.RowCount; i++)
            {
                a = PredictedIndex(ans.Row(i));
                b = PredictedIndex(pred.Row(i));
                if (a == b) right[a]++;
                else falsive[a]++;
            }
            logger.Error("i:\ttrue/false\t\ttrue/all\t\tfalse/all\t\ttrue/false");
            double truly, falsly, trfa, all;
            for (int i = 0; i < 10; i++)
            {
                all = right[i] + falsive[i];
                truly = right[i] / all;
                falsly = falsive[i] / all;
                trfa = right[i] / (double)falsive[i];
                logger.Error(
                    String.Format("{0}: {1:00}/{2:00}\t\t{3:0.00}%\t\t{4:0.00}%\t\t{5:0.00}", i, right[i], falsive[i], truly*100, falsly * 100, trfa));
            }
            return new List<int[]>() { right, falsive };
        }

        public static Vector<double> Predict(byte[] image)
        {
            Model m = new Model();
            m.Load(modelPath);
            Data data = new Data(new Vector<double>[1] { Vector<double>.Build.DenseOfArray(image.Select(x => 255.0-(double)(int)x).ToArray()) }, new Vector<double>[0]);
            MnistConverter.SavePicture(data, 0);
            foreach (var item in m.layers)
            {
                item.InputDataSize = data.InputDataSize;
            }
            return m.Predict(data).Row(0);
        }

        public static double Train()
        {
            var startTime = System.Diagnostics.Stopwatch.StartNew();
            Data data = allData.Take(trainDataSize);

            int inputSize = 28 * 28, outputSize = 10, deep;

            deep = width.Count + 1;
            //double[] init = new double[5] { 1, 1, 1, 1, 1 }, bias = new double[5] { 1, 1, 1, 1, 1 };
            ILossFunction<double> loss = new LogLoss();

            Model m = new Model(deep, width.ToArray(), 1, 1, inputSize, outputSize, true, trainMatrixRandomCenter, trainMatrixRandomOffset, trainReLUCoef, trainSigmoidCoef);

            m.LogEpoch = logEpoch;
            m.LogBatch = logBatch;

            m.Train(data, trainEpoch, trainBatch, trainTeachRate, loss);
            m.Save(modelPath);
            startTime.Stop();
            return startTime.ElapsedMilliseconds;
        }
    }
}
