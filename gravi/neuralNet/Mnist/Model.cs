using Mnist.Functions;
using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using System.Text;
using Microsoft.Extensions.Logging;
using Mnist.Fabrics;
using System.Reflection.Metadata;
using System.IO;
using System.Linq;
using NLog;

namespace Mnist
{
    public class Model : IModel<double>
    {
        public List<Layer> layers;
        public static Logger loggerPredict = LogManager.GetLogger("predict"), loggerTrain = LogManager.GetLogger("train"), logger = LogManager.GetLogger("console");

        public int Deep
        {
            get => layers.Count;
        }

        public int LogEpoch
        {
            get => _logEpoch;
            set => _logEpoch = value;
        }

        public int LogBatch
        {
            get => _logBatch;
            set => _logBatch = value;
        }

        static private int _logEpoch = 5, _logBatch = 50;

        public Model()
        {
            layers = new List<Layer>();
        }

        public Model(int deep, int[] width, double init, double b, int inputSize = 2, int outputSize = 1, bool randomize = false, double center = 0, double offset = 1E-1, double reluCoef = 1E-1, double sigmoidCoef = 1E-1)
        {
            Sigmoid f1 = new Sigmoid(sigmoidCoef);
            ReLU f2 = new ReLU(reluCoef);
            SoftMax f3 = new SoftMax(outputSize);
            if (randomize)
                layers = LayerBuilder.BuildRandom(inputSize, outputSize, width, deep, mOffset: offset, mCenter: center, hidden: f2, input: f2, output: f3);
            //else
            //    layers = LayerBuilder.BuildDense(inputSize, outputSize, width, deep, );
        }

        public Model(int deep, int[] width, double[] init, double[] bias, int inputSize = 2, int outputSize = 1, bool randomize = false, double center = 0, double offset = 1E-1, double reluCoef = 1E-1, double sigmoidCoef = 1E-1)
        {
            Sigmoid f1 = new Sigmoid(sigmoidCoef);
            ReLU f2 = new ReLU(reluCoef);
            SoftMax f3 = new SoftMax(outputSize);
            if (randomize)
                layers = LayerBuilder.BuildRandom(inputSize, outputSize, width, deep, mOffset: offset, mCenter: center, hidden: f2, input: f2, output: f3);
            else
                layers = LayerBuilder.BuildDense(inputSize, outputSize, width, deep, init, bias, hidden: f2, input: f2, output: f3);
        }

        public void Load(string path)
        {
            string filename;
            string[] func;
            int inputSize, width;
            IActivationFunction<double> f;
            Matrix<double> W;
            Vector<double> B;
            int i = 0;
            while (true)
            {
                try
                {
                    filename = Path.Combine(path, $"layer_{i}_f.txt");
                    using (StreamReader file = new StreamReader(filename))
                    {
                        func = file.ReadLine().Split(' ');
                        switch (func[1])
                        {
                            case "SoftMax":
                                f = new SoftMax(Int32.Parse(func[2]));
                                break;
                            case "ReLU":
                                f = new ReLU(Double.Parse(func[2]));
                                break;
                            case "Sigmoid":
                                f = new Sigmoid(Double.Parse(func[2]));
                                break;
                            default:
                                throw new Exception();
                                break;
                        }
                    }
                    filename = Path.Combine(path, $"layer_{i}_w.txt");
                    using (StreamReader file = new StreamReader(filename))
                    {
                        file.ReadLine();
                        W = Matrix<double>.Build.DenseOfRowVectors(
                            file.ReadToEnd().Split('\n').Where(x => x != "").Select(
                                x => Vector<double>.Build.DenseOfArray(x.Split(' ').Where(x => x != "").Select(s => Double.Parse(s)).ToArray()))
                            .ToArray()
                            );
                    }
                    filename = Path.Combine(path, $"layer_{i}_b.txt");
                    using (StreamReader file = new StreamReader(filename))
                    {
                        file.ReadLine();
                        B = Vector<double>.Build.DenseOfArray(
                            file.ReadToEnd().Split('\n').Where(x => x != "").Select(x => Double.Parse(x)).ToArray());
                    }
                    layers.Add(new Layer(W, B, f));
                }
                catch (FileNotFoundException ex)
                {
                    break;
                }
                i++;
            }
        }

        public void Save(string path)
        {
            string filename;
            int maxRows, maxColumns;
            if (!Directory.Exists(path))
                Directory.CreateDirectory(path);

            for (int i = 0; i < layers.Count; i++)
            {
                filename = Path.Combine(path, $"layer_{i}_f.txt");
                using (StreamWriter file = new StreamWriter(filename))
                {
                    file.WriteLine("Name: " + layers[i].activation.ToString());
                }
                filename = Path.Combine(path, $"layer_{i}_w.txt");
                using (StreamWriter file = new StreamWriter(filename))
                {
                    maxColumns = layers[i].matrix.ColumnCount;
                    maxRows = layers[i].matrix.RowCount;
                    file.Write(layers[i].matrix.ToString(maxRows, maxColumns));
                }
                filename = Path.Combine(path, $"layer_{i}_b.txt");
                using (StreamWriter file = new StreamWriter(filename))
                {
                    maxColumns = layers[i].bias.Count;
                    file.Write(layers[i].bias.ToString(maxColumns, 15));
                }
            }
        }

        public Matrix<double> forward(Matrix<double> signal)
        {
            for (int k = 0, n = layers.Count; k < n; k++)
                signal = layers[k].forward(signal);
            return signal;
        }

        public void backPropagation(Matrix<double> signal, Matrix<double> answer, Matrix<double> input, double rate, ILossFunction<double> loss)
        {
            int currentFreeMatrixStorage = 0;
            Matrix<double>[] matrixStorage = new Matrix<double>[2];
            if (layers[layers.Count - 1].activation.GetType() == typeof(SoftMax))
            {
                signal = layers[layers.Count - 1].backPropagation(layers[layers.Count - 2].A, rate, out matrixStorage[currentFreeMatrixStorage]);
            }
            else
            {
                signal = loss.backPropagation(signal, answer);
                //Console.WriteLine($"Current error: \n{error.ToString()}");
                signal = layers[layers.Count - 1].backPropagation(layers[layers.Count - 2].A.Transpose(), signal, rate, out matrixStorage[currentFreeMatrixStorage]);
            }
            currentFreeMatrixStorage = 1;
            //Console.WriteLine("Backward signal through layers");
            for (int k = layers.Count - 2; k > 0; k--)
            {
                //Console.WriteLine($"#{k}");
                signal = layers[k].backPropagation(layers[k - 1].A.Transpose(), signal * layers[k + 1].matrix.Transpose(), rate, out matrixStorage[currentFreeMatrixStorage++]);
                currentFreeMatrixStorage %= 2;
                layers[k + 1].matrix -= matrixStorage[currentFreeMatrixStorage];
            }

            layers[0].backPropagation(input.Transpose(), signal * layers[1].matrix.Transpose(), rate, out matrixStorage[currentFreeMatrixStorage++]);
            currentFreeMatrixStorage %= 2;
            layers[1].matrix -= matrixStorage[currentFreeMatrixStorage++];
            currentFreeMatrixStorage %= 2;
            layers[0].matrix -= matrixStorage[currentFreeMatrixStorage];
        }

        public void Train(Data data, int epoch, int batch, double rate, ILossFunction<double> loss)
        {
            double maxLoss = -1, currentLoss, prevLoss = -1;
            Vector<double> currentLossVector = Vector<double>.Build.Dense(batch, 0);
            Matrix<double> signal, currentLossVectorString = Matrix<double>.Build.DenseOfRowVectors(currentLossVector);
            Matrix<double> answer;

            foreach (var layer in layers)
                layer.InputDataSize = batch;

            for (int i = 0; i < epoch; i++)
            {
                currentLossVector.Clear();
                logger.Error("----------------------------------------------------------------------");
                loggerTrain.Info($"Epoch #{i}/{epoch}");
                logger.Fatal($"Epoch #{i}/{epoch}");
                for (int j = 0; j < data.input; j += batch)
                {
                    loggerTrain.Info($"Batch #{j}/{data.input}");
                    logger.Fatal($"Batch #{j}/{data.input}");

                    answer = data.Answer(j, batch);

                    signal = forward(data.Signal(j, batch));

                    currentLossVector = loss.call(signal, answer);
                    if (layers[layers.Count - 1].activation.GetType() == typeof(SoftMax))
                        layers[layers.Count - 1].activation.SetAnswer(answer);

                    backPropagation(signal, answer, data.Signal(j, batch), rate, loss);

                    currentLoss = currentLossVector.L2Norm();
                    currentLossVectorString = Matrix<double>.Build.DenseOfRowVectors(currentLossVector);
                    if (currentLoss > maxLoss)
                        maxLoss = currentLoss;

                    if (j % (_logBatch * batch) == 0)
                        if (currentLoss > maxLoss)
                        {
                            logger.Fatal($"Previous loss: {prevLoss}");
                            logger.Fatal($"Current loss: {currentLoss}");
                            logger.Fatal($"MaxLoss: {maxLoss}");
                            logger.Fatal(currentLossVectorString.ToString());
                        }
                        else if(currentLoss > prevLoss)
                        {
                            logger.Error($"Previous loss: {prevLoss}");
                            logger.Error($"Current loss: {currentLoss}");
                            logger.Error($"MaxLoss: {maxLoss}");
                            logger.Error(currentLossVectorString.ToString());
                        }
                        else
                        {
                            logger.Info($"Previous loss: {prevLoss}");
                            logger.Info($"Current loss: {currentLoss}");
                            logger.Info($"MaxLoss: {maxLoss}");
                        }


                    loggerTrain.Info($"Previous loss: {prevLoss}");
                    loggerTrain.Info($"Current loss: {currentLoss}");
                    loggerTrain.Info($"MaxLoss: {maxLoss}");

                    prevLoss = currentLoss;
                }
                if (i % _logEpoch == 0)
                {
                    foreach (var layer in layers)
                        layer.InputDataSize = data.input;
                    logger.Fatal(Vector<double>.Build.Dense((forward(data.AllSignal) - data.AllAnswer).EnumerateRows().Select(x => x.L2Norm()).ToArray()).ToString());
                    foreach (var layer in layers)
                        layer.InputDataSize = batch;
                }
            }
            logger.Error("----------------------------------------------------------------------");
        }
        

        public Matrix<double> Predict(Data data)
        {
            return forward(data.AllSignal);
            
        }
    }
}