using Mnist.Functions;
using System;
using System.Collections.Generic;
using System.Text;

namespace Mnist.Fabrics
{
    static public class LayerBuilder
    {
        static public Layer BuildDense(int inputSize, int outputSize, double matrixInit, double biasInit, IActivationFunction<double> f)
        {
            return new Layer(outputSize, inputSize, matrixInit, biasInit, f);
        }

        static public Layer BuildRandom(int inputSize, int outputSize, IActivationFunction<double> f, double mCenter, double mOffset, double bCenter, double bOffset)
        {
            Layer l = new Layer(outputSize, inputSize, 0, 0, f);
            l.RandomMatrix(mCenter, mOffset);
            l.RandomBias(bCenter, bOffset);
            return l;
        }

        static public List<Layer> BuildDense(int inputSize, int outputSize, int hiddenSize, int deep, double matrixInit, double biasInit, IActivationFunction<double> hidden, IActivationFunction<double> input = null, IActivationFunction<double> output = null)
        {
            List<Layer> layers = new List<Layer>(deep);

            input ??= new ReLU();
            output ??= new ReLU();

            if (deep < 1)
                throw new Exception("Too few layers!");
            else if(deep == 1)
            {
                layers.Add(new Layer(inputSize, outputSize, matrixInit, biasInit, hidden));
            }
            else
            {
                layers.Add(new Layer(hiddenSize, inputSize, matrixInit, biasInit, input));
                for (int i = 1; i < deep - 1; i++)
                    layers.Add(new Layer(hiddenSize, hiddenSize, matrixInit, biasInit, hidden));
                layers.Add(new Layer(outputSize, hiddenSize, matrixInit, biasInit, output));
            }
            return layers;
        }

        static public List<Layer> BuildRandom(int inputSize, int outputSize, int hiddenSize, int deep, double mCenter = 0, double mOffset = 1, double bCenter = 0, double bOffset = 1, IActivationFunction<double> hidden = null, IActivationFunction<double> input = null, IActivationFunction<double> output = null)
        {
            List<Layer> layers = new List<Layer>(deep);

            input ??= new ReLU();
            output ??= new ReLU();
            hidden ??= new ReLU();

            if (deep < 1)
                throw new Exception("Too few layers!");
            else if (deep == 1)
            {
                layers.Add(BuildRandom(inputSize, outputSize, output, mCenter, mOffset, bCenter, bOffset));
            }
            else
            {
                layers.Add(BuildRandom(inputSize, hiddenSize, input, mCenter, mOffset, bCenter, bOffset));
                for (int i = 1; i < deep - 1; i++)
                    layers.Add(BuildRandom(hiddenSize, hiddenSize, hidden, mCenter, mOffset, bCenter, bOffset));
                layers.Add(BuildRandom(hiddenSize, outputSize, output, mCenter, mOffset, bCenter, bOffset));
            }
            return layers;
        }

        static public List<Layer> BuildDense(int inputSize, int outputSize, int[] hiddenSize, int deep, double[] matrixInit, double[] biasInit, IActivationFunction<double> hidden, IActivationFunction<double> input = null, IActivationFunction<double> output = null)
        {
            List<Layer> layers = new List<Layer>(deep);

            input ??= new ReLU();
            output ??= new ReLU();

            if (deep < 1)
                throw new Exception("Too few layers!");
            else if (deep == 1)
            {
                layers.Add(new Layer(inputSize, outputSize, matrixInit[0], biasInit[0], hidden));
            }
            else
            {
                layers.Add(new Layer(hiddenSize[0], inputSize, matrixInit[0], biasInit[0], input));
                for (int i = 1; i < deep - 1; i++)
                    layers.Add(new Layer(hiddenSize[i], hiddenSize[i-1], matrixInit[i], biasInit[i], hidden));
                layers.Add(new Layer(outputSize, hiddenSize[deep - 2], matrixInit[deep - 1], biasInit[deep - 1], output));
            }
            return layers;
        }

        static public List<Layer> BuildRandom(int inputSize, int outputSize, int[] hiddenSize, int deep, double mCenter = 0, double mOffset = 1, double bCenter = 0, double bOffset = 1, IActivationFunction<double> hidden = null, IActivationFunction<double> input = null, IActivationFunction<double> output = null)
        {
            List<Layer> layers = new List<Layer>(deep);

            input ??= new ReLU();
            output ??= new ReLU();
            hidden ??= new ReLU();

            if (deep < 1)
                throw new Exception("Too few layers!");
            else if (deep == 1)
            {
                layers.Add(BuildRandom(inputSize, outputSize, hidden, mCenter, mOffset, bCenter, bOffset));
            }
            else
            {
                layers.Add(BuildRandom(inputSize, hiddenSize[0], input, mCenter, mOffset, bCenter, bOffset));
                for (int i = 1; i < deep - 1; i++)
                    layers.Add(BuildRandom(hiddenSize[i-1], hiddenSize[i], hidden, mCenter, mOffset, bCenter, bOffset));
                layers.Add(BuildRandom(hiddenSize[deep - 2], outputSize, output, mCenter, mOffset, bCenter, bOffset));
            }
            return layers;
        }
    }
}
