using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using System.Text;
using System.Runtime.InteropServices.WindowsRuntime;
using Mnist.Functions;
using System.Linq;

namespace Mnist
{
    public class Layer : ILayer<double>
    {
        public int width;

        public IActivationFunction<double> activation;

        public Matrix<double> matrix;
        public Vector<double> bias;

        public int InputDataSize {
            set => SetUp(value);
            get => B.ColumnCount;
        }

        private Matrix<double> B; // bias as matrix
        private Vector<double> zVector; // v * matrix + bias
        private Matrix<double> zMatrix; // v * matrix + bias
        private Vector<double> aVector; // f(zVector)
        private Matrix<double> aMatrix; // f(zMatrix)

        public Matrix<double> Z { set => zMatrix = value; }
        public Matrix<double> A { set => aMatrix = value; get => aMatrix; }

        //public List<INode<double>> nodes;  // for non full-connected

        public ILayerInfo<double> info;

        public Layer(int nodesCount, int inputVectorSize, double init, double b, IActivationFunction<double> activation)
        {
            width = nodesCount;
            this.activation = activation;
            matrix = Matrix<double>.Build.Dense(inputVectorSize, nodesCount, init);
            bias = Vector<double>.Build.Dense(nodesCount, b);

            //nodes = new List<INode<double>>(nodesCount); // for non full-connected
            //for (int i = 0; i < nodesCount; i++)
            //    nodes.Add(new Node<double>(activation));
        }

        public Layer(Matrix<double> m, Vector<double> b, IActivationFunction<double> activation)
        {
            width = b.Count;
            this.activation = activation;
            matrix = m;
            bias = b;

            //nodes = new List<INode<double>>(nodesCount); // for non full-connected
            //for (int i = 0; i < nodesCount; i++)
            //    nodes.Add(new Node<double>(activation));
        }

        public void RandomMatrix(double center, double offset)
        {
            Random r = new Random();
            for (int i = 0; i < matrix.ColumnCount; i++)
                for (int j = 0; j < matrix.RowCount; j++)
                    matrix[j, i] = (r.NextDouble() - 0.5) * offset * 2 + center;
        }

        public void RandomBias(double center, double offset)
        {
            Random r = new Random();
            for (int i = 0; i < bias.Count; i++)
                bias[i] = (r.NextDouble() - 0.5) * offset * 2 + center;
        }

        private void SetUp(int size)
        {
            Vector<double>[] tempBias = new Vector<double>[size];
            for (int i = 0; i < size; i++)
                tempBias[i] = Vector<double>.Build.DenseOfVector(bias);
            B = Matrix<double>.Build.DenseOfRowVectors(tempBias);
        }

        public Vector<double> forward(Vector<double> input)
        {
            zVector = input * matrix + bias;
            return activation.call(zVector);
        }

        public Matrix<double> forward(Matrix<double> input)
        {
            zMatrix = input * matrix + B;
            //Console.WriteLine($"Z: \n{zMatrix.ToString()}");
            aMatrix = activation.call(zMatrix);
            //Console.WriteLine($"A: \n{aMatrix.ToString()}");
            return aMatrix;
        }

        public Vector<double> backPropagation(Vector<double> input, double rate)
        {
            //Matrix<double> ret;
            throw new NotImplementedException();
        }

        public Matrix<double> backPropagation(Matrix<double> prevW, Matrix<double> prevDelta, double rate, out Matrix<double> derivative)
        {
            Matrix<double> fz = activation.backPropagation(zMatrix);
            //Console.WriteLine($"f(Z): \n{fz.ToString()}");
            Matrix<double> delta = prevDelta.Map2((x,y) => x*y*rate, fz); // мне кажется, где-то здесь ошибка
            //Console.WriteLine($"delta: \n{delta.ToString()}");
            //Console.WriteLine($"W: \n{matrix.ToString()}");
            derivative = prevW * delta;
            //Console.WriteLine($"W: \n{matrix.ToString()}");
            return delta;
        }

        public Matrix<double> backPropagation(Matrix<double> x, double rate, out Matrix<double> derivative)
        {
            // aMatrix - S
            // input - x
            Matrix<double> delta = aMatrix.MapIndexed((i, j, v) => j == ((SoftMax)activation).rightIndecies[i] ? v - 1 : v);
            derivative = x.Transpose() * delta * rate;
            return delta;
        }
    }
}
