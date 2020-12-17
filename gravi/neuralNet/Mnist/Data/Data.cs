using MathNet.Numerics.LinearAlgebra;
using System.Linq;

namespace Mnist
{
    public class Data
    {
        public Vector<double>[] signal;
        public Vector<double>[] answer;

        public Matrix<double> AllSignal { get => Matrix<double>.Build.DenseOfRowVectors(signal); }
        public Matrix<double> AllAnswer { get => Matrix<double>.Build.DenseOfRowVectors(answer); }

        public int InputDataSize { get => input; }
        public int input, output;

        public Data(Vector<double>[] signal, Vector<double>[] answer)
        {
            this.signal = signal;
            this.answer = answer;
            this.input = signal.Length;
            this.output = answer.Length;
        }

        public Data(int input, int output)
        {
            this.input = input;
            this.output = output;
        }

        public Data this[int index]
        {
            get => new Data(new Vector<double>[1] { signal[index] }, new Vector<double>[1] { answer[index] });
        }

        public Data Take(int i)
        {
            return new Data(signal.Take(i).ToArray(), answer.Take(i).ToArray());
        }

        public Data Skip(int i)
        {
            return new Data(signal.Skip(i).ToArray(), answer.Skip(i).ToArray());
        }

        public Matrix<double> Signal(int offset, int length)
        {
            return Matrix<double>.Build.DenseOfRowVectors(signal.Skip(offset).Take(length));
        }

        public Matrix<double> Answer(int offset, int length)
        {
            return Matrix<double>.Build.DenseOfRowVectors(answer.Skip(offset).Take(length));
        }
    }
}
