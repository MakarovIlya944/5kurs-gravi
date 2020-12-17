using System;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Text;

namespace Mnist
{
    public interface IModel<T> where T : struct, IEquatable<T>, IFormattable
    {
        public void Save(string filename);
        public void Load(string filename);
        public void Train(Data data, int epochs, int batch, T rate, ILossFunction<T> loss);
        public Matrix<T> Predict(Data data);
    }
}
