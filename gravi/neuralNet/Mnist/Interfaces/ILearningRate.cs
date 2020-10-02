using System;
using System.Collections.Generic;
using System.Text;

namespace Mnist
{
    public interface ILearningRate<T>
    {
        public T get(T error);
    }
}
