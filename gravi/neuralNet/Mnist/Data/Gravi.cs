using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using System.Linq;
using System.Drawing;

namespace Mnist.Gravi
{
    static public class GraviConverter
    {
        public static Data OpenFolder(string path)
        {
            List<Vector<double>> signals = new List<Vector<double>>();
            List<Vector<double>> answers = new List<Vector<double>>();
            int i = 0;
            string file = Path.Combine(path, "\\" + i.ToString());
            while (File.Exists(file + "_in"))
            {
                signals.Add(Vector<double>.Build.DenseOfEnumerable(File.ReadAllLines(file + "_in").Select(x => Double.Parse(x))));
                answers.Add(Vector<double>.Build.DenseOfEnumerable(File.ReadAllLines(file + "_out").Select(x => Double.Parse(x))));
                i++;
            }
            
            return new Data(signals.ToArray(), answers.ToArray());
        }
    }
}
