using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using System.Linq;
using System.Drawing;

namespace Mnist.Pictures
{
    static public class MnistConverter
    {
        public static void SavePicture(Data d, int i, string path = @"./image.bmp")
        {
            var bitmap = new Bitmap(28, 28);

            for (int x = 0; x < bitmap.Width; x++)
                for (int y = 0; y < bitmap.Height; y++)
                    bitmap.SetPixel(x, y, Color.FromArgb((int)d.AllSignal[i, y * bitmap.Width + x], 0, 0, 0));

            bitmap.Save(path);
        }

        public static Data OpenMnist(string filenamelabel, string filenameimage, double percent = 1.0)
        {
            byte[] byteLabels = File.ReadAllBytes(filenamelabel), byteImages = File.ReadAllBytes(filenameimage);

            int len, output = 10;

            len = BitConverter.ToInt32(byteLabels.Skip(4).Take(4).Reverse().ToArray());
            len = (int)(len * percent);

            List<Vector<double>> listImages = new List<Vector<double>>(len);
            List<Vector<double>> listLabels = new List<Vector<double>>(len);

            Vector<double> v = Vector<double>.Build.Dense(output, 1E-16);
            for (int offset = 8, i = 0; i < len; i++)
            {
                v[byteLabels[offset++]] = 1;
                listLabels.Add(v);
                v = Vector<double>.Build.Dense(output, 1E-16);
            }

            int rows = BitConverter.ToInt32(byteImages.Skip(8).Take(4).Reverse().ToArray());
            int columns = BitConverter.ToInt32(byteImages.Skip(12).Take(4).Reverse().ToArray());
            int input = rows * columns;

            for (int offset = 16, i = 0; i < len; offset += input, i++)
                listImages.Add(Vector<double>.Build.DenseOfEnumerable(
                    byteImages
                    .Skip(offset).Take(input)
                    .Select(x => (double)(int)x)));

            return new Data(listImages.ToArray(), listLabels.ToArray());
        }
    }
}
