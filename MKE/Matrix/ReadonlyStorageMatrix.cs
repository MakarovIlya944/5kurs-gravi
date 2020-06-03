namespace MKE.Matrix
{
    public class ReadonlyStorageMatrix
    {
        private double[,] Matrix { get; set; }
        public int RowsCount { get; private set; }
        public int ColumnsCount { get; private set; }
        public ReadonlyStorageMatrix(double[,] matrix, int rows, int columns)
        {
            Matrix = matrix;
            ColumnsCount = columns;
            RowsCount = rows;
        }
        public double this[int i, int j] => Matrix[i, j];
    }
}