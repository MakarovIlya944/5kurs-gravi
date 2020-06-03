using System;

namespace MKE.Extenstion
{
    public static class TupleExtenstion
    {
        public static int LikeBinaryToInt(this ValueTuple<int, int, int> tuple)
        {
            return tuple.Item1 * 2 * 2 + tuple.Item2 * 2 + tuple.Item3;
        }
        public static int LikeBinaryToInt(this ValueTuple<int, int> tuple)
        {
            return tuple.Item1 * 2 + tuple.Item2;
        }
    }
}